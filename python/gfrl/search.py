"""Batched information-set search. Only public observations enter inference.

OracleBelief is an evaluation-only diagnostic that cheats on root hands.

Candidate paths contain externally selected choices (forced singleton stages are
handled by the engine). Random rollout actions and belief draws use an isolated
Torch stream seeded from this agent's NumPy RNG. Real-play fallback is pi_theta.
"""
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from . import SearchEnv, VecEnv
from .actor import Actor, ActResult, PREFIX_FIELDS, synchronize
from .belief import IndependentBelief, belief_from_checkpoint
from .belief_metrics import Baseline, fit_baselines, gift_probabilities
from .evaluation import _policy_rng
from .model import OBS_FIELDS
from .transfer import tensor_batch


@dataclass(frozen=True)
class SearchConfig:
    worlds: int = 16
    candidates: int = 16
    candidate_samples: int = 64
    depth: int = 10  # completed external decisions AFTER the root
    alpha: float = .002
    beta: float = .02
    select: str = 'sample'
    max_attempts: int = 100
    leaf_view: str = 'actor'
    max_inference_batch: int = 0  # 0: no splitting
    skip_forced_inference: bool = False
    skip_single_candidate: bool = False
    search_cache_device: str = 'auto'
    search_cache_mb: int = 256
    root_batch_size: int = 0  # distribution preserving; changes RNG ordering
    search_response: bool = False

    def __post_init__(self):
        if min(self.worlds, self.candidates, self.candidate_samples, self.max_attempts) <= 0 or self.depth < 0:
            raise ValueError('positive search budgets and nonnegative depth required')
        if not np.isfinite([self.alpha, self.beta]).all() or min(self.alpha, self.beta) < 0 or self.alpha+self.beta <= 0:
            raise ValueError('finite nonnegative alpha/beta with positive sum required')
        if self.leaf_view not in ('actor', 'searcher'):
            raise ValueError('leaf_view must be actor or searcher')
        if self.max_inference_batch < 0:
            raise ValueError('max_inference_batch must be nonnegative')
        if self.search_cache_device not in ('auto', 'host', 'device'):
            raise ValueError('search_cache_device must be auto, host or device')
        if self.search_cache_mb < 0 or self.root_batch_size < 0:
            raise ValueError('cache budget and root batch size must be nonnegative')
        if self.select not in ('sample', 'argmax'):
            raise ValueError('select must be sample or argmax')


def search_distribution(q, log_rho, log_pi, alpha=.002, beta=.02):
    """Ataraxos distribution, normalizing both priors on surviving candidates."""
    q, lr, lp = (np.asarray(x, dtype=np.float64) for x in (q, log_rho, log_pi))
    if q.ndim != 1 or not len(q) or q.shape != lr.shape or q.shape != lp.shape:
        raise ValueError('matching nonempty vectors required')
    if (not np.isfinite([q, lr, lp]).all() or not np.isfinite([alpha, beta]).all()
            or min(alpha, beta) < 0 or alpha+beta <= 0):
        raise ValueError('finite values and nonnegative regularization required')
    def normalize(x):
        return x - (np.log(np.exp(x-x.max()).sum())+x.max())
    logits = (q+alpha*normalize(lr)+beta*normalize(lp))/(alpha+beta)
    weights = np.exp(logits-logits.max())
    return weights/weights.sum()


@dataclass(frozen=True)
class OracleBelief:
    """Evaluation-only diagnostic: use privileged root hand labels, not inference."""


def load_search_belief(spec, fit=None, device='cpu', *, temperature=1., mix=0.):
    from .search_belief import validate_variants, wrap_belief
    validate_variants(temperature, mix)
    if spec in ('oracle', 'b1', 'uniform') and (temperature != 1 or mix):
        raise ValueError('belief variants require an autoregressive checkpoint')
    if spec == 'oracle':
        return OracleBelief()
    if spec == 'b1':
        if fit is None:
            raise ValueError('--fit is required for b1')
        return fit_baselines(torch.load(fit, map_location='cpu', weights_only=False))['B1']
    if spec == 'uniform':
        # B0 gift rates with a uniform count prior when no fitting data supplied.
        pn = (fit_baselines(torch.load(fit, map_location='cpu', weights_only=False))['B0'].count_probs
              if fit else torch.full((19,), 1/19, dtype=torch.float64))
        return Baseline(pn, gift_probabilities())
    model = belief_from_checkpoint(torch.load(spec, map_location='cpu', weights_only=False)).to(device).eval().requires_grad_(False)
    baseline = fit_baselines(torch.load(fit, map_location='cpu', weights_only=False))['B1'] if mix and fit else None
    return wrap_belief(model, temperature, mix, baseline)


_SUPPORT = {-1: 'inside_decision', 1: 'invalid_viewer', 2: 'unsupported_players',
            3: 'not_main', 4: 'not_actor', 5: 'pending_work', 6: 'reserved_gifts',
            7: 'replay_proof', 8: 'hidden_status', 9: 'hidden_confusion', 10: 'hidden_own_model',
            11: 'response_special', 12: 'response_random', 13: 'response_history'}


def response_roots(env, rows):
    """Public CLS phase (MatchPhase::response + 1), never hidden state."""
    phase = env.obs_schema()['cat']['phase']['index']
    return env.obs_cat[np.asarray(rows, np.int32), 0, phase] == 2


def search_support(env, rows, search_response=False):
    """Engine support includes public chance-conditioned response launches.

    Preserve the original main-only reason precedence when disabled.
    """
    support = env.determinize_support(rows).copy()
    if not search_response:
        support[response_roots(env, rows) & (support != -1)] = 3  # historically not_main
    return support


class _SearchActor(Actor):
    """Bound search inference workspace with a budgeted decision encoding cache.

    Heads are collected before a single sampling call, preserving RNG ordering
    regardless of chunk size. Ordinary Actor/training/evaluation are unchanged.
    """
    def __init__(self, *args, max_batch=0, cache_device="auto", cache_mb=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_batch = max_batch
        self.cache_device = cache_device
        self.cache_bytes = cache_mb * 1024**2

    def _cache_location(self, encoded):
        size = self.env.num_envs * encoded[0].numel() * encoded.element_size()
        budget = self.cache_bytes
        if self.device.type == 'cuda' and self.cache_device == 'auto':
            free, _ = torch.cuda.mem_get_info(self.device)
            budget = min(budget, free // 4)
        return (self.device if self.cache_device != 'host' and size <= budget
                else torch.device('cpu'))

    @torch.inference_mode()
    def act(self, rows=None, *, observation_groups=None, **kwargs):
        if not self.max_batch:
            return super().act(rows, observation_groups=observation_groups, **kwargs)
        env = self.env
        active = np.flatnonzero(env.stage >= 0) if rows is None else np.asarray(rows)
        fresh = active[env.substep[active] == 0]
        if np.any(~self._cached[active] & (env.substep[active] != 0)):
            raise RuntimeError('attach Actor at the start of a decision (substep == 0)')
        timings = dict(transfer=0., encoder=0., heads=0.)
        def clock():
            if self.profile:
                synchronize(self.device)
            return perf_counter()
        encode_rows, inverse = fresh, np.arange(len(fresh))
        if observation_groups is not None and len(fresh):
            _, first, inverse = np.unique(np.asarray(observation_groups)[fresh], return_index=True, return_inverse=True)
            encode_rows = fresh[first]
        # Expand duplicate encodings a chunk at a time into the bounded cache.
        for lo in range(0, len(encode_rows), self.max_batch):
            rr = encode_rows[lo:lo+self.max_batch]
            since = clock()
            obs = tensor_batch({k: getattr(env, k)[rr] for k in OBS_FIELDS}, self.device)
            timings['transfer'] += clock()-since
            since = clock()
            encoded = self.model.encode(obs)
            if self.token_cache is None:
                self.token_cache = torch.empty((env.num_envs, *encoded.shape[1:]), dtype=encoded.dtype,
                                               device=self._cache_location(encoded))
            encoded = encoded.to(self.token_cache.device)
            if observation_groups is None:
                dst, src = rr, np.arange(len(rr))
            else:
                take = (inverse >= lo) & (inverse < lo+len(rr))
                dst, src = fresh[take], inverse[take]-lo
            # Gather/copy only this chunk, on the cache's device. Duplicate
            # observations have exactly the same encoded bits.
            location = self.token_cache.device
            self.token_cache.index_copy_(0, torch.as_tensor(dst, device=location),
                                        encoded.index_select(0, torch.as_tensor(src, device=location)))
            timings['encoder'] += clock()-since
            del obs, encoded
        self._cached[fresh] = True
        lp_parts, value_parts = [], []
        for lo in range(0, len(active), self.max_batch):
            rr = active[lo:lo+self.max_batch]
            since = clock()
            tokens = self.token_cache.index_select(
                0, torch.as_tensor(rr, device=self.token_cache.device)).to(self.device)
            prefix = tensor_batch({k: getattr(env, k)[rr] for k in PREFIX_FIELDS}, self.device)
            timings['transfer'] += clock()-since
            since = clock()
            lp, vl = self.model.heads(tokens, **prefix)
            lp_parts.append(lp.float())
            value_parts.append(vl.float())
            timings['heads'] += clock()-since
            del tokens, prefix
        choices = np.zeros(env.num_envs, np.int32)
        log_prob = np.zeros(env.num_envs, np.float32)
        values = np.zeros((env.num_envs, 3), np.float32)
        all_lp = np.zeros((env.num_envs, 21), np.float32)
        if len(active):
            lp, vl = torch.cat(lp_parts), torch.cat(value_parts)
            action = torch.multinomial(lp.exp(), 1).squeeze(1) if self.sample else lp.argmax(-1)
            choices[active] = action.cpu().numpy()
            log_prob[active] = lp.gather(1, action[:, None]).squeeze(1).cpu().numpy()
            values[active], all_lp[active] = vl.cpu().numpy(), lp.cpu().numpy()
        self.last_profile = timings
        return ActResult(choices, log_prob, values, all_lp)


class SearchAgent:
    def __init__(self, model, belief, device='cpu', *, config=SearchConfig(), seed=0, num_threads=1, profile=False):
        self.device = torch.device(device)
        self.model = model.to(device).eval()
        self.belief = belief
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.num_threads = num_threads
        self.actor = Actor(model, VecEnv(1, 1, eval_pairs=0), device)
        self.counts = Counter({k: 0 for k in (
            'decisions', 'searched', 'searched_main', 'searched_response', 'skip_response_disabled',
            'selected', 'invalid_samples', 'resamples',
            'search_calls', 'search_batches', 'rollout_roots', 'rollout_worlds',
            'singleton_roots', 'singleton_validated_worlds', 'forced_inference_rows_skipped',
            'candidate_paths', 'batch_roots', 'max_batch_roots',
            'cache_cpu_actors', 'cache_cuda_actors', 'cache_mps_actors', 'cache_peak_bytes',
            'exhausted_worlds', 'invalid_loads', 'invalid_candidate_worlds', 'invalid_oracle_assignments',
            'real_mismatches', 'policy_disagreements', 'fallback_no_candidates',
            'depth_leaves', 'depth_leaves_searcher', 'depth_leaves_opponent', 'terminal_leaves')})
        self.seconds = Counter()
        self.pending = {}
        self.last_search = []
        self.env = None
        self.profile = profile
        self._workspaces = {}
        self.last_candidates = {}
        self.last_candidate_diagnostics = {}
        self.world_load_counts = Counter()
        self.last_policy_paths = {}

    def statistics(self):
        n = self.counts['searched']
        decisions = self.counts['decisions']
        return dict(counts=dict(self.counts), searched_fraction=n/decisions if decisions else 0,
                    seconds_per_search_by_root={kind: self.seconds['total_'+kind]/self.counts['searched_'+kind]
                                              if self.counts['searched_'+kind] else 0
                                              for kind in ('main', 'response')},
                    seconds=dict(self.seconds), seconds_per_search={k: v/n if n else 0 for k, v in self.seconds.items()},
                    seconds_per_call={k: v/self.counts['search_calls'] if self.counts['search_calls'] else 0
                                      for k, v in self.seconds.items()},
                    policy_disagreement_fraction=(self.counts['policy_disagreements']/self.counts['selected']
                                                  if self.counts['selected'] else 0),
                    roots_per_search_call=(n/self.counts['search_calls'] if self.counts['search_calls'] else 0),
                    disagreement_definition='selected path differs from the first full pi_theta candidate draw (before top-k filtering)')

    def _time(self):
        synchronize(self.device)
        return perf_counter()

    def _world_env(self, src, rows, assignments, seeds, depth, role=None):
        start = perf_counter()
        key = (len(rows), src.obs_schema()['K'], depth)
        cached = self._workspaces.get(role) if role else None
        if cached is not None and cached[0] == key:
            env = cached[1]
        else:
            env = SearchEnv(key[0], self.num_threads, history_len=key[1], max_decisions=depth)
            if role:
                self._workspaces[role] = (key, env)
        self.seconds['world_workspace'] += perf_counter()-start
        start = perf_counter()
        status = env.load_worlds(src, np.asarray(rows, np.int32), np.asarray(assignments, np.int32), np.asarray(seeds, np.uint64))
        self.seconds['world_load'] += perf_counter()-start
        self.counts['invalid_loads'] += int((status != 0).sum())
        names = {0: 'ok', 1: 'invalid_assignment', 2: 'unsupported_root', 3: 'empty_own_prior',
                 4: 'observation_mismatch', 5: 'decision_mismatch', 6: 'legality_mismatch'}
        for value, count in zip(*np.unique(status, return_counts=True)):
            self.world_load_counts[(role or 'other') + ':' + names[int(value)]] += int(count)
        return env, status.copy()

    def _belief_samples(self, src, rows, belief=None):
        """Chunk deterministic belief work; retain the original sampling order."""
        belief = self.belief if belief is None else belief
        cap = self.config.max_inference_batch
        if not cap:
            obs = tensor_batch({k: getattr(src, k)[rows] for k in OBS_FIELDS}, self.device)
            return belief.sample(obs, 1)[:, 0].cpu().numpy().astype(np.int32)
        chunks, count_probs, card_probs = [], [], []
        independent = isinstance(belief, IndependentBelief)
        for lo in range(0, len(rows), cap):
            rr = rows[lo:lo+cap]
            obs = tensor_batch({k: getattr(src, k)[rr] for k in OBS_FIELDS}, self.device)
            if independent:
                pn, pk = belief._log_probs(obs)
                count_probs.append(pn.exp())
                card_probs.append(pk.exp())
            else:
                cls, memory, padding = belief.memory(obs)
                count_probs.append(belief.count_head(cls).float().softmax(-1))
                chunks.append((memory.cpu(), padding.cpu()))
                del cls, memory, padding
            del obs
        counts = torch.multinomial(torch.cat(count_probs), 1, replacement=True).flatten()
        result = torch.zeros(len(rows), 297, dtype=torch.long, device=self.device)
        if independent:
            cards = torch.multinomial(torch.cat(card_probs), 18, replacement=True)+1
            result.scatter_add_(1, cards, (torch.arange(18, device=self.device) < counts[:, None]).long())
        else:
            # memory() packs to the batch's maximum valid width. Restore that
            # width across chunks, matching the unsplit decoder's padding.
            width = max(m.shape[1] for m, _ in chunks)
            chunks = [(torch.nn.functional.pad(m, (0, 0, 0, width-m.shape[1])),
                       torch.nn.functional.pad(p, (0, width-p.shape[1]), value=True)) for m, p in chunks]
            inputs = torch.zeros(len(rows), 1, dtype=torch.long, device=self.device)
            for t in range(int(counts.max())):
                parts = []
                for ci, (memory, padding) in enumerate(chunks):
                    lo = ci*cap
                    parts.append(belief.decode(memory.to(self.device), padding.to(self.device),
                                                   inputs[lo:lo+cap], counts[lo:lo+cap])[:, -1])
                card = torch.multinomial(torch.cat(parts).exp(), 1)+1
                result.scatter_add_(1, card, (counts > t).long()[:, None])
                inputs = torch.cat((inputs, card), 1)
        return result.cpu().numpy().astype(np.int32)

    def _sample_worlds(self, src, roots):
        c = self.config
        oracle = isinstance(self.belief, OracleBelief)
        rr = np.repeat(roots, c.worlds).astype(np.int32)
        from .search_belief import MixedBelief
        mixed = isinstance(self.belief, MixedBelief)
        use_b1 = self.belief.choose(self.rng, len(rr)) if mixed else None
        assignments = np.zeros((len(rr), 297), np.int32)
        valid = np.zeros(len(rr), bool)
        attempts = np.zeros(len(rr), int)
        while True:
            todo = np.flatnonzero(~valid & (attempts < (1 if oracle else c.max_attempts)))
            if not len(todo):
                break
            if oracle:
                # Includes public used cards, exactly like the training label.
                # Keep the same validation and load conversion as sampled beliefs.
                samples = src.label_opp_hand[rr[todo]].astype(np.int32)
            elif mixed:
                samples = np.zeros((len(todo), 297), np.int32)
                mask = use_b1[todo]
                baseline = self.belief.baseline
                pn, pk = baseline.count_probs.numpy(), baseline.card_probs.numpy()
                for j in np.flatnonzero(mask):
                    samples[j] = self.rng.multinomial(self.rng.choice(19, p=pn), pk)
                if (~mask).any():
                    samples[~mask] = self._belief_samples(src, rr[todo[~mask]], self.belief.ar)
            elif isinstance(self.belief, Baseline):
                pn, pk = self.belief.count_probs.numpy(), self.belief.card_probs.numpy()
                samples = np.array([self.rng.multinomial(self.rng.choice(19, p=pn), pk) for _ in todo], np.int32)
            else:
                samples = self._belief_samples(src, rr[todo])
            for i, sample in zip(todo, samples):
                attempts[i] += 1
                try:
                    src.unused_assignment(int(rr[i]), sample)
                except ValueError:
                    self.counts['invalid_samples'] += 1
                    if oracle:
                        self.counts['invalid_oracle_assignments'] += 1
                    continue
                assignments[i] = sample
                valid[i] = True
        self.counts['resamples'] += int(np.maximum(attempts-1, 0).sum())
        self.counts['exhausted_worlds'] += int((~valid).sum())
        seeds = self.rng.integers(0, np.iinfo(np.uint64).max, len(rr), dtype=np.uint64)
        return rr[valid], assignments[valid], seeds[valid]

    def _act(self, actor, rows, phase):
        groups = getattr(actor, '_search_root_groups', None)
        result = actor.act(rows, compact_transfer=True, observation_groups=groups)
        actor._search_root_groups = None
        if actor.token_cache is not None and not getattr(actor, '_search_cache_recorded', False):
            cache = actor.token_cache
            self.counts['cache_'+cache.device.type+'_actors'] += 1
            self.counts['cache_peak_bytes'] = max(self.counts['cache_peak_bytes'],
                                                cache.numel()*cache.element_size())
            actor._search_cache_recorded = True
        if self.profile:
            for key in ('transfer', 'encoder', 'heads'):
                self.seconds[phase+'_'+key] += actor.last_profile[key]
        return result

    @torch.inference_mode()
    def search(self, src, roots, **kwargs):
        """Search enabled roots and measure each public root kind separately.

        Main-only batches retain their original order and random draws.
        Response substeps together complete one external policy decision.
        """
        roots = np.asarray(roots, np.int32)
        roots = roots[search_support(src, roots, self.config.search_response) == 0]
        response = response_roots(src, roots)
        selected, reports, candidates, policies, diagnostics = {}, [], {}, {}, {}
        for kind, take in (('main', ~response), ('response', response)):
            rows = roots[take]
            if not len(rows):
                continue
            start = self._time()
            selected.update(self._search(src, rows, **kwargs))
            self.seconds['total_'+kind] += self._time()-start
            self.counts['searched_'+kind] += len(rows)
            reports.extend(self.last_search)
            candidates.update(self.last_candidates)
            policies.update(self.last_policy_paths)
            diagnostics.update(self.last_candidate_diagnostics)
        self.last_search, self.last_candidates, self.last_policy_paths = reports, candidates, policies
        self.last_candidate_diagnostics = diagnostics
        return selected

    @torch.inference_mode()
    def _search(self, src, roots, *, fixed_candidates=None, policy_paths=None, report_values=False):
        """Search supported roots; return row -> full external-choice path.

        For repeated q measurements, pass last_candidates as fixed_candidates
        and last_policy_paths as policy_paths. Paths, order and log priors stay
        fixed; hidden worlds, chance seeds and rollout actions are resampled.
        report_values adds per-world leaves and their sample standard error.
        The caller isolates Torch RNG (as __call__ does with _policy_rng).
        """
        c = self.config
        roots = np.asarray(roots, np.int32)
        if c.root_batch_size and len(roots) > c.root_batch_size:
            selected, reports, candidates, policies, diagnostics = {}, [], {}, {}, {}
            for lo in range(0, len(roots), c.root_batch_size):
                selected.update(self._search(src, roots[lo:lo+c.root_batch_size],
                                            fixed_candidates=fixed_candidates, policy_paths=policy_paths,
                                            report_values=report_values))
                reports.extend(self.last_search)
                candidates.update(self.last_candidates)
                policies.update(self.last_policy_paths)
                diagnostics.update(self.last_candidate_diagnostics)
            self.last_candidate_diagnostics = diagnostics
            self.last_search, self.last_candidates, self.last_policy_paths = reports, candidates, policies
            return selected
        self.counts['search_batches'] += 1
        self.counts['batch_roots'] += len(roots)
        self.counts['max_batch_roots'] = max(self.counts['max_batch_roots'], len(roots))
        self.last_search = []
        self.last_candidates = {}
        self.last_candidate_diagnostics = {int(r): dict(sampled=0, unique=0, retained=0, valid_worlds=[], loads={}, singleton=False) for r in roots}
        self.last_policy_paths = {}
        start = self._time()
        rr, assignments, seeds = self._sample_worlds(src, roots)
        self.seconds['sample'] += self._time()-start
        groups = {int(r): np.flatnonzero(rr == r) for r in roots}
        if not len(rr):
            self.last_search = []
            return {}
        start = self._time()
        if fixed_candidates is None:
            indices = np.concatenate([np.resize(ids, c.candidate_samples) for ids in groups.values() if len(ids)])
            env, status = self._world_env(src, rr[indices], assignments[indices], seeds[indices], 1, role="candidate")
            for r in roots:
                ids = np.flatnonzero(rr[indices] == r)
                self.last_candidate_diagnostics[int(r)]['loads'] = dict(Counter(int(v) for v in status[ids]))
            actor = _SearchActor(self.model, env, self.device, profile=self.profile, max_batch=c.max_inference_batch,
                                 cache_device=c.search_cache_device, cache_mb=c.search_cache_mb)
            actor._search_root_groups = rr[indices]
            paths = [[] for _ in indices]
            lp, lr = np.zeros(len(indices)), np.zeros(len(indices))
            while not env.episode_done.all():
                active = np.flatnonzero(env.stage >= 0)
                act = self._act(actor, active, "candidate")
                for i in active:
                    paths[i].append(int(act.choices[i]))
                lp[active] += act.log_prob[active]
                lr[active] -= np.log(env.mask[active].sum(1))
                env.step(act.choices)
            candidates = {int(r): {} for r in roots}
            policy_paths = {}
            for i, idx in enumerate(indices):
                if status[i] == 0:
                    policy_paths.setdefault(int(rr[idx]), tuple(paths[i]))
                    candidates[int(rr[idx])][tuple(paths[i])] = (float(lp[i]), float(lr[i]))
            for r, items in candidates.items():
                self.last_candidate_diagnostics[r]['unique'] = len(items)
                if not items and len(groups[r]):
                    self.last_candidate_diagnostics[r]['failure'] = 'world_rejection'
            candidates = {r: sorted(items.items(), key=lambda x: (-x[1][0], x[0]))[:c.candidates] for r, items in candidates.items()}
        else:
            candidates = {int(r): fixed_candidates.get(int(r), []) for r in roots}
            policy_paths = policy_paths or {}
        for r, items in candidates.items():
            self.last_candidate_diagnostics[r].update(
                sampled=(int(sum(status[i] == 0 for i, idx in enumerate(indices) if rr[idx] == r))
                         if fixed_candidates is None else None), retained=len(items))
        self.last_candidates = candidates
        self.counts['candidate_paths'] += sum(map(len, candidates.values()))
        self.last_policy_paths = policy_paths
        # Diagnostics requesting q/world values always execute full rollouts.
        singleton = {r for r, items in candidates.items() if len(items) == 1
                     and c.skip_single_candidate and not report_values}
        jobs = [(r, path, priors, int(w)) for r, items in candidates.items()
                for path, priors in items for w in groups[r]]
        self.seconds['candidate'] += self._time()-start
        if not jobs:
            return {}
        for r in singleton:
            self.last_candidate_diagnostics[r]['singleton'] = True
        self.counts['singleton_roots'] += len(singleton)
        self.counts['rollout_roots'] += sum(bool(items) and r not in singleton for r, items in candidates.items())
        aggregated = {}
        for fast in (True, False):
            subset = [j for j in jobs if (j[0] in singleton) == fast]
            if subset:
                if not fast:
                    self.counts['rollout_worlds'] += len(subset)
                aggregated.update(self._evaluate_jobs(src, rr, assignments, seeds, subset, singleton=fast))
        selected_paths, reports = {}, []
        for r, items in candidates.items():
            kept = []
            for path, priors in items:
                samples = aggregated.get((r, path), [])
                self.last_candidate_diagnostics[r]["valid_worlds"].append(len(samples))
                if samples:
                    kept.append((path, priors, float(np.mean(samples)), len(samples)))
            if not kept:
                continue
            probs = search_distribution([x[2] for x in kept], [x[1][1] for x in kept], [x[1][0] for x in kept], c.alpha, c.beta)
            pick = int(probs.argmax()) if c.select == 'argmax' else int(self.rng.choice(len(kept), p=probs))
            self.counts['selected'] += 1
            self.counts['policy_disagreements'] += kept[pick][0] != policy_paths.get(r)
            selected_paths[r] = kept[pick][0]
            reports.append(dict(row=r, candidates=[dict(path=list(x[0]), log_pi=x[1][0], log_rho=x[1][1], q=None if r in singleton else x[2], valid_worlds=x[3]) for x in kept], probabilities=probs.tolist(), selected=list(kept[pick][0])))
        if report_values:
            for report in reports:
                report['policy_path'] = list(policy_paths.get(report['row'], ()))
                for candidate in report['candidates']:
                    samples = aggregated[(report['row'], tuple(candidate['path']))]
                    candidate['world_values'] = list(samples)
                    candidate['q_se'] = float(np.std(samples, ddof=1)/np.sqrt(len(samples))) if len(samples) > 1 else None
        for report in reports:
            if report['row'] in singleton:
                report['rollout_skipped'] = 'single_candidate'
        self.last_search = reports
        return selected_paths

    def _evaluate_jobs(self, src, rr, assignments, seeds, jobs, *, singleton=False):
        c = self.config
        start = self._time()
        wi = np.array([j[3] for j in jobs])
        env, status = self._world_env(src, rr[wi], assignments[wi], seeds[wi], 1 if singleton else c.depth+1, role="singleton" if singleton else "rollout")
        actor = _SearchActor(self.model, env, self.device, profile=self.profile, max_batch=c.max_inference_batch,
                                 cache_device=c.search_cache_device, cache_mb=c.search_cache_mb)
        actor._search_root_groups = None if c.skip_forced_inference or singleton else rr[wi]
        valid = status == 0
        cursor = np.zeros(len(jobs), int)
        lengths = np.array([len(j[1]) for j in jobs])
        paths = np.zeros((len(jobs), max(lengths.max(), 1)), np.int32)
        for i, job in enumerate(jobs):
            paths[i, :lengths[i]] = job[1]
        while True:
            active = np.flatnonzero((env.stage >= 0) & valid)
            if not len(active):
                break
            forcing = active[env.decisions[active] == 0]
            infer = active[env.decisions[active] != 0] if c.skip_forced_inference or singleton else active
            self.counts['forced_inference_rows_skipped'] += len(active)-len(infer)
            # No cache is needed for forced root prefixes: the next external
            # decision starts at substep zero and refreshes its encoding.
            if len(infer):
                action_choices = self._act(actor, infer, "rollout").choices
            else:
                action_choices = np.zeros(env.num_envs, np.int32)
            choices = paths[forcing, np.minimum(cursor[forcing], paths.shape[1]-1)]
            ok = (cursor[forcing] < lengths[forcing]) & env.mask[forcing, choices]
            invalid = forcing[~ok]
            valid[invalid] = False
            self.counts['invalid_candidate_worlds'] += len(invalid)
            accepted = forcing[ok]
            action_choices[accepted] = choices[ok]
            cursor[accepted] += 1
            # Invalid rows must still receive legal choices until other rows stop.
            idle_invalid = np.flatnonzero((env.stage >= 0) & ~valid)
            action_choices[idle_invalid] = env.mask[idle_invalid].argmax(1)
            before = env.decisions.copy()
            env.step(action_choices)
            invalid = forcing[valid[forcing] & (env.decisions[forcing] > before[forcing]) & (cursor[forcing] != lengths[forcing])]
            valid[invalid] = False
            self.counts['invalid_candidate_worlds'] += len(invalid)
        self.seconds['rollout'] += self._time()-start
        if singleton:
            self.counts['singleton_validated_worlds'] += int(valid.sum())
            aggregated = {}
            for i, (r, path, _, _) in enumerate(jobs):
                if valid[i]:
                    aggregated.setdefault((r, path), []).append(0.)
            return aggregated
        start = self._time()
        values = env.result.astype(np.float64).copy()
        rows = np.flatnonzero(valid).astype(np.int32)
        # The engine actor is absent for true terminals (including draws at D).
        if len(rows):
            acting = env.next_actor_seats(rows)
            cutoff = acting >= 0
            selected = rows[cutoff]
            own = acting[cutoff] == env.searcher_seat[selected]
            self.counts['depth_leaves'] += len(selected)
            self.counts['depth_leaves_searcher'] += int(own.sum())
            self.counts['depth_leaves_opponent'] += int((~own).sum())
            self.counts['terminal_leaves'] += int((~cutoff).sum())
            seats = acting[cutoff] if c.leaf_view == 'actor' else env.searcher_seat[selected]
            batch = c.max_inference_batch or max(len(selected), 1)
            for lo in range(0, len(selected), batch):
                rr = selected[lo:lo+batch]
                leaf = env.observe(rr, seats[lo:lo+batch].astype(np.int32))
                obs = tensor_batch(leaf, self.device)
                logits = self.model.value_head(self.model.encode(obs)[:, 0]).float()
                p = logits.softmax(-1)
                v = (p[:, 0]-p[:, 1]).cpu().numpy()
                if c.leaf_view == 'actor':
                    v = np.where(own[lo:lo+batch], v, -v)
                values[rr] = v
                del obs, logits, p
        self.seconds['leaf'] += self._time()-start
        aggregated = {}
        for i, (r, path, _, _) in enumerate(jobs):
            if valid[i]:
                aggregated.setdefault((r, path), []).append(values[i])
        return aggregated

    @torch.inference_mode()
    def __call__(self, env, rows):
        if env is not self.env:
            self.pending.clear()
            self.env = env
        choices = self.actor(env, rows)
        fresh = np.asarray(rows, np.int32)[env.substep[rows] == 0]
        self.counts['decisions'] += len(fresh)
        for r in fresh:
            if self.pending.get(int(r)):
                self.counts['real_mismatches'] += 1
            self.pending.pop(int(r), None)
        if len(fresh):
            support = search_support(env, fresh, self.config.search_response)
            if not self.config.search_response:
                self.counts['skip_response_disabled'] += int(response_roots(env, fresh).sum())
            for reason in support[support != 0]:
                self.counts['skip_'+_SUPPORT.get(int(reason), str(reason))] += 1
            roots = fresh[support == 0]
            if len(roots):
                self.counts['searched'] += len(roots)
                self.counts['search_calls'] += 1
                start = self._time()
                with _policy_rng(int(self.rng.integers(2**31)), (self,)):
                    paths = self.search(env, roots)
                self.seconds['total'] += self._time()-start
                self.counts['fallback_no_candidates'] += len(roots)-len(paths)
                self.pending.update({r: list(path) for r, path in paths.items()})
        for r in rows:
            path = self.pending.get(int(r))
            if path is not None and not path:
                self.counts['real_mismatches'] += 1
                self.pending.pop(int(r), None)
            if path:
                choice = path.pop(0)
                if env.mask[r, choice]:
                    choices[r] = choice
                else:
                    self.counts['real_mismatches'] += 1
                    self.pending.pop(int(r), None)
        return choices
