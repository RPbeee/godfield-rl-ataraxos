"""Batched information-set search. Only public observations enter inference.

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
from .actor import Actor, synchronize
from .belief import belief_from_checkpoint
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

    def __post_init__(self):
        if min(self.worlds, self.candidates, self.candidate_samples, self.max_attempts) <= 0 or self.depth < 0:
            raise ValueError('positive search budgets and nonnegative depth required')
        if not np.isfinite([self.alpha, self.beta]).all() or min(self.alpha, self.beta) < 0 or self.alpha+self.beta <= 0:
            raise ValueError('finite nonnegative alpha/beta with positive sum required')
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


def load_search_belief(spec, fit=None, device='cpu'):
    if spec == 'b1':
        if fit is None:
            raise ValueError('--fit is required for b1')
        return fit_baselines(torch.load(fit, map_location='cpu', weights_only=False))['B1']
    if spec == 'uniform':
        # B0 gift rates with a uniform count prior when no fitting data supplied.
        pn = (fit_baselines(torch.load(fit, map_location='cpu', weights_only=False))['B0'].count_probs
              if fit else torch.full((19,), 1/19, dtype=torch.float64))
        return Baseline(pn, gift_probabilities())
    return belief_from_checkpoint(torch.load(spec, map_location='cpu', weights_only=False)).to(device).eval().requires_grad_(False)


_SUPPORT = {-1: 'inside_decision', 1: 'invalid_viewer', 2: 'unsupported_players',
            3: 'not_main', 4: 'not_actor', 5: 'pending_work', 6: 'reserved_gifts',
            7: 'replay_proof', 8: 'hidden_status', 9: 'hidden_confusion', 10: 'hidden_own_model'}


class SearchAgent:
    def __init__(self, model, belief, device='cpu', *, config=SearchConfig(), seed=0, num_threads=1):
        self.device = torch.device(device)
        self.model = model.to(device).eval()
        self.belief = belief
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.num_threads = num_threads
        self.actor = Actor(model, VecEnv(1, 1, eval_pairs=0), device)
        self.counts = Counter({k: 0 for k in (
            'decisions', 'searched', 'selected', 'invalid_samples', 'resamples',
            'exhausted_worlds', 'invalid_loads', 'invalid_candidate_worlds',
            'real_mismatches', 'policy_disagreements', 'fallback_no_candidates')})
        self.seconds = Counter()
        self.pending = {}
        self.last_search = []
        self.env = None

    def statistics(self):
        n = self.counts['searched']
        decisions = self.counts['decisions']
        return dict(counts=dict(self.counts), searched_fraction=n/decisions if decisions else 0,
                    seconds=dict(self.seconds), seconds_per_search={k: v/n if n else 0 for k, v in self.seconds.items()},
                    policy_disagreement_fraction=(self.counts['policy_disagreements']/self.counts['selected']
                                                  if self.counts['selected'] else 0),
                    disagreement_definition='selected path differs from the first full pi_theta candidate draw (before top-k filtering)')

    def _time(self):
        synchronize(self.device)
        return perf_counter()

    def _world_env(self, src, rows, assignments, seeds, depth):
        env = SearchEnv(len(rows), self.num_threads, history_len=src.obs_schema()['K'], max_decisions=depth)
        status = env.load_worlds(src, np.asarray(rows, np.int32), np.asarray(assignments, np.int32), np.asarray(seeds, np.uint64))
        self.counts['invalid_loads'] += int((status != 0).sum())
        return env, status.copy()

    def _sample_worlds(self, src, roots):
        c = self.config
        rr = np.repeat(roots, c.worlds).astype(np.int32)
        assignments = np.zeros((len(rr), 297), np.int32)
        valid = np.zeros(len(rr), bool)
        attempts = np.zeros(len(rr), int)
        while True:
            todo = np.flatnonzero(~valid & (attempts < c.max_attempts))
            if not len(todo):
                break
            if isinstance(self.belief, Baseline):
                pn, pk = self.belief.count_probs.numpy(), self.belief.card_probs.numpy()
                samples = np.array([self.rng.multinomial(self.rng.choice(19, p=pn), pk) for _ in todo], np.int32)
            else:
                obs = tensor_batch({k: getattr(src, k)[rr[todo]] for k in OBS_FIELDS}, self.device)
                samples = self.belief.sample(obs, 1)[:, 0].cpu().numpy().astype(np.int32)
            for i, sample in zip(todo, samples):
                attempts[i] += 1
                try:
                    src.unused_assignment(int(rr[i]), sample)
                except ValueError:
                    self.counts['invalid_samples'] += 1
                    continue
                assignments[i] = sample
                valid[i] = True
        self.counts['resamples'] += int(np.maximum(attempts-1, 0).sum())
        self.counts['exhausted_worlds'] += int((~valid).sum())
        seeds = self.rng.integers(0, np.iinfo(np.uint64).max, len(rr), dtype=np.uint64)
        return rr[valid], assignments[valid], seeds[valid]

    @torch.inference_mode()
    def search(self, src, roots):
        """Search all supplied supported roots together; returns row -> path."""
        c = self.config
        start = self._time()
        rr, assignments, seeds = self._sample_worlds(src, roots)
        self.seconds['sample'] += self._time()-start
        groups = {int(r): np.flatnonzero(rr == r) for r in roots}
        if not len(rr):
            self.last_search = []
            return {}
        start = self._time()
        indices = np.concatenate([np.resize(ids, c.candidate_samples) for ids in groups.values() if len(ids)])
        env, status = self._world_env(src, rr[indices], assignments[indices], seeds[indices], 1)
        actor = Actor(self.model, env, self.device)
        paths = [[] for _ in indices]
        lp, lr = np.zeros(len(indices)), np.zeros(len(indices))
        while not env.episode_done.all():
            active = np.flatnonzero(env.stage >= 0)
            act = actor.act(active)
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
        candidates = {r: sorted(items.items(), key=lambda x: (-x[1][0], x[0]))[:c.candidates] for r, items in candidates.items()}
        jobs = [(r, path, priors, int(w)) for r, items in candidates.items() for path, priors in items for w in groups[r]]
        self.seconds['candidate'] += self._time()-start
        if not jobs:
            return {}
        start = self._time()
        wi = np.array([j[3] for j in jobs])
        env, status = self._world_env(src, rr[wi], assignments[wi], seeds[wi], c.depth+1)
        actor = Actor(self.model, env, self.device)
        valid = status == 0
        cursor = np.zeros(len(jobs), int)
        while True:
            active = np.flatnonzero((env.stage >= 0) & valid)
            if not len(active):
                break
            act = actor.act(active)
            forcing = active[env.decisions[active] == 0]
            for i in forcing:
                path = jobs[i][1]
                if cursor[i] >= len(path) or not env.mask[i, path[cursor[i]]]:
                    valid[i] = False
                    self.counts['invalid_candidate_worlds'] += 1
                else:
                    act.choices[i] = path[cursor[i]]
                    cursor[i] += 1
            # Invalid rows must still receive legal choices until other rows stop.
            idle_invalid = np.flatnonzero((env.stage >= 0) & ~valid)
            act.choices[idle_invalid] = env.mask[idle_invalid].argmax(1)
            before = env.decisions.copy()
            env.step(act.choices)
            for i in forcing:
                if valid[i] and env.decisions[i] > before[i] and cursor[i] != len(jobs[i][1]):
                    valid[i] = False
                    self.counts['invalid_candidate_worlds'] += 1
        self.seconds['rollout'] += self._time()-start
        start = self._time()
        values = env.result.astype(np.float64).copy()
        rows = np.flatnonzero(valid).astype(np.int32)
        # Encoder category encoding is enum+1: finished=2 -> 3. Unlike
        # truncated/decisions alone, this also distinguishes engine draws at D.
        if len(rows):
            leaf = env.observe(rows, env.searcher_seat[rows].astype(np.int32))
            phase = src.obs_schema()['cat']['phase']['index']
            cutoff = leaf['obs_cat'][:, 0, phase] != 3
            selected = rows[cutoff]
            self.counts['depth_leaves'] += len(selected)
            self.counts['terminal_leaves'] += int((~cutoff).sum())
            if len(selected):
                obs = tensor_batch({k: v[cutoff] for k, v in leaf.items()}, self.device)
                logits = self.model.value_head(self.model.encode(obs)[:, 0]).float()
                p = logits.softmax(-1)
                values[selected] = (p[:, 0]-p[:, 1]).cpu().numpy()
        self.seconds['leaf'] += self._time()-start
        aggregated = {}
        for i, (r, path, _, _) in enumerate(jobs):
            if valid[i]:
                aggregated.setdefault((r, path), []).append(values[i])
        selected_paths, reports = {}, []
        for r, items in candidates.items():
            kept = []
            for path, priors in items:
                samples = aggregated.get((r, path), [])
                if samples:
                    kept.append((path, priors, float(np.mean(samples)), len(samples)))
            if not kept:
                continue
            probs = search_distribution([x[2] for x in kept], [x[1][1] for x in kept], [x[1][0] for x in kept], c.alpha, c.beta)
            pick = int(probs.argmax()) if c.select == 'argmax' else int(self.rng.choice(len(kept), p=probs))
            self.counts['selected'] += 1
            self.counts['policy_disagreements'] += kept[pick][0] != policy_paths[r]
            selected_paths[r] = kept[pick][0]
            reports.append(dict(row=r, candidates=[dict(path=list(x[0]), log_pi=x[1][0], log_rho=x[1][1], q=x[2], valid_worlds=x[3]) for x in kept], probabilities=probs.tolist(), selected=list(kept[pick][0])))
        self.last_search = reports
        return selected_paths

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
            support = env.determinize_support(fresh)
            for reason in support[support != 0]:
                self.counts['skip_'+_SUPPORT.get(int(reason), str(reason))] += 1
            roots = fresh[support == 0]
            if len(roots):
                self.counts['searched'] += len(roots)
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
