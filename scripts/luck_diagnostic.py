#!/usr/bin/env python3
"""Offline oracle Q / chance-policy noise / CRN / learner GAE diagnostic.

The native diagnostic_copy_exact entry point copies BOTH private histories and
all engine state. No determinization, public projection, search or leaf value
substitutes for terminal outcomes. CUDA is batched policy inference; C++ workers
simulate games on the host. Nothing in play/search/training imports this script.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import heapq
import hashlib
import json
from pathlib import Path
import platform
from statistics import NormalDist
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl import SearchEnv, VecEnv
from gfrl.actor import Actor, ActResult, PREFIX_FIELDS
from gfrl.evaluation import _policy_rng
from gfrl.model import OBS_FIELDS, Model, ModelConfig
from gfrl.search import _SearchActor, _RolloutRandom
from gfrl.targets import sequence_targets
from gfrl.transfer import tensor_batch


def copy_exact(dst, src, rows, seeds=None, destinations=None):
    rows = np.ascontiguousarray(rows, dtype=np.int32)
    dst.diagnostic_copy_exact(src, rows,
        np.arange(len(rows), dtype=np.int32) if destinations is None else np.asarray(destinations, np.int32),
        np.zeros(len(rows), np.uint64) if seeds is None else np.asarray(seeds, np.uint64),
        reseed=seeds is not None)


def world(args, n, depth=None):
    return SearchEnv(n, min(n, args.threads), max_decisions=depth or args.max_decisions)


def scalar(logits):
    p = torch.as_tensor(logits).softmax(-1).numpy()
    return p[:, 0]-p[:, 1]


@contextmanager
def opponent_precision(args):
    """Override the diagnostic autocast/TF32 settings only for the opponent."""
    cuda = args.device == 'cuda'
    previous = torch.backends.cuda.matmul.allow_tf32 if cuda else None
    try:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = args.opponent_training_tf32
        dtype = getattr(args, 'opponent_dtype', 'fp32')
        with torch.autocast(args.device, dtype=torch.bfloat16 if dtype == 'bf16' else torch.float16,
                            enabled=dtype != 'fp32'):
            yield
    finally:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = previous


def collection_action(args, actor, opponent):
    env = actor.env
    own = env.actor_seat == env.episode_id % 2
    active = env.stage >= 0
    selected = actor.act(np.flatnonzero(active & own), compact_transfer=True)
    with opponent_precision(args):
        other = opponent.act(np.flatnonzero(active & ~own), compact_transfer=True)
    choices = selected.choices.copy()
    choices[~own] = other.choices[~own]
    # Opponent values/log probabilities are not diagnostic observations.
    return ActResult(choices, selected.log_prob, selected.value_logits, selected.log_probs)


@torch.inference_mode()
def collect(args, model, opponent_model=None):
    """Uniform per-game reservoir, or pre-drawn fixed-window indices."""
    env = VecEnv(min(args.games, args.num_envs), args.threads, eval_pairs=(args.games+1)//2,
                 opponent='external', eval_seed=args.seed)
    archive = world(args, args.games*args.roots_per_game)
    rng = np.random.default_rng(args.seed+1)
    reservoir = args.root_selection == 'reservoir'
    targets = [set() if reservoir else set(rng.choice(args.target_range, args.roots_per_game, replace=False).tolist())
               for _ in range(args.games)]
    target_list = None if reservoir else [sorted(t) for t in targets]
    if opponent_model is None:
        actor = Actor(model, env, args.device)
        opponent = None
    else:
        actor = _SearchActor(model, env, args.device, max_batch=args.inference_batch, cache_mb=args.cache_mb)
        opponent = _SearchActor(opponent_model, env, args.device, max_batch=args.inference_batch,
                                cache_mb=args.cache_mb)
    lengths = Counter()
    own_lengths = Counter()
    sequences, pending, records_by_slot = {}, {}, {}
    phase_index = env.obs_schema()['cat']['phase']['index']
    last = perf_counter()
    with _policy_rng(args.seed+2, (actor,) if opponent is None else (actor, opponent)):
        while not env.eval_done():
            active = np.flatnonzero(env.stage >= 0)
            fresh = active[env.substep[active] == 0]
            save, dest = [], []
            for r in fresh:
                g, seat = int(env.episode_id[r]), int(env.actor_seat[r])
                d = lengths[g]
                lengths[g] += 1
                if opponent is not None and seat != g % 2:
                    continue
                selection_index = d if opponent is None else own_lengths[g]
                own_lengths[g] += 1
                seq = sequences.setdefault((g, seat), [])
                idx = None
                if g < args.games:
                    if reservoir:
                        slot = selection_index if selection_index < args.roots_per_game else int(rng.integers(selection_index+1))
                        if slot < args.roots_per_game:
                            idx = g*args.roots_per_game+slot
                    elif selection_index in targets[g]:
                        targets[g].remove(selection_index)
                        idx = len(records_by_slot)
                if idx is not None:
                    rec = dict(game=g, seat=seat, decision_index=d, sequence_index=len(seq),
                               phase='response' if env.obs_cat[r, 0, phase_index] == 2 else 'main',
                               actual_path=[], actual_log_pi=0., archive_row=idx)
                    records_by_slot[idx] = rec
                    if opponent is not None:
                        rec['diagnostic_decision_index'] = selection_index
                    pending[int(r)] = rec
                    save.append(r); dest.append(idx)
            if save:
                copy_exact(archive, env, save, destinations=dest)
            result = (actor.act(compact_transfer=True) if opponent is None
                      else collection_action(args, actor, opponent))
            probs = torch.from_numpy(result.value_logits).softmax(-1).numpy()
            for r in fresh:
                g, seat = int(env.episode_id[r]), int(env.actor_seat[r])
                if opponent is not None and seat != g % 2:
                    continue
                sequences[g, seat].append(probs[r].copy())
                if int(r) in pending:
                    pending[int(r)]['v'] = float(probs[r, 0]-probs[r, 1])
            for r, rec in pending.items():
                rec['actual_path'].append(int(result.choices[r]))
                rec['actual_log_pi'] += float(result.log_prob[r])
            env.step(result.choices)
            pending = {r: rec for r, rec in pending.items() if not env.decision_done[r]}
            if perf_counter()-last > 10:
                print(f'collect: {len(env.eval_results())}/{args.games} games, {len(records_by_slot)} roots', flush=True)
                last = perf_counter()
    games = {int(g['g']): dict(g, external_decisions=int(lengths[g['g']]))
             for g in env.eval_results() if g['g'] < args.games}
    if opponent is not None:
        for g, game in games.items():
            game['diagnostic_decisions'] = int(own_lengths[g])
    gae = {}
    for (g, seat), ps in sequences.items():
        if g in games:
            game = games[g]
            z = 0 if game['truncated'] else game['result']*(1 if game['agent_seat'] == seat else -1)
            _, gae[g, seat] = sequence_targets(ps, z, args.lambda_v, args.lambda_a, args.gamma)
    records = sorted(records_by_slot.values(), key=lambda r: r['archive_row'])
    for rec in records:
        game = games[rec['game']]
        rec.update(real_truncated=bool(game['truncated']),
                   decisions_to_end=game['external_decisions']-rec['decision_index'],
                   progress=rec['decision_index']/max(game['external_decisions'], 1),
                   gae=float(gae[rec['game'], rec['seat']][rec['sequence_index']]))
    return archive, records, list(games.values()), target_list


@torch.inference_mode()
def candidates(args, model, archive, rec):
    """Best-first exact path probabilities, with a bounded prefix frontier.

    A prefix's probability upper-bounds every descendant's probability. Batched
    expansion may over-enumerate but certification checks the remaining bound.
    Actual real action is always included, even when outside the top mass.
    """
    root = rec['archive_row']
    encoded = model.encode(tensor_batch({k: getattr(archive, k)[[root]] for k in OBS_FIELDS}, args.device))
    frontier = [(-1., ())]
    complete = {}
    root_policy = {}
    nodes = 0
    certified = False
    while frontier and nodes < args.candidate_nodes:
        top = sorted(complete.items(), key=lambda kv: (-kv[1], kv[0]))
        keep = top[:args.max_candidates]
        mass = sum(p for _, p in keep)
        if keep and ((mass >= args.mass and len(keep) >= args.min_candidates) or len(keep) == args.max_candidates):
            if keep[-1][1] >= -frontier[0][0]:
                certified = True
                break
        batch = [heapq.heappop(frontier) for _ in range(min(len(frontier), args.prefix_batch,
                                                                           args.candidate_nodes-nodes))]
        nodes += len(batch)
        env = world(args, len(batch), 1)
        copy_exact(env, archive, [root]*len(batch))
        lengths = {len(path) for _, path in batch}
        if len(lengths) == 1:
            for depth in range(next(iter(lengths))):
                env.step(np.array([path[depth] for _, path in batch], np.int32))
        else:
            # Native freezing is allowed only at substep zero. Replay each
            # distinct prefix length separately, then assemble the cursors.
            for length in sorted(lengths):
                take = [i for i, (_, p) in enumerate(batch) if len(p) == length]
                sub = world(args, len(take), 1)
                copy_exact(sub, archive, [root]*len(take))
                for d in range(length):
                    sub.step(np.array([batch[i][1][d] for i in take], np.int32))
                copy_exact(env, sub, range(len(take)), destinations=take)
        lp, _ = model.heads(encoded.expand(len(batch), -1, -1),
                            **tensor_batch({k: getattr(env, k) for k in PREFIX_FIELDS}, args.device))
        probabilities = np.exp(lp.float().cpu().numpy().astype(np.float64))
        probabilities /= probabilities.sum(1, keepdims=True)
        for i, (_, path) in enumerate(batch):
            root_policy[path] = probabilities[i]
        jobs = [(i, int(a), batch[i][1]+(int(a),), -batch[i][0]*float(probabilities[i, a]))
                for i in range(len(batch)) for a in np.flatnonzero(env.mask[i]) if probabilities[i, a] > 0]
        children = world(args, len(jobs), 1)
        copy_exact(children, env, [j[0] for j in jobs])
        children.step(np.array([j[1] for j in jobs], np.int32))
        for i, (_, _, path, probability) in enumerate(jobs):
            if children.decisions[i] > 0:
                complete[path] = probability
            else:
                heapq.heappush(frontier, (-probability, path))
    top = sorted(complete.items(), key=lambda kv: (-kv[1], kv[0]))[:args.max_candidates]
    selected = []
    mass = 0.
    for path, p in top:
        selected.append((path, p)); mass += p
        if mass >= args.mass and len(selected) >= args.min_candidates:
            break
    # No renormalization of the true policy mean: it has its own rollout arm.
    actual = tuple(rec['actual_path'])
    if actual not in dict(selected):
        # Evaluate the real path with the same normalized probabilities used
        # by the tree (the stored fp32 log-probability is only a replay witness).
        actual_env = world(args, 1, 1)
        copy_exact(actual_env, archive, [root])
        actual_probability = 1.
        for depth, choice in enumerate(actual):
            prefix = actual[:depth]
            if prefix not in root_policy:
                lp, _ = model.heads(encoded, **tensor_batch({k: getattr(actual_env, k) for k in PREFIX_FIELDS}, args.device))
                prob = np.exp(lp.float().cpu().numpy()[0].astype(np.float64))
                root_policy[prefix] = prob/prob.sum()
            actual_probability *= root_policy[prefix][choice]
            actual_env.step(np.array([choice], np.int32))
        selected.append((actual, actual_probability))
    selected.sort(key=lambda kv: (-kv[1], kv[0]))
    certified = bool(top) and (not frontier or min(p for pth, p in selected if pth in dict(top)) >= -frontier[0][0])
    rec.update(candidate_nodes=nodes, top_certified=certified,
               unexplored_prefix_bound=-frontier[0][0] if frontier else 0.,
               candidate_mass=float(sum(p for _, p in selected)),
               coverage_met=sum(p for _, p in selected) >= args.mass,
               candidate_paths=[list(p) for p, _ in selected], candidate_pi=[p for _, p in selected],
               actual_index=[p for p, _ in selected].index(actual))
    rec['tail_mass'] = max(0., 1.-rec['candidate_mass'])
    retained_prefixes = {path[:depth] for path, _ in selected for depth in range(len(path))}
    rec['root_policy_prefixes'] = [dict(path=list(path), probabilities=root_policy[path].tolist())
                                   for path in sorted(retained_prefixes)]
    return selected


@torch.inference_mode()
def wide_candidates(args, model, archive, rec):
    """Independent, bounded proposals of complete legal paths, without replacement.

    Uniform means uniform *prefix branches*, not uniform complete actions. The
    tempered law also acts at each prefix. Neither depends on rollout outcomes.
    Rejection excludes retained paths, duplicates and paths above the pi cap.
    """
    if not getattr(args, 'wide_candidates', 0):
        return
    rng = np.random.default_rng(np.random.SeedSequence(
        [args.seed, rec['game'], rec['decision_index'], 31]))
    root = rec['archive_row']
    encoded = model.encode(tensor_batch({k: getattr(archive, k)[[root]] for k in OBS_FIELDS}, args.device))
    cache = {tuple(p['path']): np.asarray(p['probabilities'], np.float64)
             for p in rec['root_policy_prefixes']}
    # Cache log probabilities as well: a legal but extremely unlikely path must
    # remain reachable by the uniform proposal even when exp(log_pi) underflows.
    logs = {}
    excluded = {tuple(p) for p in rec['candidate_paths']}
    accepted, rejected = [], Counter()
    env = world(args, 1, 1)
    for attempt in range(args.wide_draws):
        copy_exact(env, archive, [root])
        path, log_pi = (), 0.
        while env.decisions[0] == 0:
            if path not in logs:
                lp, _ = model.heads(encoded, **tensor_batch(
                    {k: getattr(env, k) for k in PREFIX_FIELDS}, args.device))
                legal = np.flatnonzero(env.mask[0])
                raw = lp.float().cpu().numpy()[0].astype(np.float64)
                if path in cache:
                    prob = cache[path]
                    # Retained-prefix probabilities must agree with the tail
                    # subtraction, including CUDA batch-dependent rounding.
                    log = np.full_like(raw, -np.inf)
                    np.log(prob, out=log, where=prob > 0)
                    log[(prob == 0) & env.mask[0].astype(bool)] = raw[(prob == 0) & env.mask[0].astype(bool)]
                else:
                    log = raw.copy()
                    log -= np.logaddexp.reduce(log[legal])
                logs[path] = (log, legal)
            log, legal = logs[path]
            if args.wide_method == 'prefix-uniform':
                choice = int(rng.choice(legal))
            else:
                # Shift before dividing, including extremely small positive T.
                with np.errstate(over='ignore'):
                    scores = (log[legal]-log[legal].max())/args.wide_temperature
                prob = np.exp(scores-scores.max()); prob /= prob.sum()
                choice = int(rng.choice(legal, p=prob))
            log_pi += float(log[choice])
            path += (choice,)
            env.step(np.array([choice], np.int32))
        if path in excluded:
            rejected['retained_or_duplicate'] += 1
        elif log_pi > np.log(args.wide_max_pi):
            rejected['above_pi_cap'] += 1
        else:
            excluded.add(path)
            accepted.append(dict(path=list(path), pi=float(np.exp(log_pi)), log_pi=log_pi,
                                 proposal_index=attempt+1))
            if len(accepted) == args.wide_candidates:
                break
    rec['wide'] = dict(actions=accepted, requested=args.wide_candidates, draws=attempt+1,
                       rejected=dict(rejected), shortfall=args.wide_candidates-len(accepted),
                       method=args.wide_method, max_pi=args.wide_max_pi)


class TailSampler:
    """Sample pi conditional on paths outside the retained set, without rejection.

    At a prefix, subtract the exactly known retained descendant probability
    from each branch's unconditional mass. Root choices use policy RNG only.
    Tiny tails use an unbiased control variate with ordinary pi sampling to
    avoid subtracting nearly equal floating-point masses.
    """
    def __init__(self, rec, n):
        self.retained = {tuple(p): v for p, v in zip(rec['candidate_paths'], rec['candidate_pi'])}
        self.subtree = Counter()
        for path, probability in self.retained.items():
            for depth in range(len(path)+1):
                self.subtree[path[:depth]] += probability
        self.prefix = [() for _ in range(n)]
        self.probability = np.ones(n)
        self.conditional = rec['tail_mass'] > 1e-7
        self.distributions = {tuple(p['path']): np.array(p['probabilities'])
                              for p in rec.get('root_policy_prefixes', [])}

    def choose(self, i, log_probs, uniform):
        prefix = self.prefix[i]
        if prefix in self.distributions:
            # Freeze retained-prefix logits from enumeration: bf16 GEMM may
            # otherwise round differently at different inference batch sizes,
            # enough to overwhelm a very small tail's subtraction.
            prob = self.distributions[prefix]
        else:
            prob = np.exp(log_probs.astype(np.float64)); prob /= prob.sum()
        if self.conditional:
            available = np.maximum(0., self.probability[i]*prob-np.array([
                self.subtree[prefix+(a,)] for a in range(len(prob))]))
            for a in range(len(prob)):
                if prefix+(a,) in self.retained:
                    available[a] = 0.
            if available.sum() <= 0:
                raise RuntimeError('tail sampler lost all remaining probability')
            distribution = available/available.sum()
        else:
            distribution = prob
        choice = min(int(np.searchsorted(distribution.cumsum(), uniform, side='right')), len(prob)-1)
        self.probability[i] *= prob[choice]
        self.prefix[i] += (choice,)
        return choice


class DiagnosticActor(_SearchActor):
    """Inverse-CDF samples, separated root/continuation streams per lane."""
    def __init__(self, args, model, env, policy_seeds):
        super().__init__(model, env, args.device, sample=False, max_batch=args.inference_batch,
                         cache_mb=args.cache_mb)
        self.continuation_random = _RolloutRandom(policy_seeds)
        self.root_random = _RolloutRandom(np.asarray(policy_seeds, np.uint64)^np.uint64(0x6A09E667F3BCC909))

    def policy_result(self, rows, **kwargs):
        return super().act(rows, **kwargs)

    def act(self, rows=None, **kwargs):
        groups = np.arange(self.env.num_envs)+1
        groups[self.env.decisions == 0] = 0
        kwargs.setdefault('observation_groups', groups)
        result = self.policy_result(rows, **kwargs)
        active = np.flatnonzero(self.env.stage >= 0) if rows is None else np.asarray(rows)
        u = np.empty(len(active), np.float32)
        root = self.env.decisions[active] == 0
        u[root] = self.root_random.uniforms(active[root])
        u[~root] = self.continuation_random.uniforms(active[~root])
        cdf = np.exp(result.log_probs[active].astype(np.float64)).cumsum(-1)
        choice = (cdf <= u[:, None]*cdf[:, -1:]).sum(-1).clip(max=cdf.shape[-1]-1)
        choices = result.choices.copy()
        choices[active] = choice
        self.last_uniforms = np.zeros(self.env.num_envs, np.float32)
        self.last_uniforms[active] = u
        return ActResult(choices, result.log_prob, result.value_logits, result.log_probs)


class OpponentDiagnosticActor(DiagnosticActor):
    """One CRN stream per lane; separate model caches and diagnostic-only V.

    The diagnostic head sees every acting viewer, including the opponent viewer
    used for signed post V. Only opponent policy logits replace its predictions.
    """
    def __init__(self, args, model, opponent_model, env, policy_seeds, seat):
        super().__init__(args, model, env, policy_seeds)
        self.args, self.seat = args, seat
        self.opponent = _SearchActor(opponent_model, env, args.device, sample=False,
                                    max_batch=args.inference_batch, cache_mb=args.cache_mb)

    def policy_result(self, rows, **kwargs):
        result = super().policy_result(rows, **kwargs)
        active = np.flatnonzero(self.env.stage >= 0) if rows is None else np.asarray(rows)
        other_rows = active[self.env.actor_seat[active] != self.seat]
        if len(other_rows):
            with opponent_precision(self.args):
                other = self.opponent.act(other_rows, **kwargs)
            result.log_probs[other_rows] = other.log_probs[other_rows]
        return result


@torch.inference_mode()
def panel(args, model, archive, rec, opponent_model=None):
    paths = rec['candidate_paths']
    K = len(paths)
    extra = [a['path'] for a in rec.get('wide', {}).get('actions', [])]
    arms = paths+[None]+extra  # original tail keeps its original index K
    A, C, P = len(arms), args.chance_repeats, args.policy_repeats
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, rec['game'], rec['decision_index'], 17]))
    chance_seeds = rng.integers(0, 2**64-1, C, dtype=np.uint64)
    policy_seeds = rng.integers(0, 2**64-1, P, dtype=np.uint64)
    shape = (A, C, P)
    output = {k: np.full(shape, np.nan) for k in ('outcome', 'gae', 'post_v', 'next_own_v')}
    tail_paths = np.full((C, P), -1, np.int32)  # retained action index, or -1 for tail
    status = np.zeros(shape, np.int8)  # 0 completed, 1 load failure, 2 truncated
    decisions = 0
    last = perf_counter()
    # Keep the original arm batches unchanged, even when adding wide arms.
    boundary = (K+1)*C*P
    batches = [(lo, min(lo+args.world_batch, end)) for begin, end in
               ((0, boundary), (boundary, A*C*P)) for lo in range(begin, end, args.world_batch)]
    for lo, hi in batches:
        flat = np.arange(lo, hi)
        aa, cc, pp = np.unravel_index(flat, shape)
        env = world(args, len(flat))
        try:
            copy_exact(env, archive, [rec['archive_row']]*len(flat), chance_seeds[cc])
        except (ValueError, RuntimeError) as error:
            # Retry lanes individually to keep failures separate, never silently
            # replace a failed world with a different seed.
            for i in range(len(flat)):
                try:
                    copy_exact(env, archive, [rec['archive_row']], chance_seeds[cc[i:i+1]], [i])
                except (ValueError, RuntimeError):
                    status.ravel()[flat[i]] = 1
            rec.setdefault('load_errors', []).append(str(error))
        valid = status.ravel()[flat] != 1
        actor = (DiagnosticActor(args, model, env, policy_seeds[pp]) if opponent_model is None else
                 OpponentDiagnosticActor(args, model, opponent_model, env, policy_seeds[pp], rec['seat']))
        tail_sampler = TailSampler(rec, len(flat))
        seat = rec['seat']
        accum = np.full(len(flat), -rec['v'], np.float64)
        weight = np.ones(len(flat), np.float64)
        post = np.full(len(flat), np.nan)
        next_own = np.full(len(flat), np.nan)
        while np.any((env.stage >= 0) & valid):
            active = np.flatnonzero((env.stage >= 0) & valid)
            fresh = active[env.substep[active] == 0]
            result = actor.act(active, compact_transfer=True)
            values = scalar(result.value_logits)
            after = active[(env.decisions[active] > 0) & np.isnan(post[active])]
            post[after] = values[after]*np.where(env.actor_seat[after] == seat, 1, -1)
            own = fresh[(env.decisions[fresh] > 0) & (env.actor_seat[fresh] == seat)]
            first = own[np.isnan(next_own[own])]
            next_own[first] = values[first]
            accum[own] += args.gamma*(1-args.lambda_a)*weight[own]*values[own]
            weight[own] *= args.gamma*args.lambda_a
            choices = result.choices
            for i in active[env.decisions[active] == 0]:
                if aa[i] != K:
                    choices[i] = arms[aa[i]][env.substep[i]]
                else:
                    choices[i] = tail_sampler.choose(i, result.log_probs[i], actor.last_uniforms[i])
            # Failed rows remain inactive from construction; all live valid rows
            # participate. step(-1) is needed only for any failed loaded cursor.
            if not valid.all():
                choices[~valid] = -1
            env.step(choices)
            if perf_counter()-last > 10:
                print(f'root {rec["archive_row"]}: worlds {lo}/{A*C*P}, '
                      f'active {len(active)}, decisions {int(env.decisions.sum())}', flush=True)
                last = perf_counter()
        truncated = env.truncated.copy() & valid
        status.ravel()[flat[truncated]] = 2
        z = env.result.astype(np.float64)
        complete = valid & ~truncated
        post[np.isnan(post)] = z[np.isnan(post)]
        next_own[np.isnan(next_own)] = z[np.isnan(next_own)]
        accum += args.gamma*weight*z
        for name, data in [('outcome', z), ('gae', accum), ('post_v', post), ('next_own_v', next_own)]:
            output[name].ravel()[flat[complete]] = data[complete]
        for i in np.flatnonzero(aa == len(paths)):
            path = tail_sampler.prefix[i]
            if path in tail_sampler.retained:
                tail_paths[cc[i], pp[i]] = paths.index(list(path))
        decisions += int(env.decisions.sum())
        del actor, env
    rec.update(world_counts=dict(requested=A*C*P, completed=int((status == 0).sum()),
                                 load_failed=int((status == 1).sum()), truncated=int((status == 2).sum())),
               continuation_decisions=decisions)
    return dict(**output, status=status, chance_seeds=chance_seeds, policy_seeds=policy_seeds,
                tail_retained_index=tail_paths)


def policy_baseline(rec, data):
    """Exact pi-weighted retained arms + independently sampled conditional tail."""
    x = data['outcome']
    pi = np.asarray(rec['candidate_pi'])
    K = len(pi)
    known = np.einsum('a,acp->cp', pi, x[:K])
    if rec['tail_mass'] > 1e-7:
        return known+rec['tail_mass']*x[K]
    # E[1{a in S} Y_a] = sum_S pi(a) Q_a. Per-cell CRN makes this
    # control variate cancel exactly whenever the sampled path is retained.
    residual = x[K].copy()
    for c, p in np.ndindex(residual.shape):
        j = data['tail_retained_index'][c, p]
        if j >= 0:
            residual[c, p] -= x[j, c, p]
    return known+residual


def crossed_cov(x):
    """Unbiased two-factor random-effects covariance, including interaction.

    x [variables, chance, policy]. Negative components are deliberately retained.
    The deterministic simulator has no residual after both RNG streams are fixed.
    """
    x = np.asarray(x, np.float64)
    if x.ndim != 3 or min(x.shape[1:]) < 2 or not np.isfinite(x).all():
        raise ValueError('finite complete crossed grid with >=2 seeds per factor required')
    _, C, P = x.shape
    mean = x.mean((1, 2))
    c, p = x.mean(2), x.mean(1)
    residual = x-c[:, :, None]-p[:, None, :]+mean[:, None, None]
    inter = np.einsum('acp,bcp->ab', residual, residual)/((C-1)*(P-1))
    dc, dp = c-mean[:, None], p-mean[:, None]
    chance = dc@dc.T/(C-1)-inter/P
    policy = dp@dp.T/(P-1)-inter/C
    total = chance+policy+inter
    mc = chance/C+policy/P+inter/(C*P)
    return dict(mean=mean, chance=chance, policy=policy, interaction=inter, total=total, mc=mc)


def contrast(cov, weights):
    return float(np.asarray(weights)@cov@np.asarray(weights))


def required(gap_squared, variance):
    return float(4*variance/gap_squared) if gap_squared > 0 and variance >= 0 else None


def pi_bucket(log_pi):
    for edge, label in [(1e-6, 'lt_1e-6'), (1e-4, '1e-6_to_1e-4'),
                        (1e-3, '1e-4_to_1e-3'), (1e-2, '1e-3_to_1e-2')]:
        if log_pi < np.log(edge):
            return label
    return 'ge_1e-2'


def wide_budgets(n):
    return sorted({n} | {2**i for i in range(n.bit_length()) if 2**i <= n})


def analyze_wide_root(rec, data):
    """Outcome-independent proposals; disjoint C AND P folds for selection.

    A held-out sign is a noisy validation rate, not an unbiased estimate of the
    prevalence of true oracle improvements. Simultaneous lower bounds supply a
    separate conservative evidence rate against ALL original retained actions.
    """
    wide = rec['wide']
    wide.pop('summary', None)
    if np.any(data['status']):
        wide['metrics_excluded'] = 'incomplete_crossed_grid'
        return
    wide.pop('metrics_excluded', None)
    K, N = len(rec['candidate_paths']), len(wide['actions'])
    summary = dict(available_actions=N, requested_actions=wide['requested'],
                   shortfall=wide['shortfall'], proposal_draws=wide['draws'], has_extra=float(N > 0))
    wide['summary'] = summary
    if not N:
        # Shortfall roots remain in discovery denominators; no fabricated gap.
        for budget in wide_budgets(wide['requested']):
            for key in ('examined', 'train_positive_fraction', 'heldout_win_fraction',
                        'evidence_fraction', 'evidence_unresolved_fraction', 'promoted_gap'):
                summary[f'discovery/{budget}/{key}'] = 0.
        return
    raw = data['outcome']
    x = np.concatenate([raw[:K], raw[K+1:], policy_baseline(rec, data)[None]])
    est = crossed_cov(x)
    q, mc = est['mean'], est['mc']
    advantage = q[K:K+N]-q[-1]
    avar = np.diag(mc)[K:K+N]+mc[-1, -1]-2*mc[K:K+N, -1]
    post = data['post_v'][K+1:]
    joint = crossed_cov(np.concatenate([x[K:K+N], post]))
    for i, action in enumerate(wide['actions']):
        se = float(np.sqrt(avar[i])) if avar[i] >= 0 else None
        action.update(bucket=pi_bucket(action['log_pi']), q=float(q[K+i]),
                      advantage=float(advantage[i]), advantage_mc_variance=float(avar[i]),
                      advantage_normal_95=[float(advantage[i]-1.96*se), float(advantage[i]+1.96*se)] if se is not None else None,
                      post_v=float(joint['mean'][N+i]))
    for bucket in ['all', 'lt_1e-6', '1e-6_to_1e-4', '1e-4_to_1e-3', '1e-3_to_1e-2', 'ge_1e-2']:
        take = np.array([i for i, a in enumerate(wide['actions']) if bucket == 'all' or a['bucket'] == bucket], int)
        if not len(take):
            continue
        a, v = advantage[take], avar[take]
        prefix = f'distribution/{bucket}/'
        summary[prefix+'actions'] = len(take)
        summary[prefix+'mean_advantage'] = float(a.mean())
        summary[prefix+'second_moment_mc_corrected'] = float(np.mean(a*a-v))
        summary[prefix+'mean_mc_variance'] = float(v.mean())
        # Root-balanced empirical CDF of noisy estimates; never call this a
        # deconvolved true-Q distribution or use its signs as proof of gain.
        for edge in (-.5, -.1, 0., .1, .5):
            summary[prefix+f'observed_cdf_at_{edge:g}'] = float(np.mean(a <= edge))
    M = np.eye(N)/N-np.ones((N, N))/N**2
    qm, vm = joint['mean'][:N], joint['mean'][N:]
    summary.update({
        'value/q_variance_corrected': float(qm@M@qm-np.trace(M@joint['mc'][:N, :N])),
        'value/v_variance_corrected': float(vm@M@vm-np.trace(M@joint['mc'][N:, N:])),
        'value/q_v_covariance_corrected': float(qm@M@vm-np.trace(M@joint['mc'][N:, :N]))})
    C, P = x.shape[1:]
    folds = [(slice(0, C//2), slice(0, P//2), slice(C//2, None), slice(P//2, None)),
             (slice(C//2, None), slice(P//2, None), slice(0, C//2), slice(0, P//2))]
    wide['discovery'] = {}
    for budget in wide_budgets(wide['requested']):
        n, results = min(budget, N), []
        for tc, tp, ec, ep in folds:
            train = x[:, tc, tp].mean((1, 2))
            test = crossed_cov(x[:, ec, ep])
            top = int(train[:K].argmax())
            chosen = K+int(train[K:K+n].argmax())
            h = np.zeros(len(x)); h[chosen] = 1; h[top] = -1
            gap = float(h@test['mean'])
            variance = contrast(test['mc'], h)
            train_gap = float(train[chosen]-train[top])
            # Two folds and K comparisons: one-sided Bonferroni normal bounds.
            z = NormalDist().inv_cdf(1-.05/(2*K))
            deltas = test['mean'][chosen]-test['mean'][:K]
            variances = test['mc'][chosen, chosen]+np.diag(test['mc'])[:K]-2*test['mc'][chosen, :K]
            identified = bool(np.all(variances >= 0))
            lower = deltas-z*np.sqrt(np.maximum(variances, 0))
            value_choice = K+int(post[:n, tc, tp].mean((1, 2)).argmax())
            value_gap = float(test['mean'][value_choice]-test['mean'][K:K+n].mean())
            results.append(dict(top_candidate=top, wide_candidate=chosen-K, train_gap=train_gap,
                                heldout_gap=gap, heldout_mc_variance=variance,
                                heldout_normal_95=[gap-1.96*np.sqrt(variance), gap+1.96*np.sqrt(variance)] if variance >= 0 else None,
                                beats_all_original_lower_bound=float(lower.min()) if identified else None,
                                evidence=bool(identified and np.all(lower > 0)),
                                value_candidate=value_choice-K, value_gain_vs_wide_mean=value_gap,
                                value_selected_advantage=float(test['mean'][value_choice]-test['mean'][-1])))
        wide['discovery'][str(budget)] = results
        prefix = f'discovery/{budget}/'
        summary.update({prefix+'examined': n,
                        prefix+'train_positive_fraction': float(np.mean([r['train_gap'] > 0 for r in results])),
                        prefix+'heldout_gap': float(np.mean([r['heldout_gap'] for r in results])),
                        prefix+'heldout_win_fraction': float(np.mean([r['heldout_gap'] > 0 for r in results])),
                        prefix+'evidence_fraction': float(any(r['evidence'] for r in results)),
                        prefix+'evidence_unresolved_fraction': float(any(r['beats_all_original_lower_bound'] is None for r in results)),
                        prefix+'promoted_gap': float(np.mean([r['heldout_gap'] if r['train_gap'] > 0 else 0. for r in results])),
                        prefix+'value_gain_vs_wide_mean': float(np.mean([r['value_gain_vs_wide_mean'] for r in results])),
                        prefix+'value_selected_advantage': float(np.mean([r['value_selected_advantage'] for r in results]))})


def analyze_root(rec, data):
    """Complete-case grids only: never treat truncation or failure as a draw."""
    if 'wide' in rec:
        analyze_wide_root(rec, data)
        K = len(rec['candidate_paths'])
        data = {k: v[:K+1] if k in ('outcome', 'gae', 'post_v', 'next_own_v', 'status') else v
                for k, v in data.items()}
    if np.any(data['status']):
        rec.pop('metrics', None)
        rec['metrics_excluded'] = 'incomplete_crossed_grid'
        return
    rec.pop('metrics_excluded', None)
    raw_outcomes = data['outcome']
    x = raw_outcomes.copy()
    x[-1] = policy_baseline(rec, data)
    cov = crossed_cov(x)
    q, mc, total = cov['mean'], cov['mc'], cov['total']
    K = len(rec['candidate_paths'])
    pi = np.asarray(rec['candidate_pi'], np.float64)
    w = pi/pi.sum()
    M = np.diag(w)-np.outer(w, w)
    raw_signal = float(q[:K]@M@q[:K])
    signal = raw_signal-float(np.trace(M@mc[:K, :K]))
    uniform_M = np.eye(K)/K-np.ones((K, K))/K**2
    noise = float(w@np.diag(total)[:K])
    second_moment = float(policy_baseline(rec, dict(data, outcome=raw_outcomes**2)).mean())
    full_policy_single_variance = second_moment-q[-1]**2+mc[-1, -1]
    actual = rec['actual_index']
    d = np.zeros(K+1); d[actual] = 1; d[-1] = -1
    best = int(q[:K].argmax())
    b = np.zeros(K+1); b[best] = 1; b[-1] = -1
    gap = float(b@q)
    gap2 = gap**2-contrast(mc, b)
    # Selection and evaluation use disjoint chance AND policy seed sets.
    C, P = x.shape[1:]
    splits = [(x[:, :C//2, :P//2], x[:, C//2:, P//2:]),
              (x[:, C//2:, P//2:], x[:, :C//2, :P//2])]
    heldout = []
    if min(C//2, P//2) >= 2:
        for train, test in splits:
            picked = int(train[:K].mean((1, 2)).argmax())
            estimate = crossed_cov(test)
            h = np.zeros(K+1); h[picked] = 1; h[-1] = -1
            delta = float(h@estimate['mean'])
            squared = delta**2-contrast(estimate['mc'], h)
            paired = contrast(estimate['total'], h)
            independent = float(estimate['total'][picked, picked]+full_policy_single_variance)
            heldout.append(dict(candidate=picked, gap=delta, gap_squared_corrected=squared,
                                variance_paired=paired, variance_independent=independent,
                                n_per_arm_independent_2sigma=required(squared, independent) if delta > 0 else None,
                                n_pairs_crn_2sigma=required(squared, paired) if delta > 0 else None,
                                worlds_per_crn_replicate=K+1))
    # All unordered candidate pairs, weighted by pi(a)pi(a'); excludes pseudo arm.
    pair_weights = np.outer(w, w)
    upper = np.triu(np.ones((K, K), bool), 1)
    weights = pair_weights[upper]
    independent = (np.diag(total)[:K, None]+np.diag(total)[None, :K])[upper]
    paired = (np.diag(total)[:K, None]+np.diag(total)[None, :K]-2*total[:K, :K])[upper]
    grid_ind = (np.diag(mc)[:K, None]+np.diag(mc)[None, :K])[upper]
    grid_pair = (np.diag(mc)[:K, None]+np.diag(mc)[None, :K]-2*mc[:K, :K])[upper]
    std = np.sqrt(np.maximum(np.diag(total), 0.))
    denom = np.outer(std, std)
    correlation = np.divide(total, denom, out=np.full_like(total, np.nan), where=denom > 0)
    def pairmean(v):
        return float(weights@v/weights.sum()) if weights.sum() else None
    chance, policy, interaction = (float(w@np.diag(cov[k])[:K]) for k in ('chance', 'policy', 'interaction'))
    # Joint MC covariance corrects regressions involving post-action means.
    joint = crossed_cov(np.stack([x[actual], data['post_v'][actual], data['next_own_v'][actual],
                                  x[-1], data['gae'][actual]]))
    all_post = crossed_cov(np.concatenate([x[:K], data['post_v'][:K]]))
    post_mean = all_post['mean'][K:]
    post_variance = float(post_mean@M@post_mean-np.trace(M@all_post['mc'][K:, K:]))
    post_covariance = float(q[:K]@M@post_mean-np.trace(M@all_post['mc'][K:, :K]))
    gaecov = crossed_cov(data['gae'])
    metrics = dict(signal_raw=raw_signal, signal=signal, outcome_noise=noise,
                   unweighted_signal=float(q[:K]@uniform_M@q[:K]-np.trace(uniform_M@mc[:K, :K])),
                   chance_main=chance, policy_main=policy, interaction=interaction,
                   chance_frozen_variance=policy+interaction, policy_frozen_variance=chance+interaction,
                   chance_shapley=chance+.5*interaction, policy_shapley=policy+.5*interaction,
                   pair_variance_independent=pairmean(independent), pair_variance_crn=pairmean(paired),
                   pair_grid_mean_variance_independent=pairmean(grid_ind), pair_grid_mean_variance_crn=pairmean(grid_pair),
                   pair_outcome_correlation=pairmean(correlation[:K, :K][upper]) if np.isfinite(correlation[:K, :K][upper]).all() else None,
                   post_action_signal_variance=post_variance, post_action_signal_covariance=post_covariance,
                   full_policy_single_game_variance=full_policy_single_variance,
                   best_gap_optimistic=gap, best_gap_squared_corrected_selection_biased=gap2,
                   heldout_best_gap=float(np.mean([h['gap'] for h in heldout])) if heldout else None,
                   heldout_best_gap_squared=float(np.mean([h['gap_squared_corrected'] for h in heldout])) if heldout else None,
                   heldout_variance_independent=float(np.mean([h['variance_independent'] for h in heldout])) if heldout else None,
                   heldout_variance_crn=float(np.mean([h['variance_paired'] for h in heldout])) if heldout else None,
                   actual_advantage=float(d@q), actual_advantage_mc_variance=contrast(mc, d),
                   gae_conditional_mean=float(gaecov['mean'][actual]),
                   gae_conditional_variance=float(gaecov['total'][actual, actual]),
                   gae_conditional_mean_mc_variance=float(gaecov['mc'][actual, actual]),
                   gae_advantage_mc_covariance=float(joint['mc'][4, 0]-joint['mc'][4, 3]),
                   q_policy=float(q[-1]), q_actual=float(q[actual]), post_v=float(joint['mean'][1]),
                   next_own_v=float(joint['mean'][2]), q_policy_mc_variance=float(mc[-1, -1]),
                   q_actual_mc_variance=float(mc[actual, actual]), post_v_mc_variance=float(joint['mc'][1, 1]),
                   next_own_v_mc_variance=float(joint['mc'][2, 2]),
                   post_q_mc_covariance=float(joint['mc'][1, 0]), next_own_q_mc_covariance=float(joint['mc'][2, 0]))
    rec.update(metrics=metrics, q=q.tolist(), q_grid_mc_covariance=mc.tolist(),
               q_mc_se=[float(np.sqrt(v)) if v >= 0 else None for v in np.diag(mc)],
               q_normal_95=[[float(qi-1.96*np.sqrt(v)), float(qi+1.96*np.sqrt(v))] if v >= 0 else None
                            for qi, v in zip(q, np.diag(mc))],
               baseline_method='conditional_tail' if rec['tail_mass'] > 1e-7 else 'pi_control_variate',
               outcome_covariance=total.tolist(), outcome_components={k: cov[k].tolist() for k in ('chance', 'policy', 'interaction')},
               outcome_correlation=[[float(v) if np.isfinite(v) else None for v in row] for row in correlation],
               heldout_best=heldout)


def safe_ratio(x, y):
    return float(x/y) if y > 0 else None


def regression(x, y, vx=0., vy=0., cxy=0.):
    """Across-root centered moments, corrected for heteroskedastic MC error."""
    n = len(x)
    if n < 2:
        return dict(correlation=None, r2=None, residual_mse=None)
    varx, vary = np.var(x, ddof=1)-np.mean(vx), np.var(y, ddof=1)-np.mean(vy)
    cov = np.cov(x, y, ddof=1)[0, 1]-np.mean(cxy)
    return dict(correlation=float(cov/np.sqrt(varx*vary)) if min(varx, vary) > 0 else None,
                r2=float(cov**2/(varx*vary)) if min(varx, vary) > 0 else None,
                residual_mse=float(np.mean((np.asarray(x)-y)**2)-np.mean(vx)-np.mean(vy)+2*np.mean(cxy)))


def aggregate(rows):
    rows = [r for r in rows if 'metrics' in r and not r['real_truncated']]
    result = dict(roots=len(rows), games=len({r['game'] for r in rows}))
    if not rows:
        return result
    def col(k):
        return np.array([r['metrics'][k] for r in rows], np.float64)
    for k in rows[0]['metrics']:
        vals = col(k)
        if np.isfinite(vals).any():
            result[k] = float(np.nanmean(vals))
    m = result
    for label, num, den in [('snr_variance', 'signal', 'outcome_noise'),
                            ('chance_main_fraction', 'chance_main', 'outcome_noise'),
                            ('policy_main_fraction', 'policy_main', 'outcome_noise'),
                            ('interaction_fraction', 'interaction', 'outcome_noise'),
                            ('chance_shapley_fraction', 'chance_shapley', 'outcome_noise'),
                            ('policy_shapley_fraction', 'policy_shapley', 'outcome_noise'),
                            ('crn_pair_variance_ratio', 'pair_variance_crn', 'pair_variance_independent'),
                            ('crn_grid_variance_ratio', 'pair_grid_mean_variance_crn', 'pair_grid_mean_variance_independent')]:
        if num in m and den in m:
            m[label] = safe_ratio(m[num], m[den])
    if 'heldout_best_gap_squared' in m:
        m['heldout_n_per_arm_2sigma'] = (required(m['heldout_best_gap_squared'], m['heldout_variance_independent'])
                                        if m['heldout_best_gap'] > 0 else None)
        m['heldout_n_crn_pairs_2sigma'] = (required(m['heldout_best_gap_squared'], m['heldout_variance_crn'])
                                        if m['heldout_best_gap'] > 0 else None)
    if 'post_action_signal_variance' in m:
        m['post_value_within_root_q_r2'] = (m['post_action_signal_covariance']**2/(m['signal']*m['post_action_signal_variance'])
                                           if min(m['signal'], m['post_action_signal_variance']) > 0 else None)
    # Report the exact variance identity. GAE need not be unbiased for true A.
    h, a = np.array([r['gae'] for r in rows]), col('actual_advantage')
    va = col('actual_advantage_mc_variance')
    if len(rows) >= 2:
        var_h, var_a = float(np.var(h, ddof=1)), float(np.var(a, ddof=1)-va.mean())
        cov_ha = float(np.cov(h, a, ddof=1)[0, 1])
        var_e = float(np.var(h-a, ddof=1)-va.mean())
        cross = 2*(cov_ha-var_a)
        m.update(gae_variance=var_h, true_advantage_variance=var_a,
                 gae_error_variance=var_e, gae_cross_term=cross,
                 gae_variance_identity_residual=var_h-var_a-var_e-cross,
                 gae_advantage_correlation=float(cov_ha/np.sqrt(var_h*var_a)) if min(var_h, var_a) > 0 else None,
                 gae_advantage_correlation_raw=regression(h, a)['correlation'],
                 gae_bias=float(np.mean(h-a)))
        m['advantage_mc_to_true_variance_ratio'] = safe_ratio(float(va.mean()), var_a)
        # Independent continuation estimate of E[GAE|s,a] reveals bootstrap bias.
        b = col('gae_conditional_mean')-a
        vb = col('gae_conditional_mean_mc_variance')+va-2*col('gae_advantage_mc_covariance')
        m['gae_conditional_bias_mse'] = float(np.mean(b*b-vb))
        var_bias = float(np.var(b, ddof=1)-np.mean(vb))
        cov_ab = float(np.cov(a, b, ddof=1)[0, 1]-np.mean(col('gae_advantage_mc_covariance')-va))
        m.update(gae_conditional_bias_variance=var_bias, gae_conditional_bias_cross_term=2*cov_ab,
                 gae_total_variance_prediction=var_a+var_bias+2*cov_ab+m['gae_conditional_variance'])
    pairs = [
        ('value_policy_q', np.array([r['v'] for r in rows]), col('q_policy'), 0., col('q_policy_mc_variance'), 0.),
        ('value_actual_q', np.array([r['v'] for r in rows]), col('q_actual'), 0., col('q_actual_mc_variance'), 0.),
        ('post_value_actual_q', col('post_v'), col('q_actual'), col('post_v_mc_variance'), col('q_actual_mc_variance'), col('post_q_mc_covariance')),
        ('next_own_value_actual_q', col('next_own_v'), col('q_actual'), col('next_own_v_mc_variance'), col('q_actual_mc_variance'), col('next_own_q_mc_covariance'))]
    for label, xx, yy, vx, vy, cv in pairs:
        for key, value in regression(xx, yy, vx, vy, cv).items():
            m[label+'_'+key] = value
    m['mean_candidate_mass'] = float(np.mean([r['candidate_mass'] for r in rows]))
    m['coverage_met_fraction'] = float(np.mean([r['coverage_met'] for r in rows]))
    m['top_certified_fraction'] = float(np.mean([r['top_certified'] for r in rows]))
    m['negative_signal_root_fraction'] = float(np.mean(col('signal') < 0))
    return m


def summarize(rows, games, bootstrap, seed):
    """Resample deal-pair clusters, preserving both seats and all their roots."""
    groups = {'overall': lambda r: True, 'main': lambda r: r['phase'] == 'main',
              'response': lambda r: r['phase'] == 'response',
              'coverage_met': lambda r: r.get('coverage_met', False),
              'coverage_missed': lambda r: not r.get('coverage_met', False),
              'multi_candidate': lambda r: len(r.get('candidate_paths', [])) > 1,
              'early': lambda r: r['progress'] < 1/3, 'middle': lambda r: 1/3 <= r['progress'] < 2/3,
              'late': lambda r: r['progress'] >= 2/3,
              'start_0_9': lambda r: r['decision_index'] < 10,
              'start_10_29': lambda r: 10 <= r['decision_index'] < 30,
              'start_30_plus': lambda r: r['decision_index'] >= 30,
              'end_1_9': lambda r: r['decisions_to_end'] < 10,
              'end_10_29': lambda r: 10 <= r['decisions_to_end'] < 30,
              'end_30_plus': lambda r: r['decisions_to_end'] >= 30}
    for phase in ('main', 'response'):
        for stage in ('early', 'middle', 'late'):
            groups[phase+'_'+stage] = lambda r, phase=phase, stage=stage: r['phase'] == phase and groups[stage](r)
    clusters = sorted({g['g']//2 for g in games})
    rng = np.random.default_rng(seed)
    draws = [rng.choice(clusters, len(clusters), replace=True) for _ in range(bootstrap)] if clusters else []
    result = {}
    for label, predicate in groups.items():
        selected = [r for r in rows if predicate(r)]
        estimates = aggregate(selected)
        by_cluster = {g: [r for r in selected if r['game']//2 == g] for g in clusters}
        intervals, valid_fraction = {}, {}
        samples = [aggregate([r for g in draw for r in by_cluster[g]]) for draw in draws]
        for key, value in estimates.items():
            if key in ('roots', 'games'):
                continue
            values = [s[key] for s in samples if s.get(key) is not None and np.isfinite(s[key])]
            valid_fraction[key] = len(values)/max(len(samples), 1)
            if value is None or not values:
                intervals[key] = None
            elif key in ('heldout_n_per_arm_2sigma', 'heldout_n_crn_pairs_2sigma'):
                # Nonpositive/undetected gaps mean an unbounded sample budget,
                # not an observation to drop from the percentile distribution.
                ordered = sorted(values)
                intervals[key] = [ordered[int(q*(len(samples)-1))] if int(q*(len(samples)-1)) < len(ordered) else None
                                  for q in (.025, .975)]
            elif len(values) < .975*len(samples):
                intervals[key] = None
            else:
                intervals[key] = np.quantile(values, [.025, .975]).tolist()
        result[label] = dict(estimate=estimates, bootstrap_deal_pair_95=intervals,
                             bootstrap_valid_fraction=valid_fraction)
    return result


def summarize_wide(rows, games, bootstrap, seed):
    """Root-balanced statistics, with root and paired-deal cluster intervals.

    A bucket conditions on roots having at least one sampled action in it, then
    weights actions equally within each root. Discovery includes shortfall roots.
    Missing/invalid statistics remain missing in bootstrap replicates.
    """
    valid = [r for r in rows if not r['real_truncated'] and 'summary' in r.get('wide', {})]
    groups = {'overall': valid}
    for phase in ('main', 'response'):
        groups[phase] = [r for r in valid if r['phase'] == phase]
    for i, stage in enumerate(('early', 'middle', 'late')):
        groups[stage] = [r for r in valid if min(int(r['progress']*3), 2) == i]
        for phase in ('main', 'response'):
            groups[phase+'_'+stage] = [r for r in groups[stage] if r['phase'] == phase]
    clusters = sorted({g['g']//2 for g in games})
    rng = np.random.default_rng(seed)
    cluster_weights = rng.multinomial(len(clusters), np.full(len(clusters), 1/max(len(clusters), 1)), size=bootstrap) if clusters else np.zeros((bootstrap, 0))
    result = {}
    for label, selected in groups.items():
        n = len(selected)
        if not n:
            result[label] = dict(roots=0)
            continue
        keys = sorted({k for r in selected for k in r['wide']['summary']})
        matrix = np.array([[r['wide']['summary'].get(k, np.nan) for k in keys] for r in selected], np.float64)
        available = np.isfinite(matrix)
        filled = np.where(available, matrix, 0.)
        def means(weights):
            den = weights@available.astype(float)
            return np.divide(weights@filled, den, out=np.full_like(den, np.nan), where=den > 0)
        point = means(np.ones((1, n)))
        root_weights = rng.multinomial(n, np.full(n, 1/n), size=bootstrap)
        deal_weights = cluster_weights[:, [clusters.index(r['game']//2) for r in selected]]
        samples = {'root': means(root_weights), 'deal_pair': means(deal_weights)}
        # Corrected within-root moments quantify discrimination of low-pi hands.
        derived = ['value/within_root_correlation', 'value/within_root_r2']
        def derived_values(values):
            qv, vv, cv = (values[:, keys.index('value/'+k)] for k in
                          ('q_variance_corrected', 'v_variance_corrected', 'q_v_covariance_corrected'))
            den = qv*vv
            ok = (qv > 0) & (vv > 0)
            correlation = np.divide(cv, np.sqrt(np.maximum(den, 0)), out=np.full_like(cv, np.nan), where=ok)
            return np.column_stack([values, correlation, correlation**2])
        if 'value/q_variance_corrected' in keys:
            point = derived_values(point)
            samples = {k: derived_values(v) for k, v in samples.items()}
            keys += derived
        estimates = {k: float(v) if np.isfinite(v) else None for k, v in zip(keys, point[0])}
        output = dict(roots=n, games=len({r['game'] for r in selected}), estimate=estimates,
                      metric_roots={k: int(available[:, i].sum()) for i, k in enumerate(keys[:matrix.shape[1]])})
        for kind, values in samples.items():
            intervals, fractions = {}, {}
            for i, k in enumerate(keys):
                good = values[:, i][np.isfinite(values[:, i])]
                fractions[k] = len(good)/bootstrap
                intervals[k] = np.quantile(good, [.025, .975]).tolist() if estimates[k] is not None and len(good) >= .975*bootstrap else None
            output['bootstrap_'+kind+'_95'] = intervals
            output['bootstrap_'+kind+'_valid_fraction'] = fractions
        result[label] = output
    return result


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')


def collection_summary(games):
    """Diagnostic-seat score rate; never quietly count truncations as draws."""
    complete = [g for g in games if not g['truncated']]
    wins = sum(g['result'] == 1 for g in complete)
    losses = sum(g['result'] == -1 for g in complete)
    draws = len(complete)-wins-losses
    return dict(games=len(games), completed=len(complete), truncated=len(games)-len(complete),
                wins=wins, losses=losses, draws=draws,
                win_rate=(wins+.5*draws)/len(complete) if complete else None,
                strict_win_rate=wins/len(complete) if complete else None,
                evaluation_win_rate=(wins+.5*(len(games)-wins-losses))/len(games) if games else None,
                by_seat={str(seat): dict(games=sum(g['agent_seat'] == seat for g in games),
                    completed=sum(g['agent_seat'] == seat for g in complete),
                    win_rate=float(np.mean([(g['result']+1)/2 for g in complete if g['agent_seat'] == seat]))
                    if any(g['agent_seat'] == seat for g in complete) else None) for seat in (0, 1)})


def save_report(args, records, games, targets, start, ready, final=False):
    completed = [r for r in records if 'world_counts' in r]
    seconds = perf_counter()-start
    totals = {k: sum(r['world_counts'][k] for r in completed) for k in ('requested', 'completed', 'load_failed', 'truncated')}
    decision_count = sum(r['continuation_decisions'] for r in completed)
    report = dict(schema_version=1, config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                          if (not k.startswith('wide_') or getattr(args, 'wide_candidates', 0))
                                          and (not k.startswith('opponent_') or getattr(args, 'opponent_checkpoint', None))},
                  runtime=dict(python=sys.version, torch=str(torch.__version__), platform=platform.platform(),
                               cuda=torch.version.cuda, device_name=torch.cuda.get_device_name() if args.device == 'cuda' else platform.machine()),
                  summary_config=dict(bootstrap=args.bootstrap, seed=args.seed+9, cluster='deal_pair'),
                  final=final, methodology=__doc__, roots=records, games=games,
                  selection=dict(method=args.root_selection, targets_by_game=targets, selected=len(records),
                                 selected_by_game=[[r['decision_index'] for r in records if r['game'] == g] for g in range(args.games)],
                                 missed=args.games*args.roots_per_game-len(records)),
                  counts=dict(**totals, measured_roots=len(completed),
                              metric_roots=sum('metrics' in r and not r['real_truncated'] for r in records),
                              real_truncated_games=sum(bool(g['truncated']) for g in games)),
                  timing=dict(total_seconds=seconds, collection_seconds=ready-start,
                              continuation_decisions=decision_count, diagnostic_seconds=perf_counter()-ready,
                              continuation_decisions_per_second=decision_count/max(perf_counter()-ready, 1e-12)),
                  summary=summarize(completed, games, args.bootstrap, args.seed+9) if final else None)
    if getattr(args, 'wide_candidates', 0):
        report['schema_version'] = 2
        report['wide_summary'] = summarize_wide(completed, games, args.bootstrap, args.seed+10) if final else None
        report['summary_config']['wide_seed'] = args.seed+10
    if getattr(args, 'opponent_checkpoint', None):
        report['schema_version'] = 3
        report['collection_summary'] = collection_summary(games)
        report['selection']['decision_scope'] = 'diagnostic_seat'
        report['selection']['selected_diagnostic_indices_by_game'] = [
            [r['diagnostic_decision_index'] for r in records if r['game'] == g] for g in range(args.games)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix+'.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False, default=_json_default)+'\n')
    temp.replace(args.out)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/hp-lr1e5/ckpt-4500.pt')
    p.add_argument('--opponent-checkpoint', help='fixed opponent; omit for the exact legacy self-play diagnostic')
    p.add_argument('--opponent-weights', choices=('online', 'ema'), default='online')
    p.add_argument('--opponent-dtype', choices=('auto', 'fp32', 'bf16', 'fp16'), default='auto',
                   help='independent opponent inference precision; auto follows its checkpoint on CUDA')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--root-selection', choices=('reservoir', 'window'), default='reservoir',
                   help='reservoir: uniform decisions across each complete game; window: fixed pre-drawn indices')
    p.add_argument('--weights', choices=('online', 'ema'), default='online',
                   help='online uses checkpoint model, matching the training collector')
    p.add_argument('--dtype', choices=('auto', 'fp32', 'bf16', 'fp16'), default='auto',
                   help='auto: checkpoint training dtype on CUDA; fp32 on CPU')
    p.add_argument('--out', type=Path, default=Path('measurements/luck-diagnostic.json'))
    p.add_argument('--summarize', type=Path, help='recompute summaries/intervals from an existing report; no model needed')
    p.add_argument('--wide-candidates', type=int, default=0,
                   help='extra distinct low-pi actions per root; 0 preserves the original diagnostic')
    p.add_argument('--wide-method', choices=('prefix-uniform', 'tempered'), default='prefix-uniform')
    p.add_argument('--wide-temperature', type=float, default=2.,
                   help='tempered proposal uses pi(choice|prefix) ** (1/temperature)')
    p.add_argument('--wide-max-pi', type=float, default=.01,
                   help='maximum unconditional joint policy probability of an extra action')
    p.add_argument('--wide-draws', type=int, default=1024,
                   help='bounded proposal attempts per root; report shortfalls without replacement')
    for name, default in [('games', 128), ('roots-per-game', 2), ('target-range', 100), ('num-envs', 32),
                          ('chance-repeats', 16), ('policy-repeats', 16), ('max-candidates', 32), ('min-candidates', 2),
                          ('candidate-nodes', 2048), ('prefix-batch', 32), ('world-batch', 2048),
                          ('inference-batch', 256), ('cache-mb', 512), ('threads', 4),
                          ('max-decisions', 10000), ('bootstrap', 1000), ('seed', 20261117)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('mass', .95), ('lambda-v', .8), ('lambda-a', .5), ('gamma', 1.)]:
        p.add_argument('--'+name, type=float, default=default)
    return p


def load_model(args, p, *, opponent=False):
    prefix = 'opponent_' if opponent else ''
    path = getattr(args, prefix+'checkpoint')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    with Path(path).open('rb') as source:
        setattr(args, prefix+'checkpoint_sha256', hashlib.file_digest(source, 'sha256').hexdigest())
    setattr(args, prefix+'model_config', checkpoint['model_config'])
    model = Model(ModelConfig(**checkpoint['model_config']))
    weights = getattr(args, prefix+'weights')
    model.load_state_dict(checkpoint['model' if weights == 'online' else 'ema'])
    model = model.to(args.device).eval().requires_grad_(False)
    setattr(args, prefix+'checkpoint_iteration', int(checkpoint['iteration']) if 'iteration' in checkpoint else None)
    training = checkpoint.get('config', {})
    setattr(args, prefix+'training_dtype', training.get('dtype'))
    setattr(args, prefix+'training_tf32', bool(training.get('tf32', False)))
    dtype = getattr(args, prefix+'dtype')
    if dtype == 'auto':
        dtype = training.get('dtype', 'fp32') if args.device == 'cuda' else 'fp32'
        setattr(args, prefix+'dtype', dtype)
    if dtype not in ('fp32', 'bf16', 'fp16') or (args.device == 'cpu' and dtype != 'fp32'):
        p.error(prefix+'dtype: supported dtype is fp32 on CPU; fp32/bf16/fp16 on CUDA')
    return model


def main():
    p = parser(); args = p.parse_args()
    if args.bootstrap <= 0 or args.seed < 0:
        p.error('bootstrap must be positive and seed nonnegative')
    if args.summarize:
        report = json.loads(args.summarize.read_text())
        report['summary'] = summarize(report['roots'], report['games'], args.bootstrap, args.seed+9)
        report['summary_config'] = dict(bootstrap=args.bootstrap, seed=args.seed+9, cluster='deal_pair')
        if 'wide_summary' in report or any('wide' in r for r in report['roots']):
            report['wide_summary'] = summarize_wide(report['roots'], report['games'], args.bootstrap, args.seed+10)
            report['summary_config']['wide_seed'] = args.seed+10
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False, default=_json_default)+'\n')
        print(json.dumps(report['summary']['overall'], indent=2, allow_nan=False, default=_json_default)); return
    positive = ('games', 'roots_per_game', 'target_range', 'num_envs', 'chance_repeats', 'policy_repeats',
                'max_candidates', 'min_candidates', 'candidate_nodes', 'prefix_batch', 'world_batch', 'inference_batch',
                'threads', 'max_decisions', 'bootstrap')
    if any(getattr(args, k) <= 0 for k in positive) or min(args.chance_repeats, args.policy_repeats) < 4 or args.cache_mb < 0 or args.seed < 0:
        p.error('positive budgets, >=4 seeds per factor, nonnegative seed/cache required')
    if (not 0 < args.mass <= 1 or not 0 <= args.lambda_a <= 1 or not 0 <= args.lambda_v <= 1 or not 0 < args.gamma <= 1
            or (args.root_selection == 'window' and args.roots_per_game > args.target_range)):
        p.error('invalid mass, lambda, gamma or window roots-per-game > target-range')
    if args.min_candidates > args.max_candidates:
        p.error('min-candidates must be <= max-candidates')
    if (args.wide_candidates < 0 or args.wide_draws <= 0 or
            not 0 < args.wide_max_pi <= 1 or not np.isfinite(args.wide_temperature) or args.wide_temperature <= 0):
        p.error('wide-candidates >=0, wide-draws >0, 0 < wide-max-pi <=1 and finite wide-temperature >0 required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    torch.set_num_threads(args.threads)
    start = perf_counter()
    model = load_model(args, p)
    opponent_model = load_model(args, p, opponent=True) if args.opponent_checkpoint else None
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = args.training_tf32
    def precision():
        return torch.autocast(args.device, dtype=torch.bfloat16 if args.dtype == 'bf16' else torch.float16,
                              enabled=args.dtype != 'fp32')
    with precision():
        archive, records, games, targets = collect(args, model, opponent_model)
    ready = perf_counter()
    rawdir = args.out.with_suffix('.samples')
    rawdir.mkdir(parents=True, exist_ok=True)
    save_report(args, records, games, targets, start, ready)
    final = False
    try:
        for number, rec in enumerate(records, 1):
            since = perf_counter()
            with precision():
                candidates(args, model, archive, rec)
                wide_candidates(args, model, archive, rec)
                data = panel(args, model, archive, rec, opponent_model)
            rawpath = rawdir/f'root-{rec["archive_row"]:05d}.npz'
            np.savez_compressed(rawpath, **data)
            rec['samples'] = str(rawpath)
            analyze_root(rec, data)
            rec['seconds'] = perf_counter()-since
            save_report(args, records, games, targets, start, ready)
            print(f'root {number}/{len(records)}: {len(rec["candidate_paths"])} candidates, '
                  f'mass={rec["candidate_mass"]:.4f}, {rec["seconds"]:.1f}s, {rec["world_counts"]}', flush=True)
        final = True
    finally:
        report = save_report(args, records, games, targets, start, ready, final=True)
        report['final'] = final
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False, default=_json_default)+'\n')
    output = dict(counts=report['counts'], timing=report['timing'], overall=report['summary']['overall']['estimate'])
    if 'collection_summary' in report:
        output['collection_summary'] = report['collection_summary']
    print(json.dumps(output, indent=2, allow_nan=False, default=_json_default))
    print(f'Saved {args.out}; raw crossed panels: {rawdir}')


if __name__ == '__main__':
    main()
