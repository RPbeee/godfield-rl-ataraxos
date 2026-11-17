#!/usr/bin/env python3
"""Offline oracle Q / chance-policy noise / CRN / learner GAE diagnostic.

The native diagnostic_copy_exact entry point copies BOTH private histories and
all engine state. No determinization, public projection, search or leaf value
substitutes for terminal outcomes. CUDA is batched policy inference; C++ workers
simulate games on the host. Nothing in play/search/training imports this script.
"""
import argparse
from collections import Counter
import heapq
import hashlib
import json
from pathlib import Path
import platform
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


@torch.inference_mode()
def collect(args, model):
    """Uniform per-game reservoir, or pre-drawn fixed-window indices."""
    env = VecEnv(min(args.games, args.num_envs), args.threads, eval_pairs=(args.games+1)//2,
                 opponent='external', eval_seed=args.seed)
    archive = world(args, args.games*args.roots_per_game)
    rng = np.random.default_rng(args.seed+1)
    reservoir = args.root_selection == 'reservoir'
    targets = [set() if reservoir else set(rng.choice(args.target_range, args.roots_per_game, replace=False).tolist())
               for _ in range(args.games)]
    target_list = None if reservoir else [sorted(t) for t in targets]
    actor = Actor(model, env, args.device)
    lengths = Counter()
    sequences, pending, records_by_slot = {}, {}, {}
    phase_index = env.obs_schema()['cat']['phase']['index']
    last = perf_counter()
    with _policy_rng(args.seed+2, (actor,)):
        while not env.eval_done():
            active = np.flatnonzero(env.stage >= 0)
            fresh = active[env.substep[active] == 0]
            save, dest = [], []
            for r in fresh:
                g, seat = int(env.episode_id[r]), int(env.actor_seat[r])
                d = lengths[g]
                seq = sequences.setdefault((g, seat), [])
                idx = None
                if g < args.games:
                    if reservoir:
                        slot = d if d < args.roots_per_game else int(rng.integers(d+1))
                        if slot < args.roots_per_game:
                            idx = g*args.roots_per_game+slot
                    elif d in targets[g]:
                        targets[g].remove(d)
                        idx = len(records_by_slot)
                if idx is not None:
                    rec = dict(game=g, seat=seat, decision_index=d, sequence_index=len(seq),
                               phase='response' if env.obs_cat[r, 0, phase_index] == 2 else 'main',
                               actual_path=[], actual_log_pi=0., archive_row=idx)
                    records_by_slot[idx] = rec
                    pending[int(r)] = rec
                    save.append(r); dest.append(idx)
                lengths[g] += 1
            if save:
                copy_exact(archive, env, save, destinations=dest)
            result = actor.act(compact_transfer=True)
            probs = torch.from_numpy(result.value_logits).softmax(-1).numpy()
            for r in fresh:
                g, seat = int(env.episode_id[r]), int(env.actor_seat[r])
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

    def act(self, rows=None, **kwargs):
        groups = np.arange(self.env.num_envs)+1
        groups[self.env.decisions == 0] = 0
        kwargs.setdefault('observation_groups', groups)
        result = super().act(rows, **kwargs)
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


@torch.inference_mode()
def panel(args, model, archive, rec):
    paths = rec['candidate_paths']
    A, C, P = len(paths)+1, args.chance_repeats, args.policy_repeats
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, rec['game'], rec['decision_index'], 17]))
    chance_seeds = rng.integers(0, 2**64-1, C, dtype=np.uint64)
    policy_seeds = rng.integers(0, 2**64-1, P, dtype=np.uint64)
    shape = (A, C, P)
    output = {k: np.full(shape, np.nan) for k in ('outcome', 'gae', 'post_v', 'next_own_v')}
    tail_paths = np.full((C, P), -1, np.int32)  # retained action index, or -1 for tail
    status = np.zeros(shape, np.int8)  # 0 completed, 1 load failure, 2 truncated
    decisions = 0
    last = perf_counter()
    for lo in range(0, A*C*P, args.world_batch):
        flat = np.arange(lo, min(lo+args.world_batch, A*C*P))
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
        actor = DiagnosticActor(args, model, env, policy_seeds[pp])
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
                if aa[i] < len(paths):
                    choices[i] = paths[aa[i]][env.substep[i]]
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
    known = np.einsum('a,acp->cp', pi, x[:-1])
    if rec['tail_mass'] > 1e-7:
        return known+rec['tail_mass']*x[-1]
    # E[1{a in S} Y_a] = sum_S pi(a) Q_a. Per-cell CRN makes this
    # control variate cancel exactly whenever the sampled path is retained.
    residual = x[-1].copy()
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


def analyze_root(rec, data):
    """Complete-case grids only: never treat truncation or failure as a draw."""
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


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')


def save_report(args, records, games, targets, start, ready, final=False):
    completed = [r for r in records if 'world_counts' in r]
    seconds = perf_counter()-start
    totals = {k: sum(r['world_counts'][k] for r in completed) for k in ('requested', 'completed', 'load_failed', 'truncated')}
    decision_count = sum(r['continuation_decisions'] for r in completed)
    report = dict(schema_version=1, config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix+'.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False, default=_json_default)+'\n')
    temp.replace(args.out)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/hp-lr1e5/ckpt-4500.pt')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--root-selection', choices=('reservoir', 'window'), default='reservoir',
                   help='reservoir: uniform decisions across each complete game; window: fixed pre-drawn indices')
    p.add_argument('--weights', choices=('online', 'ema'), default='online',
                   help='online uses checkpoint model, matching the training collector')
    p.add_argument('--dtype', choices=('auto', 'fp32', 'bf16', 'fp16'), default='auto',
                   help='auto: checkpoint training dtype on CUDA; fp32 on CPU')
    p.add_argument('--out', type=Path, default=Path('measurements/luck-diagnostic.json'))
    p.add_argument('--summarize', type=Path, help='recompute summaries/intervals from an existing report; no model needed')
    for name, default in [('games', 128), ('roots-per-game', 2), ('target-range', 100), ('num-envs', 32),
                          ('chance-repeats', 16), ('policy-repeats', 16), ('max-candidates', 32), ('min-candidates', 2),
                          ('candidate-nodes', 2048), ('prefix-batch', 32), ('world-batch', 2048),
                          ('inference-batch', 256), ('cache-mb', 512), ('threads', 4),
                          ('max-decisions', 10000), ('bootstrap', 1000), ('seed', 20261117)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('mass', .95), ('lambda-v', .8), ('lambda-a', .5), ('gamma', 1.)]:
        p.add_argument('--'+name, type=float, default=default)
    return p


def main():
    p = parser(); args = p.parse_args()
    if args.bootstrap <= 0 or args.seed < 0:
        p.error('bootstrap must be positive and seed nonnegative')
    if args.summarize:
        report = json.loads(args.summarize.read_text())
        report['summary'] = summarize(report['roots'], report['games'], args.bootstrap, args.seed+9)
        report['summary_config'] = dict(bootstrap=args.bootstrap, seed=args.seed+9, cluster='deal_pair')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report['summary']['overall'], indent=2)); return
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
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    torch.set_num_threads(args.threads)
    start = perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    with Path(args.checkpoint).open('rb') as source:
        args.checkpoint_sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
    args.model_config = checkpoint['model_config']
    model = Model(ModelConfig(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model' if args.weights == 'online' else 'ema'])
    model = model.to(args.device).eval().requires_grad_(False)
    args.checkpoint_iteration = int(checkpoint['iteration']) if 'iteration' in checkpoint else None
    training = checkpoint.get('config', {})
    args.training_dtype = training.get('dtype')
    args.training_tf32 = bool(training.get('tf32', False))
    if args.dtype == 'auto':
        args.dtype = training.get('dtype', 'fp32') if args.device == 'cuda' else 'fp32'
    if args.dtype not in ('fp32', 'bf16', 'fp16') or (args.device == 'cpu' and args.dtype != 'fp32'):
        p.error('supported dtype is fp32 on CPU; fp32/bf16/fp16 on CUDA')
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = args.training_tf32
    del checkpoint
    def precision():
        return torch.autocast(args.device, dtype=torch.bfloat16 if args.dtype == 'bf16' else torch.float16,
                              enabled=args.dtype != 'fp32')
    with precision():
        archive, records, games, targets = collect(args, model)
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
                data = panel(args, model, archive, rec)
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
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(counts=report['counts'], timing=report['timing'],
                         overall=report['summary']['overall']['estimate']), indent=2))
    print(f'Saved {args.out}; raw crossed panels: {rawdir}')


if __name__ == '__main__':
    main()
