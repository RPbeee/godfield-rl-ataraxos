#!/usr/bin/env python3
"""Offline value luck floor using terminal self-play worlds (no network/build).

p is expected win score P(win)+0.5 P(draw). Each game pre-draws distinct target
external-decision indices uniformly from [0,100), across both seats. Targets
past the end or not loadable at that exact index are skipped, without replacement
by later decisions. This conditions the observed root distribution on survival
and loadability; distributions from start and to end are reported explicitly.
Real games run once. Selected rows freeze at substep zero until a root batch is
ready. BOTH variants' K worlds per root share ONE SearchEnv and policy batch.
Real-game Torch/engine RNG is isolated from continuation sampling and chance.

For n nontruncated scores X in {0,.5,1}, subtract s_X^2/n from (mean(X)-y)^2:
  [p_hat*(1-p_hat) - .25*draws/n] / (n-1).
This is unbiased for MC noise conditional on the root and real outcome; the
Bernoulli formula alone overcorrects draws. n<2 or real truncations are excluded
from metrics. Negative corrected estimates are retained. Truncation censoring
can bias estimates, so only zero-truncation runs support the floor interpretation.
The bound is conditional on existing load_worlds semantics (including projected
opponent history and public-prior own Dream reconstruction); info additionally depends on AR belief accuracy. Policy
sampling itself, as well as engine chance, contributes to irreducible randomness.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

import numpy as np
import torch
from gfrl import SearchEnv, VecEnv
from gfrl.actor import Actor
from gfrl.evaluation import _policy_rng, load_policy_model
from gfrl.search import OracleBelief, SearchAgent, SearchConfig, _SearchActor, load_search_belief

BUCKETS = (0, 5, 10, 20, 30, 45, 60)
METRICS = ('brier_value', 'brier_state_raw', 'brier_state_corrected',
           'brier_info_raw', 'brier_info_corrected', 'mean_abs_v_state',
           'reducible_part_upper_bound')


def bucket(decisions):
    if decisions < 0:
        raise ValueError('decisions must be nonnegative')
    i = int(np.searchsorted(BUCKETS, decisions, side='right')) - 1
    return f'[{BUCKETS[i]},{BUCKETS[i+1] if i+1 < len(BUCKETS) else "inf"})'


def brier_counts(wins, losses, draws, y):
    """Return expected score, raw Brier, unbiased corrected Brier; need n>=2."""
    if min(wins, losses, draws) < 0 or not 0 <= y <= 1:
        raise ValueError('invalid counts/outcome')
    n = wins + losses + draws
    if n < 2:
        raise ValueError('at least two completed continuations required')
    p = (wins + .5 * draws) / n
    raw = (p-y)**2
    correction = (p*(1-p) - .25*draws/n) / (n-1)
    return p, raw, raw-correction


def root_metrics(row):
    if row['real_truncated'] or any(row[v]['completed'] < 2 for v in ('state', 'info')):
        return None
    y = row['y']
    ps, sr, sc = brier_counts(*(row['state'][k] for k in ('wins', 'losses', 'draws')), y)
    _, ir, ic = brier_counts(*(row['info'][k] for k in ('wins', 'losses', 'draws')), y)
    bv = ((row['v']+1)/2-y)**2
    return np.array([bv, sr, sc, ir, ic, abs(row['v']-(2*ps-1)), bv-sc])


def summarize(rows, bootstrap=2000, seed=0):
    valid = [(r, root_metrics(r)) for r in rows]
    valid = [(r, m) for r, m in valid if m is not None]
    def stats(items):
        return dict(n=len(items), **dict(zip(METRICS, np.mean([m for _, m in items], axis=0).tolist()
                                          if items else [None]*len(METRICS))))
    overall = stats(valid)
    groups = {}
    for r, m in valid:
        groups.setdefault(r['game'], []).append(m)
    ci = {key: None for key in METRICS}
    if groups:
        # Resample whole games and keep the root-weighted estimator, including
        # variable root counts. Excluded/zero-root games remain empty clusters.
        game_ids = sorted({r['game'] for r in rows})
        sums = np.array([np.sum(groups[g], axis=0) if g in groups else np.zeros(len(METRICS))
                         for g in game_ids])
        counts = np.array([len(groups.get(g, [])) for g in game_ids])
        rng = np.random.default_rng(seed)
        means = []
        for _ in range(bootstrap):
            take = rng.integers(len(game_ids), size=len(game_ids))
            n = counts[take].sum()
            if n:
                means.append(sums[take].sum(axis=0)/n)
        if means:
            ci = dict(zip(METRICS, np.quantile(means, [.025, .975], axis=0).T.tolist()))
    return dict(overall=overall, overall_bootstrap_game_95=ci,
                buckets={bucket(lo): stats([(r, m) for r, m in valid if bucket(r['decisions_to_end']) == bucket(lo)])
                         for lo in BUCKETS}, excluded_roots=len(rows)-len(valid))


def make_env(args):
    # Evaluation schedules pairs; only the first --games results are reported.
    return VecEnv(min(args.num_envs, args.games), args.threads,
                  eval_pairs=(args.games+1)//2, opponent='external',
                  eval_seed=args.seed, track_projection=True)


def draw_targets(args):
    rng = np.random.default_rng(args.seed+3)
    return [set(map(int, rng.choice(getattr(args, 'target_range', 100), args.roots_per_game, replace=False)))
            for _ in range(args.games)]


@torch.inference_mode()
def real_pass(args, model, measure=None, *, targets=None, sample=True):
    """One online pass. Optional explicit targets/greedy policy support CPU QA."""
    env = make_env(args)
    actor = Actor(model, env, args.device, sample=sample)
    lengths = np.zeros(2*((args.games+1)//2), np.int64)
    targets = [set(ds) for ds in (draw_targets(args) if targets is None else targets)]
    selected = [[] for _ in range(args.games)]
    validation = Counter()
    digest = hashlib.sha256()
    frozen = {}  # physical row -> mutable output record, retained until release
    workspace = None
    last = perf_counter()
    with _policy_rng(args.seed, (actor,)):
        while not env.eval_done():
            starts = np.array([r for r in np.flatnonzero((env.stage >= 0) & (env.substep == 0))
                               if r not in frozen], np.int32)
            candidates = []
            for r in starts:
                g = int(env.episode_id[r])
                d = int(lengths[g])
                if g < args.games and d in targets[g]:
                    targets[g].remove(d)
                    candidates.append(r)
            rr = np.array(candidates, np.int32)
            if len(rr):
                support = env.determinize_support(rr)
                validation.update({'unsupported': int((support != 0).sum())})
                rr = rr[support == 0]
                valid = []
                for r in rr:
                    try:
                        env.unused_assignment(int(r), env.label_opp_hand[r].astype(np.int32))
                        valid.append(r)
                    except ValueError:
                        validation['invalid_assignment'] += 1
                rr = np.array(valid, np.int32)
                if len(rr):
                    if workspace is None or workspace.num_envs != len(rr):
                        workspace = SearchEnv(len(rr), args.threads)
                    status = workspace.load_worlds(env, rr, env.label_opp_hand[rr].astype(np.int32),
                                                   np.zeros(len(rr), np.uint64)).copy()
                    validation.update(map(int, status))
                    for r in rr[status == 0]:
                        g = int(env.episode_id[r])
                        selected[g].append(int(lengths[g]))
                        frozen[int(r)] = dict(game=g, decision_index=int(lengths[g]),
                                              actor_seat=int(env.actor_seat[r]))
            active = np.array([r for r in np.flatnonzero(env.stage >= 0) if r not in frozen], np.int32)
            released = {}
            if frozen and (len(frozen) >= args.root_batch or not len(active)):
                released, frozen = frozen, {}
                roots = np.array(list(released), np.int32)
                if measure is not None:
                    measure(env, roots, list(released.values()))
                active = np.flatnonzero(env.stage >= 0).astype(np.int32)
            result = actor.act(rows=active, compact_transfer=True)
            # Released roots are still at precisely the same decision. Take v
            # from their actual real-policy action, avoiding a spare RNG draw.
            for r, record in released.items():
                p = torch.from_numpy(result.value_logits[r]).softmax(-1).numpy()
                record['v'] = float(p[0]-p[1])
            choices = result.choices
            if frozen:
                choices[list(frozen)] = -1
            digest.update(choices.tobytes())
            starts = active[env.substep[active] == 0]
            np.add.at(lengths, env.episode_id[starts].astype(int), 1)
            env.step(choices)
            if perf_counter()-last >= 10:
                print(f'online: {len(env.eval_results())}/{args.games} games, '
                      f'{int(lengths.sum())} real decisions, {len(frozen)} frozen roots', flush=True)
                last = perf_counter()
    results = {int(r['g']): dict(r, external_decisions=int(lengths[r['g']]))
               for r in env.eval_results() if r['g'] < args.games}
    return selected, results, digest.hexdigest(), dict(validation)


class Continuations:
    def __init__(self, args, model, belief):
        self.args = args
        config = SearchConfig(worlds=args.k, max_inference_batch=args.inference_batch)
        self.agents = {
            'state': SearchAgent(model, OracleBelief(), args.device, config=config,
                                 seed=args.seed+1, num_threads=args.threads),
            'info': SearchAgent(model, belief, args.device, config=config,
                                seed=args.seed+2, num_threads=args.threads)}
        self.rows = []
        self.seconds = Counter()
        self.decisions = 0
        self.last = perf_counter()
        self.env = None
        self.batches = []
        self.rng = np.random.default_rng(args.seed+5)

    @torch.inference_mode()
    def __call__(self, src, roots, records):
        args = self.args
        start = perf_counter()
        parts = []
        # One isolated stream includes belief draws and all continuation actions.
        with _policy_rng(int(self.rng.integers(0, 2**31)), tuple(self.agents.values())):
            for variant, agent in self.agents.items():
                since = perf_counter()
                rr, assignments, seeds = agent._sample_worlds(src, roots)
                parts.append((variant, rr, assignments, seeds))
                self.seconds[variant] += perf_counter()-since
            since = perf_counter()
            rr = np.concatenate([p[1] for p in parts])
            counts = {(v, int(r)): Counter() for v in self.agents for r in roots}
            if len(rr):
                history = src.obs_schema()['K']
                if self.env is None or self.env.num_envs != len(rr):
                    self.env = SearchEnv(len(rr), args.threads, history_len=history,
                                         max_decisions=args.max_decisions)
                env = self.env
                status = env.load_worlds(src, rr, np.concatenate([p[2] for p in parts]),
                                         np.concatenate([p[3] for p in parts])).copy()
                actor = _SearchActor(next(iter(self.agents.values())).model, env, args.device,
                                     sample=True, max_batch=args.inference_batch,
                                     cache_device='auto', cache_mb=args.cache_mb)
                while np.any(env.stage >= 0):
                    env.step(actor.act(compact_transfer=True).choices)
                    if perf_counter()-self.last >= 10:
                        print(f'continuations: {len(roots)} roots, {len(rr)} worlds, '
                              f'{env.decisions.sum()} batch decisions', flush=True)
                        self.last = perf_counter()
                decisions = int(env.decisions.sum())
                self.decisions += decisions
                offset = 0
                for variant, rows, _, _ in parts:
                    agent = self.agents[variant]
                    ss = status[offset:offset+len(rows)]
                    agent.counts['invalid_loads'] += int((ss != 0).sum())
                    for value, n in zip(*np.unique(ss, return_counts=True)):
                        agent.world_load_counts[f'floor:status_{int(value)}'] += int(n)
                    for i, r in enumerate(rows, offset):
                        key = ('invalid_loads' if status[i] else 'truncations' if env.truncated[i]
                               else 'wins' if env.result[i] == 1 else 'losses' if env.result[i] == -1 else 'draws')
                        counts[variant, int(r)][key] += 1
                    offset += len(rows)
                self.batches.append(dict(roots=len(roots), worlds=len(rr), decisions=decisions,
                                         cache_device=str(actor.token_cache.device) if actor.token_cache is not None else None))
                # Release the token cache before potentially allocating a new env.
                del actor
            for variant in self.agents:
                for r, record in zip(roots, records):
                    c = counts[variant, int(r)]
                    completed = sum(c[k] for k in ('wins', 'losses', 'draws'))
                    record[variant] = dict({k: c[k] for k in ('wins', 'losses', 'draws', 'truncations', 'invalid_loads')},
                                           completed=completed, requested=args.k,
                                           exhausted_samples=args.k-sum(c.values()),
                                           p_hat=(c['wins']+.5*c['draws'])/completed if completed else None)
            self.seconds['rollout'] += perf_counter()-since
        self.rows.extend(records)
        print(f'measured {len(self.rows)} roots ({perf_counter()-start:.1f}s batch; '
              f'{self.seconds.total():.1f}s continuations)', flush=True)


def print_summary(summary):
    print('| decisions to end | n | value | state raw | state corrected | info raw | info corrected | mean abs v-state | reducible part (upper bound) |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for name, stats in [('overall', summary['overall']), *summary['buckets'].items()]:
        values = ['—' if stats[k] is None else f'{stats[k]:.6f}' for k in METRICS]
        print('| ' + ' | '.join([name, str(stats['n']), *values]) + ' |')
    print('Overall bootstrap by game 95% intervals:')
    for key, ci in summary['overall_bootstrap_game_95'].items():
        print(f'  {key}: {ci}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/play/m-2166.pt')
    parser.add_argument('--belief', default='runs/play/belief-ar-m.pt')
    for name, default in [('games', 400), ('roots-per-game', 1), ('k', 64), ('seed', 81000),
                          ('num-envs', None), ('threads', 8), ('max-decisions', 10000),
                          ('root-batch', 64), ('target-range', 100), ('inference-batch', None), ('cache-mb', None), ('bootstrap', 2000)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--out', type=Path, default=Path('measurements/value-floor.json'))
    args = parser.parse_args()
    args.num_envs = args.games if args.num_envs is None else args.num_envs
    args.inference_batch = (1024 if args.device == 'mps' else 128) if args.inference_batch is None else args.inference_batch
    args.cache_mb = (4096 if args.device == 'mps' else 256) if args.cache_mb is None else args.cache_mb
    if any(getattr(args, k) <= 0 for k in ('games', 'roots_per_game', 'num_envs', 'threads',
                                           'max_decisions', 'root_batch', 'inference_batch', 'bootstrap')) or args.k < 2 or args.seed < 0 or args.cache_mb < 0 or args.roots_per_game > 100:
        parser.error('positive budgets, 1<=roots-per-game<=100, k>=2, cache-mb>=0 and seed>=0 required')
    torch.set_num_threads(args.threads)
    start = perf_counter()
    model = load_policy_model(args.checkpoint, args.device)
    belief = load_search_belief(args.belief, device=args.device)
    ready = perf_counter()
    continuations = Continuations(args, model, belief)
    selected, games, digest, validation = real_pass(args, model, continuations)
    rows = sorted(continuations.rows, key=lambda r: (r['game'], r['decision_index']))
    if len(rows) != sum(map(len, selected)):
        raise RuntimeError('selected roots missing')
    for r in rows:
        game = games[r['game']]
        outcome = game['result'] if game['agent_seat'] == r['actor_seat'] else -game['result']
        r.update(y=(outcome+1)/2, real_truncated=bool(game['truncated']),
                 decisions_to_end=game['external_decisions']-r['decision_index'])
    summary = summarize(rows, args.bootstrap, args.seed+4)
    totals = {v: {key: sum(r[v][key] for r in rows)
                  for key in ('requested', 'completed', 'wins', 'losses', 'draws', 'truncations', 'invalid_loads', 'exhausted_samples')}
              for v in ('state', 'info')}
    elapsed = perf_counter()-start
    timing = dict(total_seconds=elapsed, load_seconds=ready-start,
                  enumeration_seconds=0.0, continuation_seconds=dict(continuations.seconds),
                  continuation_decisions=continuations.decisions,
                  completed_worlds_per_second=sum(t['completed'] for t in totals.values())/max(continuations.seconds.total(), 1e-12),
                  continuation_decisions_per_second=continuations.decisions/max(continuations.seconds.total(), 1e-12))
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  methodology=__doc__, real_trajectory_sha256=digest, replay_verified=False,
                  validation_status_counts=validation, eligible_roots_by_game=list(map(len, selected)),
                  root_selection=dict(targets_by_game=[sorted(ds) for ds in draw_targets(args)],
                                      selected_by_game=selected,
                                      missed_targets=args.games*args.roots_per_game-len(rows),
                                      decisions_from_start=dict(Counter(bucket(r['decision_index']) for r in rows)),
                                      decisions_to_end=dict(Counter(bucket(r['decisions_to_end']) for r in rows))),
                  continuation_batches=continuations.batches,
                  real_truncations=sum(g['truncated'] for g in games.values()),
                  continuation_totals=totals, timing=timing, summary=summary,
                  sampler_statistics={v: a.statistics() for v, a in continuations.agents.items()},
                  games=list(games.values()), roots=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Root distribution: ' + json.dumps({k: report['root_selection'][k] for k in
          ('missed_targets', 'decisions_from_start', 'decisions_to_end')}), flush=True)
    print_summary(summary)
    print(f'Totals: {totals}\nTiming: {timing}\nSaved {args.out}', flush=True)


if __name__ == '__main__':
    main()
