#!/usr/bin/env python3
"""Repeat information-set search at fixed policy-played evaluation roots.

Candidates AND their priors are frozen after repeat 0 within each setting.
Correlations pool matching root/candidate pairs over all repeat pairs; centered
correlation subtracts each root/repeat's candidate mean. Missing candidates are
excluded from correlations and reported; TV uses the union (missing mass = 0).
top_q_se_gap_ratio uses the best-q candidate SE / top-two q gap.
se_gap_ratio uses sqrt(SE1**2 + SE2**2) / gap, ignoring common-world covariance.
Policy agreement uses the first full pi_theta draw before candidate filtering,
as in eval_search.py. seconds_per_root averages R searches (candidate generation
only in repeat 0), excluding root collection. Null means undefined, not zero.
"""
import argparse
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl import VecEnv
from gfrl.actor import Actor
from gfrl.evaluation import _policy_rng, load_policy_model
from gfrl.runtime import configure_precision
from gfrl.search import SearchAgent, SearchConfig, load_search_belief


def mean(xs):
    return float(np.mean(xs)) if xs else None


def correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def metrics(roots):
    raw_x, raw_y, center_x, center_y = [], [], [], []
    agree, q_agree, sample_agree, tv = [], [], [], []
    ses, gaps, ratios, top_ratios, policy, maxima = [], [], [], [], [], []
    missing = failed = zero_gaps = multiple = 0
    for repeats in roots:
        for report in repeats:
            if report is None:
                failed += 1
                continue
            cs = report['candidates']
            multiple += len(cs) > 1
            ses.extend(c['q_se'] for c in cs if c['q_se'] is not None)
            policy.append(report['selected'] == report['policy_path'])
            maxima.append(max(report['probabilities']))
            top = sorted(cs, key=lambda c: -c['q'])[:2]
            if len(top) == 2:
                gap = top[0]['q']-top[1]['q']
                gaps.append(gap)
                if gap == 0:
                    zero_gaps += 1
                elif all(c['q_se'] is not None for c in top):
                    ratios.append(float(np.hypot(*(c['q_se'] for c in top))/gap))
                    top_ratios.append(top[0]['q_se']/gap)
        for a, b in combinations(repeats, 2):
            if a is None or b is None:
                continue
            ca = {tuple(c['path']): c for c in a['candidates']}
            cb = {tuple(c['path']): c for c in b['candidates']}
            shared = [p for p in ca if p in cb]
            missing += len(ca.keys() ^ cb.keys())
            if shared:
                x = np.array([ca[p]['q'] for p in shared])
                y = np.array([cb[p]['q'] for p in shared])
                raw_x.extend(x); raw_y.extend(y)
                center_x.extend(x-x.mean()); center_y.extend(y-y.mean())
            pa = dict(zip(ca, a['probabilities']))
            pb = dict(zip(cb, b['probabilities']))
            agree.append(max(pa, key=pa.get) == max(pb, key=pb.get))
            q_agree.append(max(ca, key=lambda p: ca[p]['q']) == max(cb, key=lambda p: cb[p]['q']))
            sample_agree.append(a['selected'] == b['selected'])
            tv.append(.5*sum(abs(pa.get(p, 0)-pb.get(p, 0)) for p in pa.keys() | pb.keys()))
    return dict(q_correlation=correlation(raw_x, raw_y), centered_q_correlation=correlation(center_x, center_y),
                argmax_agreement=mean(agree), q_argmax_agreement=mean(q_agree), sample_agreement=mean(sample_agree),
                total_variation=mean(tv), q_se=mean(ses), top_two_gap=mean(gaps), se_gap_ratio=mean(ratios),
                top_q_se_gap_ratio=mean(top_ratios), zero_gaps=zero_gaps, se_gap_count=len(ratios), se_gap_median=float(np.median(ratios)) if ratios else None,
                multi_candidate_searches=multiple, policy_agreement=mean(policy), max_probability=mean(maxima),
                repeat_pairs=len(tv), matched_q_pairs=len(raw_x), missing_candidate_pairs=missing, failed_searches=failed)


def markdown(results):
    fields = ['worlds', 'depth', 'beta', 'q_correlation', 'centered_q_correlation', 'argmax_agreement',
              'sample_agreement', 'total_variation', 'q_se', 'top_two_gap', 'top_q_se_gap_ratio', 'se_gap_ratio',
              'policy_agreement', 'max_probability', 'seconds_per_root']
    lines = ['| '+' | '.join(fields)+' |', '| '+' | '.join(['---:']*len(fields))+' |']
    for result in results:
        row = {**result['config'], **result['metrics']}
        lines.append('| '+' | '.join('—' if row[k] is None else f'{row[k]:.4f}' if isinstance(row[k], float) else str(row[k]) for k in fields)+' |')
    return '\n'.join(lines)+'\n'


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--belief', required=True,
                   help='checkpoint path, b1, uniform, or oracle (evaluation-only diagnostic; cheats using true opponent hands)')
    p.add_argument('--fit')
    p.add_argument('--worlds', type=int, nargs='+', default=[16])
    p.add_argument('--depth', type=int, nargs='+', default=[10])
    p.add_argument('--beta', type=float, nargs='+', default=[.02])
    p.add_argument('--leaf-view', choices=('actor', 'searcher'), default='actor')
    p.add_argument('--max-inference-batch', type=int, default=0,
                   help='search inference row limit (0: unsplit; start with 128 on a 10 GB GPU)')
    p.add_argument('--alpha', type=float, default=.002)
    for name, default in [('roots', 200), ('repeats', 2), ('candidates', 16), ('candidate-samples', 64),
                          ('num-envs', 32), ('threads', 1), ('seed', 0), ('warmup', 20)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--profile', action='store_true', help='synchronized Actor phase timings (adds overhead)')
    p.add_argument('--save-world-values', action='store_true')
    p.add_argument('--out', type=Path, default=Path('search-noise.json'))
    args = p.parse_args()
    if min(args.roots, args.num_envs, args.threads) <= 0 or args.repeats < 2 or min(args.seed, args.warmup) < 0:
        p.error('positive roots/num-envs/threads, repeats >= 2, nonnegative seed/warmup required')
    try:
        configs = [SearchConfig(worlds=w, depth=d, beta=b, alpha=args.alpha, candidates=args.candidates,
                                candidate_samples=args.candidate_samples, leaf_view=args.leaf_view,
                                max_inference_batch=args.max_inference_batch) for w, d, b in product(args.worlds, args.depth, args.beta)]
    except ValueError as exc:
        p.error(str(exc))
    torch.set_num_threads(args.threads)
    configure_precision(args.device, args.tf32)
    model = load_policy_model(args.checkpoint, args.device)
    belief = load_search_belief(args.belief, args.fit, args.device)
    agents = [SearchAgent(model, belief, args.device, config=c, seed=args.seed, num_threads=args.threads,
                          profile=args.profile) for c in configs]
    data = [[] for _ in configs]
    elapsed = np.zeros(len(configs))
    env = VecEnv(args.num_envs, args.threads, eval_pairs=max(args.roots*100, args.num_envs),
                 opponent='external', eval_seed=args.seed, track_projection=True)
    actor = Actor(model, env, args.device)
    collected, stages = 0, 0
    locations = []
    start = perf_counter()
    with _policy_rng(args.seed, (actor,)):
        while collected < args.roots:
            if env.eval_done():
                raise RuntimeError('evaluation exhausted before enough supported roots')
            active = np.flatnonzero(env.stage >= 0)
            fresh = active[env.substep[active] == 0].astype(np.int32)
            eligible = fresh[env.determinize_support(fresh) == 0] if stages >= args.warmup else fresh[:0]
            rows = eligible[:args.roots-collected]
            if len(rows):
                locations.extend(dict(root=collected+i, game=int(env.episode_id[r]), row=int(r), stage_step=stages)
                                 for i, r in enumerate(rows))
                for ci, agent in enumerate(agents):
                    repeats = [[] for _ in rows]
                    fixed, policy_paths = None, None
                    since = agent._time()
                    for rep in range(args.repeats):
                        seed = int(np.random.SeedSequence([args.seed, collected, rep]).generate_state(1)[0])
                        agent.rng = np.random.default_rng(seed)
                        with _policy_rng(seed, (agent,)):
                            agent.search(env, rows, fixed_candidates=fixed, policy_paths=policy_paths, report_values=True)
                        if rep == 0:
                            fixed, policy_paths = agent.last_candidates, agent.last_policy_paths
                        reports = {r['row']: r for r in agent.last_search}
                        for i, row in enumerate(rows):
                            report = reports.get(int(row))
                            if report and not args.save_world_values:
                                for c in report['candidates']:
                                    c.pop('world_values')
                            repeats[i].append(report)
                    elapsed[ci] += agent._time()-since
                    data[ci].extend(repeats)
                    # Bound retained SearchEnv memory across the entire grid.
                    agent._workspaces.clear()
                collected += len(rows)
                print(f'Collected {collected}/{args.roots} roots', file=sys.stderr, flush=True)
            env.step(actor.act(active).choices)
            stages += 1
    results = []
    for ci, (config, agent) in enumerate(zip(configs, agents)):
        result = metrics(data[ci])
        result['seconds_per_root'] = float(elapsed[ci]/args.roots/args.repeats)
        multi = [rs for rs in data[ci] if any(r and len(r['candidates']) > 1 for r in rs)]
        results.append(dict(config=asdict(config), metrics=result, multi_candidate_roots=len(multi),
                            multi_candidate_metrics=metrics(multi), seconds=float(elapsed[ci]),
                            phase_seconds=dict(agent.seconds), counts=dict(agent.counts), roots=data[ci]))
    report = dict(belief=args.belief, diagnostic_only=args.belief == 'oracle',
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  definitions=__doc__, root_locations=locations, results=results, elapsed_seconds=perf_counter()-start)
    md = markdown(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    args.out.with_suffix('.md').write_text(md)
    print(md)


if __name__ == '__main__':
    main()
