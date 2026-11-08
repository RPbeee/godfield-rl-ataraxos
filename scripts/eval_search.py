#!/usr/bin/env python3
"""Paired-seed search versus policy or Greedy, with CPU/GPU timing statistics."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import torch
from gfrl import VecEnv
from gfrl.actor import Actor
from gfrl.evaluation import evaluate_policies, load_policy_model, _policy_rng, summarize
from gfrl.runtime import configure_precision
from gfrl.search import SearchAgent, SearchConfig, load_search_belief


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--opponent', choices=('policy', 'greedy'), default='policy')
    p.add_argument('--opponent-checkpoint')
    p.add_argument('--belief', required=True, help='checkpoint path, b1, or uniform')
    p.add_argument('--fit', help='B1 fitting data (optional count prior for uniform)')
    for name, default in (('worlds', 16), ('candidates', 16), ('candidate-samples', 64), ('depth', 10),
                          ('pairs', 500), ('num-envs', 32), ('threads', 1), ('seed', 0)):
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--leaf-view', choices=('actor', 'searcher'), default='actor')
    p.add_argument('--max-inference-batch', type=int, default=0,
                   help='search inference row limit (0: unsplit; start with 128 on a 10 GB GPU)')
    p.add_argument('--alpha', type=float, default=.002)
    p.add_argument('--beta', type=float, default=.02)
    p.add_argument('--select', choices=('sample', 'argmax'), default='sample')
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    p.add_argument('--profile', action='store_true', help='synchronized Actor phase timings (adds overhead)')
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--out', type=Path, default=Path('eval-search.json'))
    args = p.parse_args()
    if min(args.pairs, args.num_envs, args.threads) <= 0:
        p.error('pairs, num-envs, threads must be positive')
    try:
        config = SearchConfig(**{k: getattr(args, k) for k in ('worlds', 'candidates', 'candidate_samples', 'depth', 'alpha', 'beta', 'select', 'leaf_view', 'max_inference_batch')})
        belief = load_search_belief(args.belief, args.fit, args.device)
    except ValueError as exc:
        p.error(str(exc))
    torch.set_num_threads(args.threads)
    configure_precision(args.device, args.tf32)
    agent = SearchAgent(load_policy_model(args.checkpoint, args.device), belief, args.device,
                        config=config, seed=args.seed, num_threads=args.threads, profile=args.profile)
    start = perf_counter()
    if args.opponent == 'policy':
        opponent = Actor(load_policy_model(args.opponent_checkpoint or args.checkpoint, args.device),
                         VecEnv(1, 1, eval_pairs=0), args.device)
        summary = evaluate_policies(agent, opponent, args.pairs, args.num_envs, args.threads,
                                    args.seed, track_projection=True)
    else:
        with _policy_rng(args.seed, (agent,)):
            env = VecEnv(args.num_envs, args.threads, eval_pairs=args.pairs, opponent='greedy',
                         eval_seed=args.seed, track_projection=True)
            import numpy as np
            while not env.eval_done():
                rows = np.flatnonzero(env.stage >= 0)
                env.step(agent(env, rows))
            summary = summarize(env.eval_results())
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  **summary, search=agent.statistics(), elapsed_seconds=perf_counter()-start)
    interval = lambda v: f'{v[0]:.2%}–{v[1]:.2%}'
    md = ('| Opponent | Games | Win rate (draw=½) | Wilson 95% | Pair bootstrap 95% |\n'
          '|---|---:|---:|---|---|\n'
          f"| {args.opponent} | {summary['games']} | {summary['win_rate']:.2%} | "
          f"{interval(summary['wilson_interval'])} | {interval(summary['bootstrap_interval'])} |\n\n"
          '| Search phase | Seconds / searched decision |\n|---|---:|\n' +
          ''.join(f'| {k} | {v:.6f} |\n' for k, v in report['search']['seconds_per_search'].items()) +
          '\n```json\n'+json.dumps(report['search'], indent=2)+'\n```\n')
    print(md, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    args.out.with_suffix('.md').write_text(md)


if __name__ == '__main__':
    main()
