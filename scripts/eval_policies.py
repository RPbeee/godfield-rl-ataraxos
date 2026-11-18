#!/usr/bin/env python3
"""Evaluate checkpoint EMAs or pool snapshots against one another on paired deals."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

import torch
from gfrl import VecEnv
from gfrl.actor import Actor
from gfrl.evaluation import evaluate_policies, list_pool_iterations, load_policy_model, restriction_categories
from gfrl.runtime import add_runtime_args, configure_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', help='checkpoint.pt or checkpoint.pt@pool:N')
    parser.add_argument('--b', nargs='+', help='one or more opponent checkpoint specifications')
    parser.add_argument('--pairs', type=int, default=500)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--eval-seed', type=int, default=0)
    parser.add_argument('--a-restrict', default='none',
                        help='A only: none, official, or comma list of kind,dream,self_target')
    parser.add_argument('--pool-epsilon', type=float, default=0.,
                        help='stage-wise epsilon-uniform legal noise for B only (same as training pool)')
    parser.add_argument('--out', type=Path, default=Path('eval-policies.json'))
    parser.add_argument('--list', nargs='*', metavar='CHECKPOINT',
                        help='list pool iterations for these paths, or for --a / --b')
    add_runtime_args(parser)
    args = parser.parse_args()
    if args.list is not None:
        paths = args.list or ([args.a] if args.a else []) + (args.b or [])
        if not paths:
            parser.error('--list requires checkpoint paths or --a / --b')
        for path in dict.fromkeys(paths):
            print(json.dumps(dict(checkpoint=path, iterations=list_pool_iterations(path))))
        return
    if not args.a or not args.b:
        parser.error('--a and --b are required for evaluation')
    if args.pairs <= 0 or args.num_envs <= 0 or args.threads <= 0:
        parser.error('--pairs, --num-envs and --threads must be positive')
    if not 0 <= args.pool_epsilon <= 1:
        parser.error('--pool-epsilon must be in [0,1]')
    if args.compile != 'off':
        parser.error('policy evaluation does not use compile; use --compile off')
    try:
        active_categories = restriction_categories(args.a_restrict)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    configure_precision(args.device, args.tf32)
    # Actors bind to the evaluator's environment on first use.
    idle_env = VecEnv(1, 1, eval_pairs=0)
    actor_a = Actor(load_policy_model(args.a, args.device), idle_env,
                    args.device, sample=True)
    results = []
    print('| B | Games | A win rate | Wilson 95% | Pair bootstrap 95% | First | Second | Draws | Mean decisions | A win decisions | A loss decisions | Truncations |', flush=True)
    print('|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|', flush=True)
    for opponent in args.b:
        actor_b = Actor(load_policy_model(opponent, args.device), idle_env,
                        args.device, sample=True, epsilon=args.pool_epsilon)
        summary = evaluate_policies(actor_a, actor_b, args.pairs, args.num_envs,
                                    args.threads, args.eval_seed, a_restrict=args.a_restrict)
        results.append(dict(a=args.a, b=opponent, **summary))
        interval = lambda v: f'{v[0]:.2%}–{v[1]:.2%}'
        length = lambda key: f'{summary[key]:.2f}' if summary[key] is not None else '—'
        print(f"| {opponent} | {summary['games']} | {summary['win_rate']:.2%} | "
              f"{interval(summary['wilson_interval'])} | {interval(summary['bootstrap_interval'])} | "
              f"{summary['first_win_rate']:.2%} | {summary['second_win_rate']:.2%} | "
              f"{summary['draws']} | {length('mean_game_length')} | "
              f"{length('mean_win_decisions')} | {length('mean_loss_decisions')} | "
              f"{summary['truncations']} |", flush=True)
        del actor_b
    args.out.parent.mkdir(parents=True, exist_ok=True)
    restriction_metadata = (dict(a_restrict=args.a_restrict, a_restrict_categories=list(active_categories))
                            if active_categories else {})
    args.out.write_text(json.dumps(dict(pairs=args.pairs, eval_seed=args.eval_seed,
                                      num_envs=args.num_envs, threads=args.threads,
                                      device=args.device, tf32=args.tf32,
                                      pool_epsilon=args.pool_epsilon,
                                      **restriction_metadata,
                                      results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
