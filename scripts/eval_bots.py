#!/usr/bin/env python3
"""Run all six ordered matchups of the baseline bots."""
import argparse
from itertools import permutations
from time import perf_counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import json
import torch
from gfrl import VecEnv
from gfrl.actor import Actor
from gfrl.model import Model, ModelConfig
from gfrl.evaluation import evaluate_bots, evaluate_policy
from gfrl.runtime import add_runtime_args, configure_precision, configure_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=5000)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--eval-seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path, help='evaluate checkpoint EMA against the bots')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--dtype', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--num-envs', type=int, default=256)
    add_runtime_args(parser)
    args = parser.parse_args()
    if args.checkpoint:
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.eval_seed)
        configure_precision(args.device, args.tf32)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model = Model(ModelConfig(**state['model_config'])).to(args.device).eval().requires_grad_(False)
        model.load_state_dict(state['ema'])
        configure_model(model, args)
        actor = Actor(model, VecEnv(args.num_envs, args.threads, args.eval_seed),
                      args.device, dtype=args.dtype)
        for opponent in ('random', 'proposal', 'greedy'):
            result = evaluate_policy(actor, opponent, args.pairs, args.num_envs,
                                     args.threads, args.eval_seed)
            print(json.dumps(dict(agent='ema', opponent=opponent, **result)), flush=True)
        return
    print('| Agent | Opponent | Win rate | Wilson 95% | Pair bootstrap 95% | Truncated | Mean decisions | Seconds | Games/s |', flush=True)
    print('|---|---|---:|---|---|---:|---:|---:|---:|', flush=True)
    started = perf_counter()
    for a, b in permutations(('random', 'proposal', 'greedy'), 2):
        start = perf_counter()
        s = evaluate_bots(a, b, args.pairs, args.threads, args.eval_seed)
        elapsed = perf_counter() - start
        interval = lambda v: f'{v[0]:.2%}–{v[1]:.2%}'
        rate = 'N/A' if s['win_rate'] is None else f"{s['win_rate']:.2%}"
        trunc = 'N/A' if s['truncation_rate'] is None else f"{s['truncation_rate']:.2%}"
        decisions = 'N/A' if s['mean_decisions'] is None else f"{s['mean_decisions']:.2f}"
        print(f"| {a} | {b} | {rate} | {interval(s['wilson_interval'])} | {interval(s['bootstrap_interval'])} | {trunc} | {decisions} | {elapsed:.2f} | {s['games']/elapsed:.1f} |", flush=True)
    print(f'Total: {perf_counter() - started:.2f}s', flush=True)


if __name__ == '__main__':
    main()
