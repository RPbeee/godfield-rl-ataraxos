#!/usr/bin/env python3
"""Run all six ordered matchups of the baseline bots."""
import argparse
from itertools import permutations
from time import perf_counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gfrl.evaluation import evaluate_bots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=5000)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--eval-seed', type=int, default=0)
    args = parser.parse_args()
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
