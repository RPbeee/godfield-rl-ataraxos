#!/usr/bin/env python3
"""Measure the 94+232 legal-mask exception on unchanged CPU self-play games."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl import _core, VecEnv
from gfrl.actor import Actor
from gfrl.evaluation import _policy_rng, load_policy_model, summarize


def measure(args):
    torch.set_num_threads(args.threads)
    env = VecEnv(args.num_envs, args.threads, eval_pairs=args.games//2,
                 opponent='external', eval_seed=args.seed)
    actor = Actor(load_policy_model(args.checkpoint, 'cpu'), env, 'cpu')
    counts = dict(decisions=0, holding_pair=0, testable=0, mask_changed=0,
                  selected_pair=0, selected_sensitive=0, holding_with_opponent_209=0)
    # Per row state is reset at each fresh external decision, covering both seats.
    sensitive = np.zeros(args.num_envs, bool)
    start = perf_counter()
    last_report = start
    with _policy_rng(args.seed, (actor,)):
        while not env.eval_done():
            rows = np.flatnonzero(env.stage >= 0)
            for r in rows[env.substep[rows] == 0]:
                counts['decisions'] += 1
                holding, testable, changed, opp209 = _core._measure_danger_probe(env, int(r))
                sensitive[r] = changed
                counts['holding_pair'] += holding
                counts['testable'] += testable
                counts['mask_changed'] += changed
                counts['holding_with_opponent_209'] += holding and opp209
            choices = actor(env, rows)
            for r in rows:
                if _core._measure_danger_selected(env, int(r), int(choices[r])):
                    counts['selected_pair'] += 1
                    counts['selected_sensitive'] += int(sensitive[r])
            env.step(choices)
            if perf_counter()-last_report >= 30:
                print(f'{len(env.eval_results())}/{args.games} games; {counts}', flush=True)
                last_report = perf_counter()
    ratio = lambda a, b: counts[a]/counts[b] if counts[b] else None
    return dict(config=vars(args) | {'out': str(args.out)}, counts=counts,
                mask_changed_per_holding=ratio('mask_changed', 'holding_pair'),
                mask_changed_per_testable=ratio('mask_changed', 'testable'),
                selected_per_holding=ratio('selected_pair', 'holding_pair'),
                selected_per_decision=ratio('selected_pair', 'decisions'),
                elapsed_seconds=perf_counter()-start, games=summarize(env.eval_results()),
                definition='Both seats; fresh external decisions only. Holding = unused displayed 94 and 232. '
                'Counterfactuals preserve hand count, tokens, used cards, public faces and exact MatchView. '
                'Only main roots are probed. Unchangeable public 209, empty hands or incompatible Dream '
                'faces are untestable. Selected = completed play command containing exactly displayed 94+232. '
                'Mask comparisons and command probes copy state and never alter the live match or RNG.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--games', type=int, default=1000)
    p.add_argument('--num-envs', type=int, default=32)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--out', type=Path, default=Path('measurements/mask-leak.json'))
    args = p.parse_args()
    if args.games <= 0 or args.games % 2 or min(args.num_envs, args.threads) <= 0:
        p.error('games must be positive and even; num-envs and threads must be positive')
    report = measure(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
