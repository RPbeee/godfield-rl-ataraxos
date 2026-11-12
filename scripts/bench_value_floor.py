#!/usr/bin/env python3
"""Small CPU continuation throughput pilot; fixed initial roots, both variants.

Uses the first N games at external decision 0 (N=8/32/64), with K=32 per
variant, one SearchEnv per pilot. Excludes real-game execution and model loading.
This controlled early-root workload is not the online [0,100) root distribution.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from value_floor import Continuations, make_env
import numpy as np
import torch
from gfrl.evaluation import load_policy_model
from gfrl.search import load_search_belief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/play/m-2166.pt')
    parser.add_argument('--belief', default='runs/play/belief-ar-m.pt')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--inference-batch', type=int, default=128)
    parser.add_argument('--out', type=Path, default=Path('measurements/value-floor-cpu-root-batches.json'))
    args = parser.parse_args()
    args.device, args.k, args.seed = 'cpu', 32, 81000
    args.cache_mb, args.max_decisions = 256, 10000
    torch.set_num_threads(args.threads)
    model = load_policy_model(args.checkpoint, 'cpu')
    belief = load_search_belief(args.belief, device='cpu')
    results = []
    for n in (8, 32, 64):
        args.games = args.num_envs = args.root_batch = n
        src = make_env(args)
        roots = np.arange(n, dtype=np.int32)
        assert np.all(src.substep == 0) and np.all(src.determinize_support(roots) == 0)
        mc = Continuations(args, model, belief)
        start = perf_counter()
        mc(src, roots, [dict(game=i, decision_index=0) for i in range(n)])
        elapsed = perf_counter()-start
        totals = {v: {key: sum(r[v][key] for r in mc.rows) for key in
                     ('requested', 'completed', 'wins', 'losses', 'draws', 'truncations',
                      'invalid_loads', 'exhausted_samples')} for v in ('state', 'info')}
        result = dict(root_batch=n, k=32, seconds=elapsed, continuation_decisions=mc.decisions,
                      continuation_decisions_per_second=mc.decisions/elapsed,
                      totals=totals, batches=mc.batches)
        results.append(result)
        print(json.dumps(result), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(methodology=__doc__, threads=args.threads,
                                           inference_batch=args.inference_batch,
                                           checkpoint=args.checkpoint, belief=args.belief,
                                           results=results), indent=2)+'\n')
        del mc, src


if __name__ == '__main__':
    main()
