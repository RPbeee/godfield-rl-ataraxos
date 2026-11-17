#!/usr/bin/env python3
"""CPU single-root search latency for K=16, depth=1/3 and R=1/4/8."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl import VecEnv
from gfrl.evaluation import _policy_rng, load_policy_model
from gfrl.search import SearchAgent, SearchConfig, load_search_belief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/play/m-lr1e5-4500.pt')
    parser.add_argument('--belief', default='runs/belief-m2166/train/checkpoint.pt')
    parser.add_argument('--fit', default='runs/belief-m2166/fit.pt')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-inference-batch', type=int, default=128)
    parser.add_argument('--seed', type=int, default=741)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    torch.set_num_threads(1)
    model = load_policy_model(args.checkpoint, 'cpu')
    belief = load_search_belief(args.belief, args.fit, 'cpu')
    rows = np.array([0], np.int32)
    roots = [VecEnv(1, 1, args.seed+i, track_projection=True) for i in range(args.repeats)]
    results = []
    for depth in (3, 1):
        for rollouts in (1, 4, 8):
            config = SearchConfig(worlds=16, candidates=16, candidate_samples=64, depth=depth,
                                  rollouts_per_world=rollouts, select='argmax',
                                  max_inference_batch=args.max_inference_batch)
            agent = SearchAgent(model, belief, config=config, seed=args.seed)
            times, candidate_counts = [], []
            for repeat in range(-1, args.repeats):
                # One warmup per setting. Timed trials use identical public
                # roots, belief/candidate RNG and policy RNG across settings.
                index = max(repeat, 0)
                agent.rng = np.random.default_rng(args.seed+index)
                with _policy_rng(args.seed+index, (agent,)):
                    start = perf_counter()
                    selected = agent.search(roots[index], rows)
                    elapsed = perf_counter()-start
                assert selected and agent.last_search
                if repeat >= 0:
                    times.append(elapsed)
                    candidate_counts.append(len(agent.last_search[0]['candidates']))
                print(f'depth={depth} R={rollouts} trial={repeat} seconds={elapsed:.6f}', flush=True)
            results.append(dict(depth=depth, rollouts_per_world=rollouts, config=asdict(config),
                                seconds=times, median_seconds=float(np.median(times)),
                                candidates=candidate_counts))
            report = dict(checkpoint=args.checkpoint, belief=args.belief, fit=args.fit,
                          model_config=asdict(model.config), machine=platform.machine(),
                          platform=platform.platform(), torch_version=torch.__version__, threads=1,
                          seed=args.seed, repeats=args.repeats,
                          timing='single supported initial root per call; search() including belief, candidates, '
                                 'rollout, leaf, selection; excludes loading and root real-policy inference; '
                                 'one warmup per setting; CPU only; median over paired public roots',
                          results=results)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
