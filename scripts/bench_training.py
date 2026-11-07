#!/usr/bin/env python3
"""CPU legacy/columnar comparison; informational, no speed threshold."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'tests/python')]
import numpy as np
import torch
from gfrl.model import Model, ModelConfig
from gfrl.rollout import Collector
from gfrl.learner import Learner, LearnerConfig
from legacy.rollout import Collector as OldCollector
from legacy.learner import Learner as OldLearner, LearnerConfig as OldConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=512)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--pool-fraction', type=float, default=.2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = Model(ModelConfig(d=32, depth=1, heads=4, ff=128))
    results = {}
    for name, collector_type, learner_type, cfg in (
            ('old', OldCollector, OldLearner, OldConfig()),
            ('new', Collector, Learner, LearnerConfig())):
        torch.manual_seed(1)
        np.random.seed(1)
        learner = learner_type(deepcopy(model), cfg)
        collector = collector_type(learner.model, num_envs=args.num_envs, threads=args.threads, seed=42, pool_fraction=args.pool_fraction)
        pool, history = [], []
        for i in range(1,args.iterations+1):
            begin = perf_counter()
            rollout = collector.collect(args.games, pool)
            collected = perf_counter()
            metrics = learner.update(rollout.decisions, i)
            ended = perf_counter()
            pool = [learner.pool_entry(i)]
            row = dict(implementation=name, iteration=i, sampled=rollout.sampled_decisions,
                       decisions=len(rollout.decisions), collect_s=collected-begin, learn_s=ended-collected,
                       collect_per_s=rollout.sampled_decisions/(collected-begin),
                       learn_per_s=len(rollout.decisions)/(ended-collected))
            if hasattr(rollout, 'timings'):
                row['collection'] = rollout.timings
                row['learning'] = {k:v for k,v in metrics.items() if k.endswith('_s')}
            history.append(row)
            print(json.dumps(row), flush=True)
        results[name] = history
    for a,b in zip(results['old'], results['new']):
        print(json.dumps(dict(iteration=a['iteration'], collect_speedup=b['collect_per_s']/a['collect_per_s'],
                              learn_speedup=b['learn_per_s']/a['learn_per_s'])), flush=True)


if __name__ == '__main__':
    main()
