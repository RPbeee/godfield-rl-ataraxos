#!/usr/bin/env python3
"""Synchronized PPO benchmark: compilation/warmup and timed updates are separate."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl.actor import synchronize
from gfrl.model import Model, ModelConfig
from gfrl.rollout import Collector
from gfrl.learner import Learner, LearnerConfig
from gfrl.runtime import (add_runtime_args, configure_precision, configure_model,
                          compiler_metrics, metric_delta, warmup_encoder)
from gfrl.benchmark import sweep_cases, print_table


def benchmark(args):
    configure_precision(args.device, args.tf32)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(args.threads)
    config = (ModelConfig.preset(args.size) if args.size else
              ModelConfig(d=32, depth=1, heads=4, ff=128))
    learner = Learner(Model(config), LearnerConfig(minibatch_size=args.minibatch_size,
                      microbatch_size=args.microbatch_size), args.device, args.dtype)
    # Collect the fixed benchmark dataset eagerly. No compile time is hidden
    # inside data generation, and the same seed is used for each sweep case.
    collector = Collector(learner.model, device=args.device, dtype=args.dtype,
                          num_envs=args.num_envs, threads=args.threads, seed=42,
                          pool_fraction=args.pool_fraction)
    rollout = collector.collect(args.games)
    configure_model(learner.model, args)
    compile_before = compiler_metrics()
    begin = perf_counter()
    learner.model.eval()
    warmup_encoder(learner.model, args.device, training=True, dtype=args.dtype,
                   all_buckets=args.warmup_buckets == 'all')
    for i in range(args.warmup):
        learner.update(rollout.decisions, i+1)
    synchronize(learner.device)
    warmup_s = perf_counter()-begin
    warmup_metrics = metric_delta(compile_before)
    compile_before = compiler_metrics()
    begin = perf_counter()
    history = []
    for i in range(args.iterations):
        history.append(learner.update(rollout.decisions, args.warmup+i+1))
    synchronize(learner.device)
    elapsed = perf_counter()-begin
    decisions = len(rollout.decisions)*args.iterations
    return dict(size=args.size or 'tiny-d32-depth1', device=args.device, dtype=args.dtype,
                tf32=args.tf32, compile=args.compile,
                compile_mode_actor=('default' if args.device == 'cpu' else args.compile_mode_actor),
                compile_mode_learner=('default' if args.device == 'cpu' else args.compile_mode_learner),
                warmup_buckets=args.warmup_buckets, warmup_s=warmup_s,
                warmup_compiler=warmup_metrics, measured_compiler=metric_delta(compile_before),
                elapsed_s=elapsed, decisions=decisions, decisions_per_s=decisions/elapsed,
                learning=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=512)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--pool-fraction', type=float, default=.2)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--dtype', choices=('fp32', 'bf16', 'fp16'), default='fp32')
    parser.add_argument('--size', choices=('S', 'M', 'L'), help='default: tiny CPU benchmark model')
    parser.add_argument('--minibatch-size', type=int, default=4096)
    parser.add_argument('--microbatch-size', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--warmup-buckets', choices=('all', 'observed'), default='all')
    parser.add_argument('--sweep', action='store_true')
    add_runtime_args(parser)
    args = parser.parse_args()
    if min(args.games, args.num_envs, args.threads, args.iterations, args.warmup,
           args.minibatch_size) <= 0 or args.microbatch_size < 0:
        parser.error('counts must be positive; microbatch-size may be zero')
    if not 0 <= args.pool_fraction <= 1:
        parser.error('pool-fraction must be in [0,1]')
    results = []
    for case in sweep_cases(args):
        result = benchmark(case)
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)
    if args.sweep:
        print_table(results, 'decisions_per_s')


if __name__ == '__main__':
    main()
