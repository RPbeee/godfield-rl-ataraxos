"""Random-policy projection benchmark, including a separately built HEAD module.

.venv/bin/python tests/python/bench_projection.py --baseline-core /tmp/head/python/gfrl/_core.cpython-312-darwin.so

Each mode runs in a separate process. Timings include random action sampling and
step(), exclude creation/warmup, and count env.stats()['decisions'] (including
automatic decisions). The baseline module needs no track_projection argument.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


def measure(args):
    spec = importlib.util.spec_from_file_location('_core', args.core)
    core = importlib.util.module_from_spec(spec)
    sys.modules['_core'] = core
    spec.loader.exec_module(core)
    kwargs = {} if args.mode == 'baseline' else dict(track_projection=args.mode == 'true')
    env = core.VecEnv(args.num_envs, args.threads, 42, **kwargs)
    rng = np.random.default_rng(123)
    mask = env.mask

    def step():
        rank = (rng.random(len(mask)) * mask.sum(axis=1)).astype(np.int32)
        choices = (mask.cumsum(axis=1) > rank[:, None]).argmax(axis=1).astype(np.int32)
        env.step(choices)

    for _ in range(args.warmup):
        step()
    before = env.stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    seconds = time.perf_counter() - start
    counts = {k: v - before[k] for k, v in env.stats().items()}
    print(json.dumps(dict(mode=args.mode, seconds=seconds,
                         decisions_per_second=counts['decisions']/seconds, counts=counts)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-core', type=Path)
    p.add_argument('--core', type=Path)
    p.add_argument('--mode', choices=['baseline', 'false', 'true'])
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    if min(args.num_envs, args.threads, args.steps, args.repeats) <= 0 or args.warmup < 0:
        p.error('counts must be positive and warmup nonnegative')
    if args.mode:
        measure(args)
        return
    core = args.core or next((Path(__file__).resolve().parents[2]/'python/gfrl').glob('_core*.so'))
    modes = dict(false=core, true=core)
    if args.baseline_core:
        modes = dict(baseline=args.baseline_core, **modes)
    results = {mode: [] for mode in modes}
    counts = None
    # Rotate mode order to reduce systematic warmup/thermal ordering bias.
    for repeat in range(args.repeats):
        order = list(modes)
        order = order[repeat % len(order):] + order[:repeat % len(order)]
        for mode in order:
            cmd = [sys.executable, __file__, '--core', str(modes[mode]), '--mode', mode,
                   '--num-envs', str(args.num_envs), '--threads', str(args.threads),
                   '--warmup', str(args.warmup), '--steps', str(args.steps)]
            result = json.loads(subprocess.check_output(cmd, text=True))
            if counts is None:
                counts = result['counts']
            assert counts == result['counts'], 'trajectory counters differ'
            results[mode].append(result['decisions_per_second'])
            print(json.dumps(dict(repeat=repeat, **result)), flush=True)
    print(json.dumps(dict(num_envs=args.num_envs, threads=args.threads, steps=args.steps,
                          warmup=args.warmup, repeats=args.repeats,
                          median_decisions_per_second={k: statistics.median(v) for k, v in results.items()}), indent=2))


if __name__ == '__main__':
    main()
