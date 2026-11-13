#!/usr/bin/env python3
"""Strict CPU/CUDA differential replay. No tolerances and no CPU fallback."""
import argparse
import json
from time import perf_counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
import torch

from gfrl import VecEnv
from gfrl.gpu import FIELDS, GpuEnv


def validate(n=16, steps=1000, seed=42, history_len=64, canonical=True):
    cpu = VecEnv(n, num_threads=min(n, 8), base_seed=seed,
                 history_len=history_len, canonical=canonical)
    rng = np.random.default_rng(seed)
    compared = 0
    start = perf_counter()
    with GpuEnv(n, base_seed=seed, history_len=history_len, canonical=canonical) as gpu:
        for step in range(steps+1):
            for name in FIELDS:
                expected = np.asarray(getattr(cpu, name))
                actual = getattr(gpu, name).cpu().numpy()
                if not np.array_equal(expected, actual):
                    where = np.argwhere(expected != actual)
                    index = tuple(where[0])
                    raise AssertionError(f'seed={seed} step={step} {name}{index}: '
                                         f'CPU={expected[index]!r} GPU={actual[index]!r}; '
                                         f'{len(where)} differing elements')
            compared += n
            if step == steps:
                break
            # One common sequence of explicit policy-stage choices on both engines.
            mask = np.asarray(cpu.mask)
            choices = np.array([rng.choice(np.flatnonzero(row)) for row in mask], dtype=np.int32)
            if step % 29 == 0:
                # Freeze is part of the VecEnv API, including flags and history.
                choices[np.asarray(cpu.substep) == 0] = -1
            cpu.step(choices)
            gpu.step(torch.from_numpy(choices).to(gpu.device))
        stats = cpu.stats()
        if stats != gpu.stats():
            raise AssertionError(f"statistics differ: CPU={stats}, GPU={gpu.stats()}")
    return dict(status='passed', compared_states=compared,comparison_unit='policy_stage_state', steps=steps, num_envs=n,
                seed=seed, history_len=history_len, canonical=canonical,
                elapsed_s=perf_counter()-start, cpu_stats=stats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--history-len', type=int, default=64)
    p.add_argument('--canonical', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    print(json.dumps(validate(args.num_envs, args.steps, args.seed, args.history_len, args.canonical), indent=2))


if __name__ == '__main__':
    main()
