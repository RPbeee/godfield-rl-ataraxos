"""Run from the repo root: .venv/bin/python tests/python/bench_vecenv.py."""
import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from gfrl import VecEnv


def choices(mask, rng):
    # One uniform integer rank per row; no preference for early columns.
    rank = (rng.random(len(mask)) * mask.sum(axis=1)).astype(np.int32)
    return (mask.cumsum(axis=1) > rank[:, None]).argmax(axis=1).astype(np.int32)


def main():
    print("N threads stages/s decisions/s episodes/s stages/decision auto_fraction truncation_rate")
    for n in (256, 1024, 4096):
        for threads in (1, 4, 8):
            env = VecEnv(n, threads, 42)
            mask = env.mask
            rng = np.random.default_rng(123)
            before = env.stats()
            start = time.perf_counter()
            for _ in range(300):
                env.step(choices(mask, rng))
            seconds = time.perf_counter() - start
            s = {k: v - before[k] for k, v in env.stats().items()}
            print(n, threads, *(f"{v:.2f}" for v in (
                s["policy_stages"] / seconds, s["decisions"] / seconds,
                s["episodes"] / seconds, s["policy_stages"] / s["decisions"],
                s["automatic_decisions"] / s["decisions"],
                s["truncations"] / s["episodes"] if s["episodes"] else 0)), flush=True)


if __name__ == "__main__":
    main()
