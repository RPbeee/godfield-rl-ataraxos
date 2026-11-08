#!/usr/bin/env python3
"""CPU P3 environment smoke: evaluation roots, 16 B1 worlds/root, 10 decisions.

B1 is fit to the saved label-only fitting set. Invalid samples are explicitly
rejected and resampled (including non-gift cards from B1's smoothing).
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl import SearchEnv, VecEnv
from gfrl.actor import Actor
from gfrl.belief_metrics import fit_baselines
from gfrl.evaluation import load_policy_model
from gfrl.transfer import tensor_batch


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/m4-s-ppo-v1/checkpoint.pt')
    p.add_argument('--fit', default='runs/belief-smoke/fit.pt')
    p.add_argument('--roots', type=int, default=8)
    p.add_argument('--worlds', type=int, default=16)
    p.add_argument('--depth', type=int, default=10)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261108)
    args = p.parse_args()
    if min(args.roots, args.worlds, args.depth, args.threads) <= 0:
        p.error('roots, worlds, depth, threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = load_policy_model(args.checkpoint)
    src = VecEnv(args.roots, args.threads, eval_pairs=10000,
                 opponent='external', eval_seed=args.seed, track_projection=True)
    policy = Actor(model, src, 'cpu')
    all_rows = np.arange(src.num_envs, dtype=np.int32)
    # Move past the initial deal so the smoke also exercises real histories.
    for _ in range(20):
        src.step(policy.act().choices)
    # Collect supported roots from a common evaluation boundary.
    for _ in range(1000):
        eligible = np.flatnonzero(src.determinize_support(all_rows) == 0)
        if len(eligible) >= min(4, args.roots):
            break
        src.step(policy.act().choices)
    else:
        raise RuntimeError('no supported evaluation roots')
    selected = eligible[:args.roots].astype(np.int32)
    root_rows = np.repeat(selected, args.worlds)
    baseline = fit_baselines(torch.load(args.fit, map_location='cpu', weights_only=False))['B1']
    pn, pk = baseline.count_probs.numpy(), baseline.card_probs.numpy()
    assignments = np.zeros((len(root_rows), 297), dtype=np.int32)
    attempts = 0
    # Rejection uses the exact C++ conversion exposed to Python; no privileged labels.
    for i, row in enumerate(root_rows):
        for _ in range(10000):
            attempts += 1
            n = rng.choice(19, p=pn)
            assignment = rng.multinomial(n, pk).astype(np.int32)
            try:
                src.unused_assignment(int(row), assignment)
            except ValueError:
                continue
            assignments[i] = assignment
            break
        else:
            raise RuntimeError('B1 rejection sampler exhausted')
    seeds = rng.integers(0, np.iinfo(np.uint64).max, len(root_rows), dtype=np.uint64)
    env = SearchEnv(len(root_rows), args.threads, max_decisions=args.depth)
    # Warm creation buffers and engine paths, then measure repeated world loading.
    assert np.all(env.load_worlds(src, root_rows, assignments, seeds) == 0)
    repeats = 20
    start = perf_counter()
    for _ in range(repeats):
        status = env.load_worlds(src, root_rows, assignments, seeds)
    creation_seconds = perf_counter() - start
    assert np.all(status == 0)
    actor = Actor(model, env, 'cpu')
    start = perf_counter()
    stages = 0
    while not env.episode_done.all():
        env.step(actor.act().choices)
        stages += 1
    rollout_seconds = perf_counter() - start
    rows = np.arange(env.num_envs, dtype=np.int32)
    start = perf_counter()
    leaves = env.observe(rows, env.searcher_seat.astype(np.int32))
    memory = model.encode(tensor_batch(leaves, 'cpu'))
    logits = model.value_head(memory[:, 0]).float()
    probabilities = logits.softmax(-1)
    # Value classes are win/loss/draw, relative to the selected searcher.
    values = probabilities[:, 0] - probabilities[:, 1]
    leaf_seconds = perf_counter() - start
    report = dict(checkpoint=args.checkpoint, fit=args.fit, seed=args.seed,
                  roots=len(selected), worlds_per_root=args.worlds, worlds=env.num_envs,
                  depth=args.depth, cpu_threads=args.threads, sample_attempts=attempts,
                  load_status_ok=int((status == 0).sum()),
                  creation_worlds_per_second=env.num_envs*repeats/creation_seconds,
                  rollout_worlds_per_second=env.num_envs/rollout_seconds,
                  rollout_decisions_per_second=int(env.decisions.sum())/rollout_seconds,
                  rollout_seconds=rollout_seconds, policy_stages=stages,
                  decisions=int(env.decisions.sum()), terminal=int((env.episode_done & ~env.truncated).sum()),
                  truncated=int(env.truncated.sum()), terminal_results=env.result.tolist(), leaf_seconds=leaf_seconds,
                  leaf_value_mean=values.mean().item(), leaf_value_range=[values.min().item(), values.max().item()],
                  leaf_values=values.reshape(len(selected), args.worlds).tolist())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
