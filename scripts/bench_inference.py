#!/usr/bin/env python3
"""Synchronized end-to-end policy benchmark, with decision-level encoder reuse."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import torch
from gfrl import VecEnv
from gfrl.actor import Actor, synchronize
from gfrl.model import Model, ModelConfig
from gfrl.runtime import (add_runtime_args, configure_precision, configure_model,
                          compiler_metrics, metric_delta, warmup_encoder)
from gfrl.benchmark import sweep_cases, print_table


def benchmark(args):
    configure_precision(args.device, args.tf32)
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    model = Model(ModelConfig.preset(args.size, pack=args.pack, chunk=args.chunk))
    env = VecEnv(args.num_envs, args.threads, 0)
    actor = Actor(model, env, args.device, dtype=args.dtype, profile=True)
    configure_model(model, args)
    compile_before = compiler_metrics()
    warmup_start = perf_counter()
    warmup_encoder(model, args.device, dtype=args.dtype, all_buckets=args.warmup_buckets == "all")
    for _ in range(args.warmup):
        env.step(actor.act().choices)
    synchronize(actor.device)
    warmup_s = perf_counter()-warmup_start
    warmup_metrics = metric_delta(compile_before)
    compile_before = compiler_metrics()
    totals = dict(env_step=0.0, encoder=0.0, heads=0.0, transfer=0.0)
    stages = decisions = games = encoded = 0
    encoder_tokens = unpacked_tokens = 0
    begin = perf_counter()
    for _ in range(args.stages):
        stages += int((env.stage >= 0).sum())
        result = actor.act()
        for key in ("encoder", "heads", "transfer"):
            totals[key] += actor.last_profile[key]
        encoded += actor.last_profile["encoded_envs"]
        encoder_tokens += actor.last_profile["encoder_tokens"]
        unpacked_tokens += actor.last_profile["unpacked_tokens"]
        synchronize(actor.device)
        since = perf_counter()
        env.step(result.choices)
        synchronize(actor.device)
        totals["env_step"] += perf_counter() - since
        decisions += int(env.decision_done.sum())
        games += int(env.episode_done.sum())
    synchronize(actor.device)
    elapsed = perf_counter() - begin
    return dict(size=args.size, device=args.device, num_envs=args.num_envs,
                         pack=args.pack, chunk=args.chunk, dtype=args.dtype,
                         threads=args.threads, batches=args.stages, warmup_batches=args.warmup,
                         tf32=args.tf32, compile=args.compile,
                         compile_mode_actor=("default" if args.device == "cpu" else args.compile_mode_actor),
                         warmup_buckets=args.warmup_buckets, warmup_s=warmup_s,
                         warmup_compiler=warmup_metrics, measured_compiler=metric_delta(compile_before),
                         parameters=sum(p.numel() for p in model.parameters()),
                         elapsed_s=elapsed, policy_stages=stages, decisions=decisions, games=games,
                         stages_per_s=stages/elapsed, decisions_per_s=decisions/elapsed,
                         games_per_s=games/elapsed, mean_encoded_envs=encoded/args.stages,
                         mean_encoder_tokens_per_step=encoder_tokens/args.stages,
                         mean_unpacked_tokens_per_step=unpacked_tokens/args.stages,
                         ms_per_step={**{k: v*1000/args.stages for k, v in totals.items()},
                                      "other": (elapsed-sum(totals.values()))*1000/args.stages,
                                      "total": elapsed*1000/args.stages},
                         mps_allocated_bytes=(torch.mps.current_allocated_memory()
                                              if args.device == "mps" else None))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("S", "M", "L"), default="S")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--stages", type=int, default=600, help="number of batched env.step calls")
    parser.add_argument("--pack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--warmup-buckets', choices=('all', 'observed'), default='all',
                        help='all compiles all 32 shapes; observed is a quick CPU smoke option')
    parser.add_argument('--sweep', action='store_true')
    add_runtime_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.threads, args.stages, args.chunk) <= 0:
        parser.error("num-envs, threads, stages, chunk must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this Python process")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python process")
    if args.warmup < 1:
        parser.error('warmup must be positive')
    results = []
    for case in sweep_cases(args):
        result = benchmark(case)
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)
    if args.sweep:
        print_table(results, 'decisions_per_s')


if __name__ == '__main__':
    main()
