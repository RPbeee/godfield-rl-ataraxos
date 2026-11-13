#!/usr/bin/env python3
"""Checkpoint PPO benchmark on a saved, real, completed-game rollout."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl.learner import Learner, LearnerConfig
from gfrl.model import Model, ModelConfig
from gfrl.rollout import RolloutBatch
from gfrl.runtime import configure_precision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--collect', action='store_true')
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--games', type=int, default=2048)
    p.add_argument('--collect-warmup', type=int, default=20)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--dtype', choices=['fp32', 'bf16', 'fp16'], default='bf16')
    p.add_argument('--minibatch-size', type=int, default=4096)
    p.add_argument('--microbatch-size', type=int, default=512)
    p.add_argument('--microbatch-sweep', type=int, nargs='+')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--backend', default='eager')
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    configure_precision('cuda', args.tf32)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = Model(ModelConfig(**state['model_config']))
    model.load_state_dict(state['model'])
    cfg = LearnerConfig(**{f.name: state['config'][f.name] for f in fields(LearnerConfig)
                           if f.name in state['config']})
    cfg.minibatch_size, cfg.microbatch_size = args.minibatch_size, args.microbatch_size
    learner = Learner(model, cfg, 'cuda', args.dtype)
    learner.ema.load_state_dict(state['ema'])
    learner.optimizer.load_state_dict(state['optimizer'])
    learner.optimizer_steps = state.get('optimizer_steps', 0)
    iteration = state['iteration'] + 1
    if args.collect:
        from gfrl.gpu.collector import GpuCollector
        collector = GpuCollector(model, num_envs=args.num_envs, seed=42, dtype='bf16',
                                 pool_fraction=0, pooled=True, arena_bytes=262144)
        for i in range(args.collect_warmup):
            batch = collector.collect(128)
            print(json.dumps(dict(collect_warmup=i, decisions=len(batch))), flush=True)
            del batch
        batch = collector.collect(args.games)
        torch.save(dict(columns={k:v.cpu() for k,v in batch.columns.items()},
                        stages={k:v.cpu() for k,v in batch.stages.items()}), args.data)
        print(json.dumps(dict(saved=str(args.data), decisions=len(batch))), flush=True)
        collector.close()
        return
    data = torch.load(args.data, weights_only=False)
    batch = RolloutBatch({k:v.cuda() for k,v in data['columns'].items()},
                         {k:v.cuda() for k,v in data['stages'].items()})
    del data
    if args.limit:
        batch = batch.take(torch.arange(min(args.limit, len(batch)), device='cuda'))
    if args.backend != 'eager':
        from gfrl.fast_learner import configure_fast_learner
        configure_fast_learner(learner, args.backend)
    print(json.dumps(dict(checkpoint=str(args.checkpoint), sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                          model_config=vars(model.config), learner_config=vars(cfg),
                          backend=args.backend, dtype=args.dtype, decisions=len(batch), stages=len(batch.stages['choice']),
                          mean_valid_tokens=float(batch.obs_valid.float().sum(1).mean()),
                          gpu=torch.cuda.get_device_name(), sm=torch.cuda.get_device_properties(0).multi_processor_count,
                          torch=torch.__version__)), flush=True)
    for micro in args.microbatch_sweep or [args.microbatch_size]:
        cfg.microbatch_size = micro
        learner.model.load_state_dict(state['model'])
        learner.ema.load_state_dict(state['ema'])
        learner.optimizer.load_state_dict(state['optimizer'])
        learner.optimizer_steps = state['optimizer_steps']
        np.random.seed(42)
        measure(learner, batch, args, iteration)


def measure(learner, batch, args, iteration):
    from gfrl.runtime import compiler_metrics, metric_delta
    compile_before = compiler_metrics()
    begin = perf_counter()
    for i in range(args.warmup):
        learner.update(batch, iteration+i)
    torch.cuda.synchronize()
    print(json.dumps(dict(microbatch_size=learner.config.microbatch_size,
                          warmup_s=perf_counter()-begin,
                          warmup_compiler=metric_delta(compile_before))), flush=True)
    compile_before = compiler_metrics()
    torch.cuda.reset_peak_memory_stats()
    rates = []
    for i in range(args.iterations):
        begin = perf_counter()
        metrics = learner.update(batch, iteration+args.warmup+i)
        torch.cuda.synchronize()
        elapsed = perf_counter()-begin
        rates.append(elapsed)
        print(json.dumps(dict(iteration=i, elapsed_s=elapsed, decisions_per_s=len(batch)/elapsed,
                              metrics=metrics)), flush=True)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            learner.update(batch, iteration+args.warmup+args.iterations)
        print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35), flush=True)
    print(json.dumps(dict(elapsed_s=sum(rates), decisions=len(batch)*len(rates),
                          decisions_per_s=len(batch)*len(rates)/sum(rates),
                          microbatch_size=learner.config.microbatch_size,
                          tf32=args.tf32, measured_compiler=metric_delta(compile_before),
                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())), flush=True)


if __name__ == '__main__':
    main()
