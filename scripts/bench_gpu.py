#!/usr/bin/env python3
"""CUDA-resident collection benchmark. Reports decisions and policy stages separately."""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import torch

from gfrl.gpu import GpuEnv
from gfrl.gpu.actor import GpuActor,ParallelGpuActor
from gfrl.evaluation import load_policy_model


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    with GpuEnv(args.num_envs, base_seed=args.seed, history_len=args.history_len, pooled=getattr(args,'pooled',False),
                arena_bytes=getattr(args,'arena_bytes',None)) as env:
        streams=getattr(args,'streams',1)
        actor_class=ParallelGpuActor if streams>1 else GpuActor
        actor = (actor_class(load_policy_model(str(args.checkpoint), str(env.device)), env, dtype=args.dtype,
                            **({'streams':streams} if streams>1 else {})) if args.checkpoint else None)
        if actor and args.cuda_graph:
            actor.capture()
        def advance():
            if actor:
                return actor.round()[0]
            choices = torch.rand((env.num_envs,21), device=env.device).masked_fill(~env.mask, -1).argmax(-1).to(torch.int32)
            env.step(choices)
            return None
        for _ in range(args.warmup):
            advance()
        torch.cuda.synchronize()
        before = env._stats.to(torch.int64).sum(0).cpu()
        total_start = perf_counter()
        complete = None
        for _ in range(args.steps):
            complete = advance()
        torch.cuda.synchronize()
        elapsed = perf_counter()-total_start
        counts = env._stats.to(torch.int64).sum(0).cpu()-before
        if actor and not complete.all().item():
            raise RuntimeError('decision round did not finish all rows')
        env.stats()
        arena_peak = int(env._high_water.to(torch.int64).max())
        decisions = int(counts[1]-counts[3])
        if actor and decisions != args.steps*args.num_envs:
            raise RuntimeError('incorrect number of completed model decisions')
        return dict(mode='model' if actor else 'engine', checkpoint=str(args.checkpoint),
                    num_envs=args.num_envs, history_len=args.history_len, dtype=args.dtype,
                    packed_encoder=bool(actor and actor.packed_encoder is not None),
                    encoder_chunk=actor.packed_encoder.chunk if actor and actor.packed_encoder is not None else None,
                    split_step=env.split_step,compact_step=env.compact_step,
                    streams=streams,seed=args.seed,warmup=args.warmup,steps=args.steps,cuda_graph=bool(actor and args.cuda_graph),
                    model_config=asdict(actor.model.config) if actor else None,
                    mean_valid_tokens_final=float(env.obs_valid.float().sum(-1).mean()),
                    private_rules=os.environ.get('GFRL_GPU_PRIVATE_RULES','0')=='1',
                    engine_kernel=env._module.attributes('gpu_advance' if env.split_step else 'gpu_step'),
                    engine_kernels={name:env._module.attributes(name) for name in
                                    (('gpu_choose','gpu_advance','gpu_encode_view','gpu_observe') if env.split_step else ('gpu_step','gpu_observe'))},
                    tma_linear=bool(actor and actor.packed_encoder is not None and actor.packed_encoder.encoders[0].tma),
                    pooled_arena=env._pool is not None,group_actions=env.group_actions,
                    register_limit=int(os.environ.get('GFRL_GPU_MAX_REGISTERS','0')),
                    elapsed_s=elapsed, positions=decisions, positions_per_s=decisions/elapsed,
                    policy_stages=int(counts[2]), policy_stages_per_s=int(counts[2])/elapsed,
                    engine_decisions=int(counts[1]), automatic_decisions=int(counts[3]),
                    episodes=int(counts[0]), target_positions_per_s=300000,
                    target_met=decisions/elapsed>=300000,
                    arena_bytes_per_env=env.arena_bytes, arena_peak_bytes=arena_peak,
                    device=torch.cuda.get_device_name(), torch_version=torch.__version__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--history-len',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pooled',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--arena-bytes',type=int)
    p.add_argument('--streams',type=int,default=1)
    p.add_argument('--dtype',choices=['fp32','bf16','fp16'],default='bf16')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--cuda-graph',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    print(json.dumps(run(args),indent=2))


if __name__=='__main__':
    main()
