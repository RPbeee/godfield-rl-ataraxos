#!/usr/bin/env python3
"""Measure the complete GPU collector, including trajectory storage/compaction."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import torch
from gfrl.model import Model,ModelConfig
from gfrl.gpu.collector import GpuCollector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--pool-fraction',type=float,default=0.)
    p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--games',type=int,default=512)
    p.add_argument('--warmup-games',type=int,default=128)
    p.add_argument('--warmup-iterations',type=int,default=1)
    p.add_argument('--iterations',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pooled',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--arena-bytes',type=int)
    p.add_argument('--streams',type=int,default=1)
    args=p.parse_args()
    if args.iterations<1 or args.warmup_iterations<0:
        p.error('iterations must be positive and warmup-iterations nonnegative')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=Model(ModelConfig(**state['model_config'])).cuda().eval()
    model.load_state_dict(state['model'])
    pool=state.get('pool',[])
    del state
    c=GpuCollector(model,num_envs=args.num_envs,seed=args.seed,dtype='bf16',pool_fraction=args.pool_fraction,pooled=args.pooled,arena_bytes=args.arena_bytes,streams=args.streams)
    try:
        for _ in range(args.warmup_iterations if args.warmup_games else 0):
            warmup=c.collect(args.warmup_games,pool);del warmup
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampled=completed=games=external=opponent=0
        rates=[]
        begin=perf_counter()
        for _ in range(args.iterations):
            tick=perf_counter()
            batch=c.collect(args.games,pool)
            sampled+=batch.sampled_decisions;completed+=len(batch);games+=batch.games
            external+=batch.timings['total_decisions'];opponent+=batch.timings['opponent_decisions']
            rates.append(batch.sampled_decisions/(perf_counter()-tick))
            del batch
        torch.cuda.synchronize()
        elapsed=perf_counter()-begin
        rate=sampled/elapsed
        print(json.dumps(dict(mode='collection',checkpoint=str(args.checkpoint),num_envs=args.num_envs,
                              elapsed_s=elapsed,sampled_positions=sampled,
                              pool_fraction=args.pool_fraction,pool_iterations=[e['iteration'] for e in pool],
                              external_decisions=external,opponent_decisions=opponent,
                              external_decisions_per_s=external/elapsed,
                              completed_positions=completed,completed_positions_per_s=completed/elapsed,positions_per_s=rate,
                              games=games,target_positions_per_s=300000,target_met=rate>=300000,
                              iterations=args.iterations,warmup_iterations=args.warmup_iterations,
                              iteration_positions_per_s=rates,
                              streams=args.streams,pooled_arena=c.env._pool is not None,arena_bytes=c.env.arena_bytes,
                              mean_valid_tokens_final=float(c.env.obs_valid.float().sum(-1).mean()),
                              dtype='bf16',packed_encoder=c.actor.packed_encoder is not None,
                              encoder_chunk=c.actor.packed_encoder.chunk if c.actor.packed_encoder is not None else None,
                              tma_available=bool(c.actor.packed_encoder is not None and c.actor.packed_encoder.encoders[0].tma),
                              tma_linear=bool(c.actor.packed_encoder is not None and c.actor.packed_encoder.encoders[0].tma
                                              and 128<=model.config.d<=256),
                              split_step=c.env.split_step,compact_step=c.env.compact_step,
                              gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              stats=c.env.stats()),indent=2))
    finally:
        c.close()


if __name__=='__main__':
    main()
