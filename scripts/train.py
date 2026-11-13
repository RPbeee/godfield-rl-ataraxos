#!/usr/bin/env python3
"""Synchronous self-play PPO. --iterations counts additional iterations on resume."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch

from gfrl.actor import Actor
from gfrl.evaluation import evaluate_policy
from gfrl.learner import Learner, LearnerConfig
from gfrl.model import Model, ModelConfig
from gfrl.rollout import Collector
from gfrl.runtime import (add_runtime_args, configure_precision, configure_model,
                          compiler_metrics, metric_delta)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--resume', nargs='?', const='auto')
    source.add_argument('--init-from', type=Path)
    p.add_argument('--size', choices=['S', 'M', 'L'], default='S')
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--env-backend', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--selfplay-streams',type=int,default=1)
    p.add_argument('--gpu-arena-bytes',type=int)
    p.add_argument('--gpu-pooled-arena',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--rollout-storage', choices=['device', 'cpu'], default='device')
    p.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    for name, default in dict(num_envs=64, threads=4, games_per_iter=128, iterations=100,
                              pool_every=10, pool_size=8, eval_every=10, eval_pairs=100,
                              seed=0, d=0, depth=0, decision_capacity=4096,
                              stage_capacity=16384).items():
        p.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    p.add_argument('--pool-fraction', type=float, default=.2)
    p.add_argument('--run-dir', default='runs/train')
    p.add_argument('--max-hours', type=float, default=23.)
    defaults = LearnerConfig()
    for f in fields(defaults):
        value = getattr(defaults, f.name)
        p.add_argument('--'+f.name.replace('_', '-'), default=value,
                       **(dict(action=argparse.BooleanOptionalAction) if isinstance(value, bool)
                          else dict(type=type(value))))
    add_runtime_args(p)
    args = p.parse_args(argv)
    if args.config:
        config = json.loads(args.config.read_text())
        unknown = set(config) - set(vars(args))
        if unknown:
            p.error('unknown config keys: '+', '.join(sorted(unknown)))
        for k, v in config.items():
            setattr(args, k, v)
    positive = ('selfplay_streams','num_envs', 'threads', 'games_per_iter', 'iterations', 'pool_every',
                'pool_size', 'eval_pairs', 'minibatch_size', 'decision_capacity', 'stage_capacity')
    if any(getattr(args, k) <= 0 for k in positive) or args.eval_every < 0:
        p.error('counts must be positive (eval_every may be zero)')
    if args.rollout_storage not in ('device', 'cpu'):
        p.error('rollout-storage must be device or cpu')
    if args.microbatch_size < 0:
        p.error('microbatch-size must be nonnegative (0 = automatic)')
    if not 0 <= args.pool_fraction <= 1 or args.max_hours <= 0:
        p.error('invalid pool_fraction or max_hours')
    if args.env_backend == 'cuda' and (args.device != 'cuda' or args.pool_fraction != 0 or args.rollout_storage != 'device' or args.compile != 'off'):
        p.error('CUDA env backend requires --device cuda --pool-fraction 0 --rollout-storage device --compile off')
    if args.selfplay_streams>args.num_envs or (args.env_backend=='cpu' and args.selfplay_streams!=1):
        p.error('selfplay-streams requires CUDA backend and cannot exceed num-envs')
    if args.gpu_arena_bytes is not None and (args.gpu_arena_bytes<4096 or args.gpu_arena_bytes%16):
        p.error('gpu-arena-bytes must be >=4096 and divisible by 16')
    if args.env_backend=='cpu' and (args.gpu_pooled_arena or args.gpu_arena_bytes is not None):
        p.error('GPU arena settings require --env-backend cuda')
    if args.resume and args.init_from:
        p.error('--resume and --init-from are mutually exclusive')
    if not 0 < args.gamma <= 1:
        p.error('gamma must be in (0,1]')
    return args


def stream_seed(seed, domain, position):
    return int(np.random.SeedSequence([seed, domain, position]).generate_state(1, np.uint64)[0])


@contextmanager
def evaluation_rng(seed, device):
    """Restore the training sampler RNG, including the selected accelerator."""
    device = torch.device(device)
    accelerator_state = None
    if device.type == 'mps':
        accelerator_state = torch.mps.get_rng_state()
    elif device.type == 'cuda':
        accelerator_state = torch.cuda.get_rng_state(device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if device.type == 'mps':
            torch.mps.manual_seed(seed)
        elif device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            if device.type == 'mps':
                torch.mps.set_rng_state(accelerator_state)
            elif device.type == 'cuda':
                torch.cuda.set_rng_state(accelerator_state, device)


def save_checkpoint(path, learner, iteration, pool, positions, config, elapsed, stop_reason=None):
    state = dict(model=learner.model.state_dict(), ema=learner.ema.state_dict(),
                 optimizer=learner.optimizer.state_dict(), iteration=iteration, pool=pool,
                 optimizer_steps=learner.optimizer_steps,
                 seed_positions=positions.copy(), config=config,
                 model_config=asdict(learner.model.config), elapsed_s=elapsed,
                 torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state())
    state['stop_reason'] = stop_reason
    state['init_from'] = config.get('init_from')
    if learner.device.type == 'cuda':
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    elif learner.device.type == 'mps':
        state['mps_rng'] = torch.mps.get_rng_state()
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def rollout_length_metrics(decisions):
    """Mean own decisions per completed player/game sequence (pool seats excluded)."""
    c = {k: decisions.columns[k].cpu().numpy() for k in ('env', 'game', 'seat', 'outcome')}
    keys = np.column_stack([c[k] for k in ('env', 'game', 'seat')])
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    result = {'mean_own_decisions': float(counts.mean()) if len(counts) else None}
    for name, outcome in (('win', 1), ('loss', -1)):
        lengths = counts[c['outcome'][first] == outcome]
        result[f'mean_{name}_own_decisions'] = float(lengths.mean()) if len(lengths) else None
    return result


def run(args):
    start = perf_counter()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / 'checkpoint.pt'
    state = None
    provenance = None
    if checkpoint.exists() and not args.resume:
        raise ValueError(f'{checkpoint} already exists; use --resume auto')
    if args.init_from and not args.resume:
        source = Path(args.init_from).resolve()
        with source.open('rb') as f:
            provenance = dict(path=str(source), md5=hashlib.file_digest(f, 'md5').hexdigest())
        state = torch.load(source, map_location='cpu', weights_only=False)
    if args.resume:
        state = torch.load(checkpoint if args.resume == 'auto' else args.resume,
                           map_location='cpu', weights_only=False)
        # Resume the saved experiment settings; execution length/device/path remain caller-controlled.
        runtime = {k: getattr(args, k) for k in ('iterations', 'device', 'threads', 'run_dir',
                                                'resume', 'max_hours', 'config', 'microbatch_size', 'rollout_storage',
                                                'tf32', 'compile', 'compile_mode_actor', 'compile_mode_learner', 'env_backend','selfplay_streams','gpu_arena_bytes','gpu_pooled_arena')}
        if args.env_backend=='cuda':
            runtime['pool_fraction']=args.pool_fraction
        vars(args).update(state['config'])
        vars(args).update(runtime)
        provenance = state.get('init_from', state['config'].get('init_from'))
    configure_precision(args.device, args.tf32)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_config = (ModelConfig(**state['model_config']) if state else
                    ModelConfig.preset(args.size))
    if not state and (args.d or args.depth):
        values = asdict(model_config)
        if args.d:
            values.update(d=args.d, ff=args.d*4)
        if args.depth:
            values['depth'] = args.depth
        model_config = ModelConfig(**values)
    learner = Learner(Model(model_config), LearnerConfig(**{
        f.name: getattr(args, f.name) for f in fields(LearnerConfig)}), args.device, args.dtype)
    pool, iteration, elapsed_before = [], 0, 0.
    positions = dict(train=0, evaluation=0)
    if state:
        learner.model.load_state_dict(state['model'])
        learner.ema.load_state_dict(state['ema'])
        learner.optimizer.load_state_dict(state['optimizer'])
        learner.optimizer_steps = state.get('optimizer_steps', 0)
        pool, iteration = state['pool'], state['iteration']
        positions = state['seed_positions'].copy()
        elapsed_before = state['elapsed_s'] if args.resume else 0.
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        if args.device == 'cuda' and 'cuda_rng' in state:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        elif args.device == 'mps' and 'mps_rng' in state:
            torch.mps.set_rng_state(state['mps_rng'])
    configure_model(learner.model, args)
    configure_model(learner.ema, args)
    # Native VecEnv owns its within-session per-env episode stream. Reserve a new
    # namespace on every construction, so a restart never reuses unfinished deals.
    train_seed = stream_seed(args.seed, 0, positions['train'])
    positions['train'] += 1
    collector_class = Collector
    collector_kwargs={}
    if args.env_backend == 'cuda':
        from gfrl.gpu.collector import GpuCollector
        collector_class = GpuCollector
        collector_kwargs.update(streams=args.selfplay_streams,pooled=args.gpu_pooled_arena,arena_bytes=args.gpu_arena_bytes)
    collector = collector_class(learner.model, num_envs=args.num_envs, threads=args.threads,
                          seed=train_seed, device=args.device, dtype=args.dtype,
                          pool_fraction=args.pool_fraction, decision_capacity=args.decision_capacity,
                          stage_capacity=args.stage_capacity, rollout_storage=args.rollout_storage,**collector_kwargs)
    try:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config['init_from'] = provenance
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
        last_duration = 0.
        for _ in range(args.iterations):
            # Leave a full previous iteration of headroom before starting another.
            if perf_counter()-start + last_duration >= args.max_hours*3600:
                save_checkpoint(checkpoint, learner, iteration, pool, positions, config,
                                elapsed_before+perf_counter()-start, 'time_budget')
                break
            compile_before = compiler_metrics()
            if args.device == 'cuda':
                torch.cuda.reset_peak_memory_stats(learner.device)
            begin = perf_counter()
            rollout = collector.collect(args.games_per_iter, pool)
            collected = perf_counter()
            iteration += 1
            metrics = learner.update(rollout.decisions, iteration)
            learned = perf_counter()
            if args.device == 'cuda':
                metrics.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(learner.device),
                               cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(learner.device))
            if iteration % args.pool_every == 0:
                pool.append(learner.pool_entry(iteration))
                pool = pool[-args.pool_size:]
            metrics.update(rollout_length_metrics(rollout.decisions))
            metrics.update(iteration=iteration, elapsed_s=elapsed_before+learned-start,
                           games=rollout.games, decisions=len(rollout.decisions),
                           sampled_decisions=rollout.sampled_decisions,
                           truncation_rate=rollout.truncated/rollout.games,
                           collect_decisions_per_s=rollout.sampled_decisions/max(collected-begin, 1e-9),
                           learn_decisions_per_s=len(rollout.decisions)/max(learned-collected, 1e-9),
                           collect_s=collected-begin, learn_s=learned-collected,
                           pool_iterations=[entry['iteration'] for entry in pool])
            metrics.update({'collect_'+k+('_s' if k in ('env_step', 'inference', 'record', 'other') else ''): v
                            for k,v in rollout.timings.items()})
            if args.eval_every and iteration % args.eval_every == 0:
                eval_seed = stream_seed(args.seed, 1, positions['evaluation'])
                positions['evaluation'] += 1
                metrics['evaluation'] = {}
                for opponent in ('random', 'greedy'):
                    # Evaluation uses its own torch RNG as well as its own deal stream.
                    with evaluation_rng(eval_seed, args.device):
                        actor = Actor(learner.ema, collector.env, args.device, sample=True, dtype=args.dtype)
                        metrics['evaluation'][opponent] = evaluate_policy(
                            actor, opponent, args.eval_pairs, args.num_envs, args.threads, eval_seed)
            metrics.update(metric_delta(compile_before))
            metrics['elapsed_s'] = elapsed_before+perf_counter()-start
            with (run_dir / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(metrics, allow_nan=False)+'\n')
            print(json.dumps(metrics, allow_nan=False), flush=True)
            save_checkpoint(checkpoint, learner, iteration, pool, positions, config,
                            elapsed_before+perf_counter()-start)
            # Release the output-buffer lease before the next collect starts.
            del rollout
            last_duration = perf_counter()-begin
        return checkpoint
    finally:
        if args.env_backend=='cuda':
            collector.close()


if __name__ == '__main__':
    run(parse_args())
