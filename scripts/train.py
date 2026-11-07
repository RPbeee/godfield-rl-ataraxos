#!/usr/bin/env python3
"""Synchronous self-play PPO. --iterations counts additional iterations on resume."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, fields
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
    p.add_argument('--resume', nargs='?', const='auto')
    p.add_argument('--size', choices=['S', 'M', 'L'], default='S')
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
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
    positive = ('num_envs', 'threads', 'games_per_iter', 'iterations', 'pool_every',
                'pool_size', 'eval_pairs', 'minibatch_size', 'decision_capacity', 'stage_capacity')
    if any(getattr(args, k) <= 0 for k in positive) or args.eval_every < 0:
        p.error('counts must be positive (eval_every may be zero)')
    if args.microbatch_size < 0:
        p.error('microbatch-size must be nonnegative (0 = automatic)')
    if not 0 <= args.pool_fraction <= 1 or args.max_hours <= 0:
        p.error('invalid pool_fraction or max_hours')
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


def save_checkpoint(path, learner, iteration, pool, positions, config, elapsed):
    state = dict(model=learner.model.state_dict(), ema=learner.ema.state_dict(),
                 optimizer=learner.optimizer.state_dict(), iteration=iteration, pool=pool,
                 optimizer_steps=learner.optimizer_steps,
                 seed_positions=positions.copy(), config=config,
                 model_config=asdict(learner.model.config), elapsed_s=elapsed,
                 torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state())
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def run(args):
    start = perf_counter()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / 'checkpoint.pt'
    state = None
    if args.resume:
        state = torch.load(checkpoint if args.resume == 'auto' else args.resume,
                           map_location='cpu', weights_only=False)
        # Resume the saved experiment settings; execution length/device/path remain caller-controlled.
        runtime = {k: getattr(args, k) for k in ('iterations', 'device', 'threads', 'run_dir',
                                                'resume', 'max_hours', 'config', 'microbatch_size',
                                                'tf32', 'compile', 'compile_mode_actor', 'compile_mode_learner')}
        vars(args).update(state['config'])
        vars(args).update(runtime)
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
        elapsed_before = state['elapsed_s']
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
    configure_model(learner.model, args)
    configure_model(learner.ema, args)
    # Native VecEnv owns its within-session per-env episode stream. Reserve a new
    # namespace on every construction, so a restart never reuses unfinished deals.
    train_seed = stream_seed(args.seed, 0, positions['train'])
    positions['train'] += 1
    collector = Collector(learner.model, num_envs=args.num_envs, threads=args.threads,
                          seed=train_seed, device=args.device, dtype=args.dtype,
                          pool_fraction=args.pool_fraction, decision_capacity=args.decision_capacity,
                          stage_capacity=args.stage_capacity)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    last_duration = 0.
    for _ in range(args.iterations):
        # Leave a full previous iteration of headroom before starting another.
        if perf_counter()-start + last_duration >= args.max_hours*3600:
            save_checkpoint(checkpoint, learner, iteration, pool, positions, config,
                            elapsed_before+perf_counter()-start)
            break
        compile_before = compiler_metrics()
        begin = perf_counter()
        rollout = collector.collect(args.games_per_iter, pool)
        collected = perf_counter()
        iteration += 1
        metrics = learner.update(rollout.decisions, iteration)
        learned = perf_counter()
        if iteration % args.pool_every == 0:
            pool.append(learner.pool_entry(iteration))
            pool = pool[-args.pool_size:]
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


if __name__ == '__main__':
    run(parse_args())
