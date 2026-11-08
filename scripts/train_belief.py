#!/usr/bin/env python3
"""Train belief from fixed-policy self play without saving generated positions."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl.belief import (Belief, IndependentBelief, BeliefConfig,
                         belief_from_checkpoint, belief_model_type)
from gfrl.model import OBS_FIELDS
from gfrl.belief_training import PositionStream, ReplayBuffer, evaluate_belief
from gfrl.belief_metrics import fit_baselines, evaluate_baseline
from gfrl.actor import synchronize
from gfrl.runtime import configure_precision


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', choices=('autoregressive', 'independent'), default='autoregressive')
    p.add_argument('--presence-sample-batch-size', type=int, default=None)
    p.add_argument('--run-dir', default='runs/belief/train')
    p.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    p.add_argument('--resume', nargs='?', const='auto')
    p.add_argument('--validation', nargs='+', default=[])
    p.add_argument('--fit')
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    for name, value in dict(buffer_size=200000, warmup_positions=50000, batch_size=1024,
                            microbatch=32, steps=100000, warmup_steps=1000, eval_every=1000,
                            save_every=1000, log_every=100, eval_batch_size=16,
                            presence_samples=64, num_envs=64, threads=4, seed=0,
                            decoder_depth=2, decoder_heads=8, decoder_ff=0).items():
        p.add_argument('--'+name.replace('_', '-'), type=int, default=value)
    for name, value in dict(sample_prob=.25, replay_ratio=2., lr=3e-4, max_hours=23., pool_fraction=.2).items():
        p.add_argument('--'+name.replace('_', '-'), type=float, default=value)
    a = p.parse_args(argv)
    if not 0 < a.sample_prob <= 1 or not 0 <= a.pool_fraction <= 1:
        p.error('invalid sampling/routing probability')
    if not 0 < a.warmup_positions <= a.buffer_size or a.replay_ratio <= 0 or a.lr <= 0 or a.max_hours <= 0:
        p.error('require 0 < warmup <= buffer and positive ratio/lr/hours')
    for name in ('batch_size', 'microbatch', 'steps', 'eval_batch_size', 'presence_samples', 'num_envs', 'threads', 'save_every', 'log_every'):
        if getattr(a, name) <= 0:
            p.error(name+' must be positive')
    if a.presence_sample_batch_size is not None and a.presence_sample_batch_size <= 0:
        p.error('presence-sample-batch-size must be positive')
    return a


def save_checkpoint(path, model, optimizer, step, config, generation):
    state = dict(model_type=belief_model_type(model), model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                 config=config, model_config=asdict(model.config),
                 belief_config=asdict(model.belief_config), policy_checkpoint=config['checkpoint'],
                 generation=generation, torch_rng=torch.get_rng_state(),
                 numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 mps_rng=torch.mps.get_rng_state() if next(model.parameters()).device.type == 'mps' else None)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.lr * (step+1)/max(1, args.warmup_steps)
    progress = (step-args.warmup_steps)/max(1, args.steps-args.warmup_steps)
    return args.lr*.5*(1+math.cos(math.pi*min(1., progress)))


def run(args):
    start = perf_counter()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir/'checkpoint.pt'
    state = torch.load(path if args.resume == 'auto' else args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if state:
        runtime = {k: getattr(args, k) for k in ('run_dir', 'device', 'steps', 'max_hours', 'resume', 'threads', 'tf32')}
        vars(args).update(state['config']); vars(args).update(runtime)
    torch.set_num_threads(args.threads)
    configure_precision(args.device, args.tf32)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    model_class = IndependentBelief if args.model == 'independent' else Belief
    model = (belief_from_checkpoint(state) if state else
             model_class.from_policy(args.checkpoint, BeliefConfig(args.decoder_depth, args.decoder_heads, args.decoder_ff)))
    args.model = belief_model_type(model)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    step, generation = 0, 0
    if state:
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        step, generation = state['step'], state['generation']
        print(f'Resuming step {step}; replay buffer starts empty.', flush=True)
    # Each restart reserves a fresh environment stream; buffer/env are deliberately ephemeral.
    seed = int(np.random.SeedSequence([args.seed, generation]).generate_state(1, np.uint64)[0])
    stream = PositionStream(args.checkpoint, args.device, args.num_envs, args.threads,
                            seed, args.sample_prob, args.pool_fraction)
    generation += 1
    if state:
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        if args.device == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if args.device == 'mps' and state.get('mps_rng') is not None:
            torch.mps.set_rng_state(state['mps_rng'])
    config = vars(args).copy()
    (run_dir/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()), encoder=asdict(model.config), decoder=asdict(model.belief_config))), flush=True)
    validations = {p: torch.load(p, map_location='cpu', weights_only=False) for p in args.validation}
    baselines = fit_baselines(torch.load(args.fit, map_location='cpu', weights_only=False)) if args.fit else {}
    baseline_results = {p: {k: evaluate_baseline(b, data) for k, b in baselines.items()} for p, data in validations.items()}
    buffer, credit = None, 0.
    generated = learned = 0
    collect_s = learn_s = 0.
    loss_value = None
    last_warmup_log = start
    while step < args.steps and perf_counter()-start < args.max_hours*3600:
        begin = perf_counter()
        batch = stream.next()
        if buffer is None:
            buffer = ReplayBuffer(args.buffer_size, batch, args.device)
            print(f'Replay capacity {args.buffer_size}: {sum(v.numel()*v.element_size() for v in buffer.data.values())/2**30:.3f} GiB', flush=True)
        buffer.add(batch)
        synchronize(torch.device(args.device))
        collect_s += perf_counter()-begin
        n = len(batch['label_opp_hand']); generated += n
        if buffer.size < args.warmup_positions:
            if perf_counter()-last_warmup_log >= 10:
                print(json.dumps(dict(step=step, warming_up=True, buffer_positions=buffer.size,
                                      buffer_fill=buffer.size/args.buffer_size,
                                      collect_positions_per_s=generated/max(collect_s, 1e-9))), flush=True)
                last_warmup_log = perf_counter()
            continue
        credit += n*args.replay_ratio
        while credit >= args.batch_size and step < args.steps:
            if perf_counter()-start >= args.max_hours*3600:
                break
            begin = perf_counter()
            batch = buffer.sample(args.batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for i in range(0, args.batch_size, args.microbatch):
                obs = {k: batch[k][i:i+args.microbatch] for k in OBS_FIELDS}
                lp, _, _ = model.log_prob(obs, batch['label_opp_hand'][i:i+args.microbatch])
                loss = -lp.sum()/args.batch_size
                loss.backward(); loss_value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            for group in optimizer.param_groups:
                group['lr'] = lr_at(step, args)
            optimizer.step()
            synchronize(torch.device(args.device))
            learn_s += perf_counter()-begin
            step += 1; learned += args.batch_size; credit -= args.batch_size
            metrics = dict(step=step, loss=loss_value, lr=optimizer.param_groups[0]['lr'],
                           generated_positions=generated, collect_positions_per_s=generated/max(collect_s, 1e-9),
                           learn_positions_per_s=learned/max(learn_s, 1e-9), buffer_positions=buffer.size,
                           buffer_fill=buffer.size/args.buffer_size, elapsed_s=perf_counter()-start)
            if validations and ((args.eval_every and step % args.eval_every == 0) or step == args.steps):
                # Evaluation sampling cannot perturb the training RNG.
                rng = torch.get_rng_state()
                accel = torch.cuda.get_rng_state_all() if args.device == 'cuda' else None
                mps_rng = torch.mps.get_rng_state() if args.device == 'mps' else None
                torch.manual_seed(args.seed+12345)
                metrics['evaluation'] = {}
                print('| Data | Model | Multiset NLL | Count NLL | Count accuracy | ECE | Early / Middle / Late NLL |', flush=True)
                print('|---|---|---:|---:|---:|---:|---|', flush=True)
                for p, data in validations.items():
                    results = dict(baseline_results[p], belief=evaluate_belief(model, data, args.device, args.eval_batch_size, args.presence_samples, args.presence_sample_batch_size))
                    metrics['evaluation'][p] = results
                    for name, r in results.items():
                        phases = ' / '.join('—' if v['nll'] is None else f'{v["nll"]:.4f}' for v in r['phase_nll'].values())
                        print(f'| {Path(p).stem} | {name} | {r["multiset_nll"]:.4f} | {r["count_nll"]:.4f} | {r["count_accuracy"]:.2%} | {r["ece"]:.6f} | {phases} |', flush=True)
                torch.set_rng_state(rng)
                if accel is not None:
                    torch.cuda.set_rng_state_all(accel)
                if mps_rng is not None:
                    torch.mps.set_rng_state(mps_rng)
            if step % args.log_every == 0 or step == args.steps or 'evaluation' in metrics:
                with (run_dir/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(metrics, allow_nan=False)+'\n')
                print(json.dumps({k: v for k, v in metrics.items() if k != 'evaluation'}), flush=True)
            if step % args.save_every == 0:
                save_checkpoint(path, model, optimizer, step, config, generation)
    save_checkpoint(path, model, optimizer, step, config, generation)
    return path


if __name__ == '__main__':
    run(parse_args())
