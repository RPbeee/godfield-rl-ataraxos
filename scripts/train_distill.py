#!/usr/bin/env python3
"""Fine-tune checkpoint EMA on public-only, teacher-forced search shards."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl.distill import load_shard, prepare_batch, objective
from gfrl.evaluation import load_policy_model
from gfrl.runtime import configure_precision


def validation_mask(data, fraction=.03, split_seed=17):
    # Keep paired games together too: same starting deal must not cross split.
    seed = data['metadata']['seed']
    return np.array([int.from_bytes(hashlib.blake2b(f'{split_seed}:{seed}:{int(g)//2}'.encode(),
                            digest_size=8).digest(), 'little')/2**64 < fraction for g in data['game']])


def average_metrics(parts):
    total = sum(n for n, _ in parts)
    if not total:
        return None
    out = {k: sum(n*m[k] for n, m in parts)/total for k in parts[0][1] if k != 'q_roots'}
    q_count = sum(m['q_roots'] for _, m in parts)
    for k in ('q_theta', 'q_ref', 'q_search'):
        out[k] = sum(m[k]*m['q_roots'] for _, m in parts)/q_count if q_count else None
    out['q_roots'] = q_count
    out['roots'] = total
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--resume', action='store_true', help='resume last; repeat any interrupted epoch')
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    p.add_argument('--dtype', choices=('fp32', 'bf16'), default='fp32')
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    for name, default in (('epochs', 2), ('minibatch', 1024), ('microbatch', 8), ('threads', 1),
                          ('warmup-steps', 10), ('seed', 17), ('log-every', 10)):
        p.add_argument('--'+name, type=int, default=default)
    for name, default in (('kl-coef', .1), ('value-coef', .5), ('lr', 1e-5), ('grad-clip', 1.),
                          ('ema-decay', 0.), ('max-hours', .75), ('validation-fraction', .03)):
        p.add_argument('--'+name, type=float, default=default)
    args = p.parse_args()
    if min(args.epochs, args.minibatch, args.microbatch, args.threads, args.log_every) <= 0:
        p.error('epochs, batch sizes, threads and log-every must be positive')
    if not 0 < args.validation_fraction < 1 or not 0 <= args.ema_decay < 1:
        p.error('require 0 < validation-fraction < 1 and 0 <= ema-decay < 1')
    if not np.isfinite([args.max_hours, args.lr, args.kl_coef, args.value_coef, args.grad_clip]).all() or min(args.max_hours, args.lr, args.kl_coef, args.value_coef, args.grad_clip, args.warmup_steps) < 0:
        p.error('finite nonnegative optimizer/loss/time settings required')
    paths = sorted(args.data_dir.glob('shard-*.pt.gz'))
    if not paths:
        p.error('no data shards')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir/'metrics.jsonl').exists() and not args.resume:
        p.error('run-dir already contains metrics; use --resume or a new directory')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    configure_precision(args.device, args.tf32)
    rng = np.random.default_rng(args.seed)
    model = load_policy_model(args.checkpoint, args.device).requires_grad_(True)
    reference = deepcopy(model).eval().requires_grad_(False)
    if args.value_coef == 0:
        model.value_head.requires_grad_(False)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam((v for v in model.parameters() if v.requires_grad), lr=args.lr)
    start, steps, best = perf_counter(), 0, float('inf')
    split_counts = [0, 0]
    for path in paths:
        d = load_shard(path)
        mask = validation_mask(d, args.validation_fraction, args.seed)
        split_counts[0] += int((~mask).sum())
        split_counts[1] += int(mask.sum())
    if not all(split_counts):
        p.error(f'deterministic game split has train={split_counts[0]}, val={split_counts[1]} roots; '
                'generate more games or increase --validation-fraction for smoke tests')
    start_epoch = 1
    source_md5 = hashlib.md5(Path(args.checkpoint).read_bytes()).hexdigest()
    resume_path = args.run_dir/'last.pt'
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=args.device, weights_only=False)
        for key in ('checkpoint', 'data_dir', 'lr', 'kl_coef', 'value_coef', 'ema_decay', 'seed', 'validation_fraction'):
            value = str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
            if state['config'][key] != value:
                p.error(f'resume setting mismatch: {key}')
        if state['source_md5'] != source_md5 or state['split_counts'] != split_counts:
            p.error('resume checkpoint or dataset changed')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        steps = state['optimizer_steps']
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        start_epoch = state['epoch']+1 if state.get('epoch_complete') else max(1, state['epoch'])
        history = [json.loads(line) for line in (args.run_dir/'metrics.jsonl').read_text().splitlines()]
        best = min(m['val']['policy'] for m in history)
    elif args.resume and (args.run_dir/'metrics.jsonl').exists():
        p.error('metrics exist but no last checkpoint; use a new run directory')
    print(f'train_roots={split_counts[0]} val_roots={split_counts[1]} (paired deals kept together)', flush=True)
    autocast = lambda: nullcontext() if args.dtype == 'fp32' else torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16)
    def save(name, epoch, epoch_complete=False):
        weights = ema.state_dict() if args.ema_decay else model.state_dict()
        state = dict(model=model.state_dict(), ema=weights, model_config=asdict(model.config),
                     iteration=steps, pool=[], config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                     optimizer=optimizer.state_dict(), optimizer_steps=steps, seed_positions={},
                     elapsed_s=perf_counter()-start, epoch=epoch, torch_rng=torch.get_rng_state(),
                     numpy_rng=rng.bit_generator.state, split_counts=split_counts,
                     epoch_complete=epoch_complete, source_md5=source_md5)
        temp = args.run_dir/(name+'.tmp')
        torch.save(state, temp)
        temp.replace(args.run_dir/(name+'.pt'))
    def validate():
        parts = []
        # Metrics assess the exact weights exported in checkpoint['ema'].
        evaluated = ema if args.ema_decay else model
        with torch.no_grad():
            for path in paths:
                d = load_shard(path)
                rows = np.flatnonzero(validation_mask(d, args.validation_fraction, args.seed))
                for lo in range(0, len(rows), args.microbatch):
                    rr = rows[lo:lo+args.microbatch]
                    with autocast():
                        _, m = objective(evaluated, reference, prepare_batch(d, rr, args.device), args.kl_coef, args.value_coef)
                    parts.append((len(rr), m))
        return average_metrics(parts)
    def log(epoch, train, final=False):
        nonlocal best
        val = validate()
        report = dict(epoch=epoch, step=steps, elapsed_seconds=perf_counter()-start,
                      train=average_metrics(train), val=val, epoch_end=final)
        with (args.run_dir/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(report, allow_nan=False)+'\n')
        print(json.dumps(report), flush=True)
        if val['policy'] < best:
            best = val['policy']
            save('best', epoch, final)
        save('last', epoch, final)
    if start_epoch == 1 and steps == 0:
        log(0, [])
    stopped = False
    for epoch in range(start_epoch, args.epochs+1):
        parts = []
        for index in rng.permutation(len(paths)):
            d = load_shard(paths[index])
            rows = rng.permutation(np.flatnonzero(~validation_mask(d, args.validation_fraction, args.seed)))
            for lo in range(0, len(rows), args.minibatch):
                if perf_counter()-start >= args.max_hours*3600:
                    stopped = True
                    break
                rr = rows[lo:lo+args.minibatch]
                optimizer.zero_grad(set_to_none=True)
                for off in range(0, len(rr), args.microbatch):
                    micro = rr[off:off+args.microbatch]
                    with autocast():
                        loss, m = objective(model, reference, prepare_batch(d, micro, args.device), args.kl_coef, args.value_coef)
                    (loss*len(micro)/len(rr)).backward()
                    parts.append((len(micro), m))
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                for group in optimizer.param_groups:
                    group['lr'] = args.lr*min((steps+1)/max(args.warmup_steps, 1), 1.)
                optimizer.step()
                steps += 1
                if args.ema_decay:
                    with torch.no_grad():
                        for a, b in zip(ema.parameters(), model.parameters()):
                            a.lerp_(b, 1-args.ema_decay)
                if steps % args.log_every == 0:
                    log(epoch, parts)
            if stopped:
                break
        log(epoch, parts, final=not stopped)
        if stopped:
            break
    (args.run_dir/'complete.json').write_text(json.dumps(dict(steps=steps, stopped_by_time=stopped,
                                                            best_val_policy=best))+'\n')


if __name__ == '__main__':
    main()
