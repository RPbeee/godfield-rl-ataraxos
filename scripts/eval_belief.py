#!/usr/bin/env python3
"""Evaluate saved autoregressive/B2 beliefs and optional B0/B1 baselines."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import torch
from gfrl.belief import belief_from_checkpoint, belief_model_type
from gfrl.belief_training import evaluate_belief
from gfrl.belief_metrics import fit_baselines, evaluate_baseline
from gfrl.runtime import configure_precision


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    p.add_argument('--validation', type=Path, nargs='+', required=True)
    p.add_argument('--fit', type=Path)
    p.add_argument('--out', type=Path, required=True, help='JSON output; Markdown saved alongside')
    p.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--presence-samples', type=int, default=256)
    p.add_argument('--presence-sample-batch-size', type=int, default=32,
                   help='maximum samples per position in each autoregressive sampling call')
    p.add_argument('--eval-batch-size', type=int, default=8)
    args = p.parse_args(argv)
    for key in ('threads', 'presence_samples', 'presence_sample_batch_size', 'eval_batch_size'):
        if getattr(args, key) <= 0:
            p.error(key+' must be positive')
    return args


def run(args):
    torch.set_num_threads(args.threads)
    configure_precision(args.device, args.tf32)
    baselines = fit_baselines(torch.load(args.fit, map_location='cpu', weights_only=False)) if args.fit else {}
    report = dict(units='nats', config={k: str(v) if isinstance(v, Path) else
                  [str(p) for p in v] if isinstance(v, list) else v for k, v in vars(args).items()}, results=[])
    lines = ['| Data | Model | Multiset NLL | Count NLL | Count accuracy | ECE | Early / Middle / Late NLL |',
             '|---|---|---:|---:|---:|---:|---|']

    def add(path, name, result, **metadata):
        report['results'].append(dict(data=str(path), model=name, **metadata, **result))
        phases = ' / '.join('—' if v['nll'] is None else f'{v["nll"]:.4f}' for v in result['phase_nll'].values())
        lines.append(f'| {path.stem} | {name} | {result["multiset_nll"]:.4f} | {result["count_nll"]:.4f} | '
                     f'{result["count_accuracy"]:.2%} | {result["ece"]:.6f} | {phases} |')

    # Load one dataset and one model at a time, including on the accelerator.
    for path in args.validation:
        data = torch.load(path, map_location='cpu', weights_only=False)
        for name, baseline in baselines.items():
            add(path, name, evaluate_baseline(baseline, data))
        for checkpoint in args.checkpoint:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model = belief_from_checkpoint(state).to(args.device)
            kind = belief_model_type(model)
            torch.manual_seed(args.seed)
            result = evaluate_belief(model, data, args.device, args.eval_batch_size,
                                     args.presence_samples, args.presence_sample_batch_size)
            add(path, f'{kind} ({checkpoint.parent.name}/{checkpoint.name})', result,
                checkpoint=str(checkpoint), model_type=kind, step=state.get('step'),
                parameters=sum(p.numel() for p in model.parameters()))
            del model, state
        del data
    markdown = '\n'.join(lines)+'\n'
    print(markdown, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    args.out.with_suffix('.md').write_text(markdown)
    return report


if __name__ == '__main__':
    run(parse_args())
