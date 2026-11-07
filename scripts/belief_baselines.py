#!/usr/bin/env python3
"""Fit B0/B1 on label-only data and report validation metrics (nats)."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import torch
from gfrl.belief_metrics import fit_baselines, evaluate_baseline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--validation', '--eval', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True, help='JSON path; Markdown uses same stem')
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    fit = torch.load(args.fit, map_location='cpu', weights_only=False)
    baselines = fit_baselines(fit)
    report = dict(fit=str(args.fit), fit_config=fit.get('config'), units='nats',
                  baselines={k: dict(count_probs=b.count_probs.tolist(), card_probs=b.card_probs.tolist())
                             for k, b in baselines.items()}, results=[])
    lines = ['| Data | Baseline | Positions | Multiset NLL | Count NLL | Count accuracy | ECE | Early / Middle / Late NLL | Zero-rate positions / copies |',
             '|---|---|---:|---:|---:|---:|---:|---|---:|']
    notes = []
    for path in args.validation:
        data = torch.load(path, map_location='cpu', weights_only=False)
        for name, baseline in baselines.items():
            metrics = evaluate_baseline(baseline, data)
            report['results'].append(dict(data=str(path), baseline=name, config=data.get('config'), **metrics))
            phases = ' / '.join(f'{m["nll"]:.4f}' if m['nll'] is not None else '—' for m in metrics['phase_nll'].values())
            lines.append(f'| {path.stem} | {name} | {len(data["decision"])} | {metrics["multiset_nll"]:.4f} | '
                         f'{metrics["count_nll"]:.4f} | {metrics["count_accuracy"]:.2%} | {metrics["ece"]:.6f} | {phases} | '
                         f'{metrics["zero_probability_positions"]} / {metrics["zero_probability_card_copies"]} |')
        notes.append(f'{path.stem}: decision tertiles = {metrics["phase_boundaries"]}; {metrics["phase_rule"]}.')
        del data
    markdown = '\n'.join(lines) + '\n\n' + '\n\n'.join(notes) + '\n'
    print(markdown)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    args.out.with_suffix('.md').write_text(markdown)


if __name__ == '__main__':
    main()
