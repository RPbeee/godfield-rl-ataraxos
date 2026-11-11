#!/usr/bin/env python3
"""Render a completed distillation session and comparable search failure rates."""
import argparse
import json
from pathlib import Path


def search_rates(counts, worlds):
    searched = counts.get('searched', 0)
    attempts = searched*worlds+counts.get('resamples', 0)
    return dict(fallback=counts.get('fallback_no_candidates', 0)/max(searched, 1),
                rejection=counts.get('invalid_samples', 0)/max(attempts, 1),
                exhausted=counts.get('exhausted_worlds', 0)/max(searched*worlds, 1),
                invalid_loads=counts.get('invalid_loads', 0), real_mismatches=counts.get('real_mismatches', 0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    args = p.parse_args()
    data = json.loads((args.data_dir/'summary.json').read_text())
    metrics = [json.loads(line) for line in (args.out_dir/'train/metrics.jsonl').read_text().splitlines()]
    lines = ['# Search distillation session', '', '## Data', '', '```json', json.dumps(data, indent=2), '```',
             '', '## Final metrics (best checkpoint selected by validation policy loss)', '', '```json',
             json.dumps(dict(initial=metrics[0], final=metrics[-1], best=min(metrics, key=lambda m: m['val']['policy'])), indent=2),
             '```', '', '| Evaluation | Games | Win rate (draw=½) | Wilson 95% |', '|---|---:|---:|---|']
    reports = {}
    for name, title in (('policy', 'Distilled policy vs original'), ('distilled-search', 'Distilled + search vs original'),
                        ('reference-search', 'Original + search vs original')):
        path = args.out_dir/(name+'.json')
        if not path.exists():
            lines.append(f'| {title} | skipped | — | — |')
            continue
        report = json.loads(path.read_text())
        r = report['results'][0] if name == 'policy' else report
        lo, hi = r['wilson_interval']
        lines.append(f"| {title} | {r['games']} | {r['win_rate']:.2%} | {lo:.2%}–{hi:.2%} |")
        reports[name] = report
    rates = dict(both_seats=search_rates(data['search_counts'], data['provenance']['search_config']['worlds']))
    for name in ('reference-search', 'distilled-search'):
        if name in reports:
            r = reports[name]
            rates[name] = search_rates(r['search']['counts'], r['config']['worlds'])
    lines += ['', '## Both-seat vs single-seat search diagnostics', '', '```json', json.dumps(rates, indent=2), '```',
              '', 'Compare rates empirically; small CPU smoke games have wide uncertainty. '
              'Candidate-set validation does not measure actions outside retained search candidates.']
    (args.out_dir/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[-18:]))


if __name__ == '__main__':
    main()
