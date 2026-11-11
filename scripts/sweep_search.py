#!/usr/bin/env python3
"""Sweep search budgets against the same checkpoint without search (offline)."""
import argparse
from itertools import product
import json
from pathlib import Path

import eval_search as evaluation


TIMING_DEFINITION = (
    'Decision time: sum of synchronized evaluated-agent call wall times / fresh '
    'decisions (substep == 0), including all action substeps and unsearched decisions. '
    'Excludes opponent, env.step, loading and reporting. Includes first-use caches; '
    'no warm-up. Batched amortized time, not single-root latency. CUDA/MPS synchronize '
    'before and after every call. Main/response search times retain SearchAgent / '
    'eval_search definitions and internal synchronization; they exclude root policy '
    'inference and unsupported decisions. The extra call-boundary synchronization '
    'can add overhead relative to eval_search CLI. Baseline uses the same wrapper, '
    'batching, seed and paired deals. --profile adds phase timing overhead.'
)


def grid_pair(value):
    try:
        worlds, depth = map(int, value.split(':'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('grid entries must be WORLDS:DEPTH (e.g. 16:2)') from exc
    return worlds, depth


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def config_identity(config, *, baseline=False):
    ignored = {'out', 'progress_interval'}
    if baseline:
        ignored.update(('worlds', 'depth'))
    return {k: str(v) if isinstance(v, Path) else v
            for k, v in config.items() if k not in ignored}


def input_identity(args):
    """Detect replaced checkpoints/priors as well as changed CLI configuration."""
    result = {}
    for key in ('checkpoint', 'opponent_checkpoint', 'belief', 'fit'):
        value = getattr(args, key)
        if not value:
            continue
        path = Path(str(value).split('@pool:')[0])
        if path.is_file():
            stat = path.stat()
            result[key] = dict(path=str(path.resolve()), size=stat.st_size,
                               mtime_ns=stat.st_mtime_ns)
    return result


def completed_report(path, args, inputs, *, baseline=False):
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text())
    except (ValueError, OSError):
        return None  # An incomplete/corrupt result is safe to recompute.
    if (config_identity(report.get('config', {}), baseline=baseline) !=
            config_identity(vars(args), baseline=baseline) or report.get('sweep_inputs') != inputs):
        raise ValueError(f'{path}: resume configuration/input mismatch; use another --out-dir or omit --resume')
    required = ('win_rate', 'wilson_interval', 'bootstrap_interval', 'decision_timing')
    if report.get('games') != 2 * args.pairs or any(k not in report for k in required):
        return None
    if not baseline and 'search' not in report:
        return None
    return report


def summary_row(report, path, *, baseline=False):
    search = report.get('search', {})
    root_times = search.get('seconds_per_search_by_root', {})
    return dict(kind='baseline' if baseline else 'search', report=path.name,
                worlds=None if baseline else report['config']['worlds'],
                depth=None if baseline else report['config']['depth'],
                **{k: report[k] for k in ('games', 'win_rate', 'wilson_interval', 'bootstrap_interval')},
                searched_fraction=search.get('searched_fraction', 0),
                seconds_per_search_main=root_times.get('main', 0),
                seconds_per_search_response=root_times.get('response', 0),
                **report['decision_timing'])


def write_summary(out_dir, rows, inputs, expected_settings):
    summary = dict(timing_definition=TIMING_DEFINITION, inputs=inputs,
                   expected_settings=expected_settings,
                   completed_settings=sum(r['kind'] == 'search' for r in rows), results=rows)
    atomic_json(out_dir / 'summary.json', summary)
    md = ('| Budget (worlds/depth) | Games | Win rate (draw=½) | Wilson 95% | Pair bootstrap 95% | '
          'Searched | s/search main | s/search response | s/decision | Decisions |\n'
          '|---|---:|---:|---|---|---:|---:|---:|---:|---:|\n')
    for row in rows:
        label = 'baseline (no search)' if row['kind'] == 'baseline' else f"{row['worlds']}/{row['depth']}"
        intervals = [f'{row[k][0]:.2%}–{row[k][1]:.2%}'
                     for k in ('wilson_interval', 'bootstrap_interval')]
        md += (f"| {label} | {row['games']} | {row['win_rate']:.2%} | {intervals[0]} | {intervals[1]} | "
               f"{row['searched_fraction']:.2%} | {row['seconds_per_search_main']:.6f} | "
               f"{row['seconds_per_search_response']:.6f} | {row['seconds_per_decision']:.6f} | "
               f"{row['decisions']} |\n")
    md += f'\n{summary["completed_settings"]}/{expected_settings} settings complete.\n\n{TIMING_DEFINITION}\n'
    if inputs.get('belief_spec') == 'oracle':
        md += '\nOracle belief is a cheating diagnostic, not a deployable win rate.\n'
    temporary = out_dir / 'summary.md.tmp'
    temporary.write_text(md)
    temporary.replace(out_dir / 'summary.md')


def main():
    parser = evaluation.build_parser(sweep=True)
    parser.description = __doc__
    parser.add_argument('--grid', nargs='+', type=grid_pair, metavar='WORLDS:DEPTH',
                        help='explicit pairs instead of the worlds × depth Cartesian product')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='reuse completed matching reports')
    args = parser.parse_args()
    if args.opponent != 'policy' or args.opponent_checkpoint not in (None, args.checkpoint):
        parser.error('sweeps require --opponent policy and the same checkpoint for both players')
    grid = list(dict.fromkeys(args.grid or product(args.worlds, args.depth)))
    settings = []
    for worlds, depth in grid:
        common = {k: v for k, v in vars(args).items()
                  if k not in ('grid', 'out_dir', 'resume')}
        common.update(worlds=worlds, depth=depth,
                      out=args.out_dir / f'worlds-{worlds}_depth-{depth}.json')
        setting = argparse.Namespace(**common)
        evaluation.validate_args(parser, setting)
        settings.append(setting)
    inputs = input_identity(args)
    inputs['belief_spec'] = args.belief
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline_args = argparse.Namespace(**vars(settings[0]))
    baseline_args.out = args.out_dir / 'baseline.json'
    jobs = [(baseline_args, True)] + [(setting, False) for setting in settings]
    try:
        # Validate all existing reports before starting expensive work.
        cached = [completed_report(setting.out, setting, inputs, baseline=baseline)
                  if args.resume else None for setting, baseline in jobs]
        for (setting, baseline), report in zip(jobs, cached):
            if report is None:
                print(f'Run {setting.out.name}', flush=True)
                report = (evaluation.run_baseline(setting) if baseline else
                          evaluation.run_evaluation(setting, measure_decisions=True))
                report['sweep_inputs'] = inputs
                atomic_json(setting.out, report)
            else:
                print(f'Skip completed {setting.out.name}', flush=True)
            rows.append(summary_row(report, setting.out, baseline=baseline))
            write_summary(args.out_dir, rows, inputs, len(settings))
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Summary: {args.out_dir / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
