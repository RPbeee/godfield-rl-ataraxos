#!/usr/bin/env python3
"""Offline P4: paired A policy, B AR search, C B1 search, optional D oracle."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import torch
import eval_search as evaluation
from sweep_search import atomic_json, input_identity, TIMING_DEFINITION
from gfrl import VecEnv
from gfrl.actor import Actor
from gfrl.belief import Belief
from gfrl.evaluation import evaluate_policies, load_policy_model
from gfrl.runtime import configure_precision
from gfrl.search_belief import variant_options
from gfrl.search import SearchAgent, load_search_belief


def run_pair(args, left, right):
    config = evaluation.search_config(args)
    model = load_policy_model(args.checkpoint, args.device)
    agents, records = {}, {}
    for name in (left, right):
        spec = dict(A=None, B=args.belief, C='b1', D='oracle')[name]
        if spec is None:
            agent = Actor(model, VecEnv(1, 1, eval_pairs=0), args.device)
        else:
            belief = load_search_belief(spec, args.fit, args.device, **(variant_options(args) if name == 'B' else {}))
            from gfrl.search_belief import MixedBelief, TemperedBelief
            if name == 'B' and not isinstance(belief, (Belief, MixedBelief, TemperedBelief)):
                raise ValueError('--belief must be an autoregressive belief checkpoint')
            agent_class = SearchAgent
            if getattr(args, 'diagnostic_true_hand_nll', False):
                from search_diagnostics import DiagnosticSearchAgent
                agent_class = DiagnosticSearchAgent
            agent = agent_class(model, belief, args.device, config=config,
                                seed=args.seed, num_threads=args.threads, profile=args.profile)
            if getattr(args, 'diagnostic_true_hand_nll', False):
                agent.enable_diagnostic(args)
        agents[name] = evaluation.TimedPolicy(agent)
        records[name] = dict(belief=spec, search_config=asdict(config) if spec else None,
                             diagnostic_only=name == 'D')
    start = perf_counter()

    def progress(summary):
        print(f'{left} vs {right}: {summary["games"]}/{2*args.pairs} games', flush=True)

    summary = evaluate_policies(agents[left], agents[right], args.pairs, args.num_envs,
                                args.threads, args.seed, track_projection=True,
                                progress=progress if args.progress_interval else None,
                                progress_interval_s=args.progress_interval)
    for name, timed in agents.items():
        if name != 'A' and getattr(args, 'diagnostic_true_hand_nll', False):
            records[name]['diagnostic_true_hand_nll'] = timed.policy.diagnostic_report()
        records[name]['decision_timing'] = timed.statistics()
        records[name]['search'] = (timed.policy.statistics() if name != 'A'
                                   else dict(searched_fraction=0))
    diagnostic = ({'diagnostic_true_hand_nll': {n: r['diagnostic_true_hand_nll'] for n, r in records.items() if 'diagnostic_true_hand_nll' in r}}
                  if getattr(args, 'diagnostic_true_hand_nll', False) else {})
    return dict(**diagnostic, match=f'{left}_vs_{right}', left=left, right=right,
                diagnostic_only='D' in agents, **summary, agents=records,
                elapsed_seconds=perf_counter()-start)


def write_summary(args, identity, reports, total):
    atomic_json(args.out_dir / 'summary.json', dict(config=identity, expected_matches=total,
                completed_matches=len(reports), timing_definition=TIMING_DEFINITION, results=reports))
    md = ('Win rates are for the left agent; draws count as ½. All matches use the same paired deals.\n\n'
          '| Match | Games | Win rate | Wilson 95% | Pair bootstrap 95% | Searched left / right | Seconds |\n'
          '|---|---:|---:|---|---|---|---:|\n')
    for r in reports:
        ci = [f'{r[k][0]:.2%}–{r[k][1]:.2%}' for k in ('wilson_interval', 'bootstrap_interval')]
        fractions = [r['agents'][n]['search']['searched_fraction'] for n in (r['left'], r['right'])]
        md += (f'| {r["match"]} | {r["games"]} | {r["win_rate"]:.2%} | {ci[0]} | {ci[1]} | '
               f'{fractions[0]:.2%} / {fractions[1]:.2%} | {r["elapsed_seconds"]:.3f} |\n')
    from search_diagnostics import diagnostic_markdown
    for report in reports:
        md += diagnostic_markdown(report)
    md += '\nD uses true opposing hands: evaluation-only cheating upper-bound diagnostic.\n\n'
    md += TIMING_DEFINITION + '\n\nConfiguration and per-agent timing/search counters:\n\n```json\n'
    md += json.dumps(dict(config=identity, agents={r['match']: r['agents'] for r in reports}), indent=2) + '\n```\n'
    path = args.out_dir / 'summary.md.tmp'
    path.write_text(md)
    path.replace(args.out_dir / 'summary.md')


def main():
    parser = evaluation.build_parser()
    parser.description = __doc__
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--with-oracle', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    evaluation.validate_args(parser, args)
    if not args.fit or args.belief in ('b1', 'uniform', 'oracle'):
        parser.error('--belief AR_CHECKPOINT and --fit B1_DATA are required')
    if args.opponent != 'policy' or args.opponent_checkpoint:
        parser.error('P4 uses the same checkpoint for every agent')
    identity = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                              if k not in ('out', 'out_dir', 'resume', 'progress_interval', 'with_oracle')},
                    inputs=input_identity(args))
    jobs = [('B', 'A'), ('C', 'A'), ('B', 'C')] + ([('D', 'A')] if args.with_oracle else [])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cached = []
        for left, right in jobs:
            path = args.out_dir / f'{left}_vs_{right}.json'
            report = None
            if args.resume and path.exists():
                try:
                    report = json.loads(path.read_text())
                except (ValueError, OSError):
                    pass
                if report is not None:
                    if report.get('config') != identity:
                        raise ValueError(f'{path}: resume configuration/input mismatch')
                    if (report.get('games') != 2*args.pairs or report.get('match') != f'{left}_vs_{right}' or
                            any(k not in report for k in ('agents', 'wilson_interval', 'bootstrap_interval', 'elapsed_seconds'))):
                        report = None
            cached.append(report)
        torch.set_num_threads(args.threads)
        configure_precision(args.device, args.tf32)
        reports = []
        for (left, right), report in zip(jobs, cached):
            path = args.out_dir / f'{left}_vs_{right}.json'
            if report is None:
                print(f'Run {left} vs {right}', flush=True)
                report = dict(config=identity, **run_pair(args, left, right))
                atomic_json(path, report)
            else:
                print(f'Skip completed {left} vs {right}', flush=True)
            reports.append(report)
            write_summary(args, identity, reports, len(jobs))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
