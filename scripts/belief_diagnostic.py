#!/usr/bin/env python3
"""Offline fair diagnostic of search's hidden unused-hand sample distributions.

Example: .venv/bin/python scripts/belief_diagnostic.py --device mps
--fit accepts make_belief_data.py labels-only data (retained-hand labels), just
like search's B1. Without it, an independent seed+1 self-play fitting set is made.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl import _core
from gfrl.belief_data import make_belief_data
from gfrl.belief_diagnostic import public_used, true_unused, sample_metrics, summarize
from gfrl.belief_metrics import fit_baselines
from gfrl.evaluation import load_policy_model, _policy_rng
from gfrl.runtime import configure_precision
from gfrl.search import SearchAgent, SearchConfig, load_search_belief
from gfrl.search_belief import HybridBelief

BUCKETS = (0, 5, 10, 20, 30, 45, float('inf'))
# Category enum order used by the existing card-feature encoder.
CATEGORY_NAMES = ('trade', 'weapons', 'armor', 'sundries', 'miracles', 'devils', 'guardians', 'phenomena')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', default='runs/play/m-2166.pt')
    p.add_argument('--belief', default='runs/play/belief-ar-m.pt')
    p.add_argument('--fit', type=Path)
    p.add_argument('--fit-positions', type=int, default=50000)
    p.add_argument('--positions', type=int, default=5000)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--sample-prob', type=float, default=.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--max-attempts', type=int, default=100)
    p.add_argument('--max-inference-batch', type=int, default=256)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--out', type=Path, default=Path('runs/belief-diagnostic.json'))
    args = p.parse_args()
    if min(args.positions, args.fit_positions, args.samples, args.bootstrap, args.threads) <= 0:
        p.error('positions, fit-positions, samples, bootstrap and threads must be positive')
    torch.set_num_threads(args.threads)
    configure_precision(args.device, True)
    started = perf_counter()
    last_progress = started
    kwargs = dict(device=args.device, pool_fraction=0, sample_prob=args.sample_prob,
                  num_envs=args.num_envs, threads=args.threads)
    print(f'B1 fit: {args.fit or f"separate self-play, {args.fit_positions} positions, seed={args.seed+1}"}', flush=True)
    fit_count = 0
    def fit_progress(env, rows, saved):
        nonlocal fit_count, last_progress
        fit_count += len(rows)
        if perf_counter()-last_progress >= 10 or fit_count == args.fit_positions:
            print(f'fit {fit_count}/{args.fit_positions}', flush=True)
            last_progress = perf_counter()
    fit = (torch.load(args.fit, map_location='cpu', weights_only=False) if args.fit else
           make_belief_data(args.policy, positions=args.fit_positions, labels_only=True,
                            seed=args.seed+1, on_record=fit_progress, **kwargs))
    baseline = fit_baselines(fit)['B1']
    fit_config = fit.get('config')
    fit_seconds = perf_counter()-started
    del fit
    model = load_policy_model(args.policy, args.device)
    ar = load_search_belief(args.belief, device=args.device)
    config = SearchConfig(worlds=args.samples, max_attempts=args.max_attempts,
                          max_inference_batch=args.max_inference_batch)
    agents = {name: SearchAgent(model, belief, args.device, config=config, seed=args.seed+10+i)
              for i, (name, belief) in enumerate((('ar', ar), ('hybrid', HybridBelief(ar, baseline)), ('b1', baseline)))}
    feature = _core.card_features()[:, :8]
    categories = {name: np.flatnonzero(feature[:, i]) for i, name in enumerate(CATEGORY_NAMES) if feature[:, i].any()}
    names = (*agents, 'uninformed')
    values = {name: {group: [] for group in ('all', *categories)} for name in names}
    acceptance = {name: [] for name in names}
    truths, sizes = [], []
    timing = Counter()
    reference_rng = np.random.default_rng(args.seed+20)
    recorded = 0

    @torch.inference_mode()
    def record(env, rows, saved):
        nonlocal recorded, last_progress
        batch_truth = [true_unused(env, int(row)) for row in rows]
        truths.extend(batch_truth)
        sizes.extend(int(t.sum()) for t in batch_truth)
        for name in names:
            start = perf_counter()
            # Diagnostic draws must not change the self-play policy RNG stream.
            with _policy_rng(args.seed+100+recorded, list(agents.values())):
                if name in agents:
                    agent = agents[name]
                    before = agent.counts.copy()
                    rr, assignments, _ = agent._sample_worlds(env, rows)
                    samples_by_row = {int(row): [] for row in rows}
                    for row, assignment in zip(rr, assignments):
                        samples_by_row[int(row)].append(env.unused_assignment(int(row), assignment))
                    # Per-position rejection accounting by a one-root invocation
                    # would change batching. Batch totals are retained explicitly.
                    rejected = agent.counts['invalid_samples']-before['invalid_samples']
                    timing[name+'_rejected'] += rejected
                    timing[name+'_accepted'] += len(rr)
                else:
                    samples_by_row = {}
                    for row, truth in zip(rows, batch_truth):
                        accepted = []
                        used = public_used(env, int(row))
                        for _ in range(args.samples):
                            for attempt in range(args.max_attempts):
                                assignment = reference_rng.multinomial(int(truth.sum()), baseline.card_probs.numpy()).astype(np.int32)+used
                                try:
                                    unused = env.unused_assignment(int(row), assignment)
                                except ValueError:
                                    timing[name+'_rejected'] += 1
                                    continue
                                accepted.append(unused)
                                break
                        samples_by_row[int(row)] = accepted
                        timing[name+'_accepted'] += len(accepted)
                for row, truth in zip(rows, batch_truth):
                    samples = samples_by_row[int(row)]
                    acceptance[name].append(len(samples))
                    for group, cards in [('all', None), *categories.items()]:
                        values[name][group].append(sample_metrics(samples, truth, cards) if samples else np.full(37, np.nan))
            timing[name+'_seconds'] += perf_counter()-start
        recorded += len(rows)
        if perf_counter()-last_progress >= 10 or recorded == args.positions:
            print(f'diagnostic {recorded}/{args.positions}; elapsed {perf_counter()-started:.1f}s', flush=True)
            last_progress = perf_counter()

    data_start = perf_counter()
    data = make_belief_data(args.policy, positions=args.positions, seed=args.seed,
                            on_record=record, finish_games=True, **kwargs)
    collection_seconds = perf_counter()-data_start
    data['true_unused'] = torch.from_numpy(np.asarray(truths))
    data['true_unused_size'] = torch.tensor(sizes, dtype=torch.int32)
    data['diagnostic_accepted_samples'] = {name: torch.tensor(v) for name, v in acceptance.items()}
    data['diagnostic_scores'] = {name: {group: torch.from_numpy(np.asarray(v))
                                      for group, v in groups.items()} for name, groups in values.items()}
    data['diagnostic_score_columns'] = [
        'size_accuracy', 'size_mae', 'presence_brier', 'presence_log_loss',
        'active_brier', 'active_log_loss', 'copies_mse',
        *[f'ece_bin_count_{i}' for i in range(10)],
        *[f'ece_bin_prediction_sum_{i}' for i in range(10)],
        *[f'ece_bin_label_sum_{i}' for i in range(10)]]
    data_path = args.out.with_suffix('.positions.pt')
    data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, data_path)
    games = data['game_id'].numpy()
    remaining = data['decisions_to_end'].numpy()
    result = {}
    for name in names:
        rejected, accepted = int(timing[name+'_rejected']), int(timing[name+'_accepted'])
        good = np.asarray(acceptance[name]) > 0
        def summary(group, mask):
            mask = mask & good
            return summarize(np.asarray(values[name][group])[mask], games[mask], args.bootstrap, args.seed)
        result[name] = dict(accepted=accepted, rejected=rejected,
                           rejection_rate=rejected/(accepted+rejected) if accepted+rejected else None,
                           exhausted_slots=args.positions*args.samples-accepted,
                           zero_sample_positions=int((~good).sum()),
                           overall=summary('all', np.ones(len(games), bool)),
                           decisions_to_end={f'[{lo},{hi})': summary('all', (remaining >= lo) & (remaining < hi))
                                             for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:])},
                           categories={group: summary(group, np.ones(len(games), bool)) for group in categories})
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  fit=dict(source=str(args.fit) if args.fit else 'separate same-policy labels-only self-play', config=fit_config),
                  definitions=dict(target='unused_assignment accepted samples; true label minus public used',
                                   b1='retained-label fit and search assignment conversion',
                                   uninformed='B1 contents with privileged TRUE unused size, plus public used before validation',
                                   smoothing='log loss only: (presence hits + 0.5)/(accepted samples + 1)',
                                   aggregation='position means; pooled card-position ECE; 95% whole-game percentile bootstrap',
                                   active='true presence > 0 OR unsmoothed sampled presence > 0',
                                   decisions_to_end='terminal engine decision count minus zero-based current decision index (includes current)',
                                   missing='zero-accepted positions excluded; report counts; partial samples use actual denominator',
                                   categories='engine category one-hot columns; padding excluded'),
                  positions_file=str(data_path),
                  positions=dict(game_id=games.tolist(), decision=data['decision'].tolist(),
                                 decisions_to_end=remaining.tolist(), true_unused_size=sizes,
                                 true_unused= [t.tolist() for t in truths], accepted_samples=acceptance),
                  timings=dict(fit_seconds=fit_seconds, collection_and_sampling_seconds=collection_seconds, **timing), results=result)
    lines = ['# Unused-hand belief diagnostic', '',
             f'Policy: `{args.policy}`; device: {args.device}; {args.positions} positions; {args.samples} requested samples.',
             f'B1 fit: {report["fit"]["source"]}. All metrics use accepted unused hands; log loss is in nats.', '',
             '| Split | Belief | N | Size acc | Size MAE | Brier | Log loss | Active Brier | Active log loss | Copies MSE | ECE |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    rejection_lines = []
    for name, r in result.items():
        for split, s in [('all', r['overall']), *r['decisions_to_end'].items(), *r['categories'].items()]:
            cells = [f'{m["mean"]:.4f} [{m["ci95"][0]:.4f}, {m["ci95"][1]:.4f}]' for m in (s['metrics'] or {}).values()]
            lines.append(f'| {split} | {name} | {s["positions"]} | '+ ' | '.join(cells or ['—']*8)+' |')
        rejection_lines.append(f'{name}: rejection {r["rejection_rate"]:.2%}; exhausted slots {r["exhausted_slots"]}; zero-sample positions {r["zero_sample_positions"]}.')
    lines += ['', *rejection_lines, '', '95% CIs resample whole games; all beliefs use the same positions.',
              'Brier and ECE use raw frequencies; only log loss uses add-0.5 smoothing. Active support is selected before smoothing.',
              'Uninformed knows the true unused count; it is a privileged reference. No privileged labels enter policy inference.',
              f'Timings (seconds): `{json.dumps(report["timings"])}`', '']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    args.out.with_suffix('.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:8] + [line for line in lines[8:] if line.startswith('| all |')]), flush=True)
    print(f'Wrote {args.out} and {args.out.with_suffix(".md")}; total {perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
