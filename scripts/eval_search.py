#!/usr/bin/env python3
"""Paired-seed search versus policy or Greedy, with CPU/GPU timing statistics."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import torch
from gfrl import VecEnv
from gfrl.actor import Actor
from gfrl.evaluation import (evaluate_policies, load_policy_model, _policy_rng,
                             summarize, _EvaluationProgress)
from gfrl.runtime import configure_precision
from gfrl.search import SearchAgent, SearchConfig, load_search_belief


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--opponent', choices=('policy', 'greedy'), default='policy')
    p.add_argument('--opponent-checkpoint')
    p.add_argument('--belief', required=True,
                   help='checkpoint path, b1, uniform, or oracle (evaluation-only diagnostic; cheats using true opponent hands)')
    p.add_argument('--fit', help='B1 fitting data (optional count prior for uniform)')
    for name, default in (('worlds', 16), ('candidates', 16), ('candidate-samples', 64), ('depth', 10),
                          ('pairs', 500), ('num-envs', 32), ('threads', 1), ('seed', 0)):
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--search-response', action='store_true', help='search supported ordinary response roots')
    p.add_argument('--leaf-view', choices=('actor', 'searcher'), default='actor')
    p.add_argument('--max-inference-batch', type=int, default=0,
                   help='search inference row limit (0: unsplit; start with 128 on a 10 GB GPU)')
    p.add_argument('--skip-forced-inference', action=argparse.BooleanOptionalAction, default=False,
                   help='omit unused root policy inference (changes RNG order)')
    p.add_argument('--skip-single-candidate', action=argparse.BooleanOptionalAction, default=False,
                   help='validate singleton root paths, then omit rollout/value inference (changes RNG order)')
    p.add_argument('--search-cache-device', choices=('auto', 'host', 'device'), default='auto')
    p.add_argument('--search-cache-mb', type=int, default=256,
                   help='maximum device token cache MiB; larger caches use host memory')
    p.add_argument('--root-batch-size', type=int, default=0,
                   help='cap roots per search workspace; 0 disables (changes RNG order)')
    p.add_argument('--alpha', type=float, default=.002)
    p.add_argument('--beta', type=float, default=.02)
    p.add_argument('--select', choices=('sample', 'argmax'), default='sample')
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    p.add_argument('--profile', action='store_true', help='synchronized Actor phase timings (adds overhead)')
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--out', type=Path, default=Path('eval-search.json'))
    p.add_argument('--progress-interval', type=float, default=600,
                   help='progress report interval in seconds (0 disables)')
    args = p.parse_args()
    if min(args.pairs, args.num_envs, args.threads) <= 0:
        p.error('pairs, num-envs, threads must be positive')
    if not math.isfinite(args.progress_interval) or args.progress_interval < 0:
        p.error('progress-interval must be finite and nonnegative')
    try:
        config_fields = ('search_response', 'worlds', 'candidates', 'candidate_samples', 'depth', 'alpha', 'beta',
                         'select', 'leaf_view', 'max_inference_batch', 'skip_forced_inference',
                         'skip_single_candidate', 'search_cache_device', 'search_cache_mb', 'root_batch_size')
        config = SearchConfig(**{k: getattr(args, k) for k in config_fields})
        belief = load_search_belief(args.belief, args.fit, args.device)
    except ValueError as exc:
        p.error(str(exc))
    torch.set_num_threads(args.threads)
    configure_precision(args.device, args.tf32)
    agent = SearchAgent(load_policy_model(args.checkpoint, args.device), belief, args.device,
                        config=config, seed=args.seed, num_threads=args.threads, profile=args.profile)
    start = perf_counter()
    # Keep the final report's existing configuration fields and output format.
    report_config = {k: str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items() if k != 'progress_interval'}

    def progress(summary):
        elapsed = perf_counter() - start
        games, total = summary['games'], 2 * args.pairs
        remaining = elapsed * (total - games) / games if games else None
        timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
        rate = f"{summary['win_rate']:.2%}" if games else 'n/a'
        lo, hi = summary['wilson_interval']
        eta = f'{remaining:.1f}s' if remaining is not None else 'n/a'
        print(f'[{timestamp}] games={games}/{total} win_rate={rate} '
              f'Wilson95%={lo:.2%}–{hi:.2%} elapsed={elapsed:.1f}s ETA={eta}', flush=True)
        partial = dict(belief=args.belief, diagnostic_only=args.belief == 'oracle',
                       config=report_config, **summary, search=agent.statistics(),
                       timestamp=timestamp, total_games=total, elapsed_seconds=elapsed,
                       estimated_remaining_seconds=remaining)
        path = args.out.with_suffix('.partial.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace atomically so an interrupted write preserves the last report.
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(partial, indent=2) + '\n')
        temporary.replace(path)

    callback = progress if args.progress_interval else None
    if args.opponent == 'policy':
        opponent = Actor(load_policy_model(args.opponent_checkpoint or args.checkpoint, args.device),
                         VecEnv(1, 1, eval_pairs=0), args.device)
        summary = evaluate_policies(agent, opponent, args.pairs, args.num_envs, args.threads,
                                    args.seed, track_projection=True, progress=callback,
                                    progress_interval_s=args.progress_interval)
    else:
        reporter = _EvaluationProgress(callback, args.progress_interval) if callback else None
        with _policy_rng(args.seed, (agent,)):
            env = VecEnv(args.num_envs, args.threads, eval_pairs=args.pairs, opponent='greedy',
                         eval_seed=args.seed, track_projection=True)
            import numpy as np
            while not env.eval_done():
                rows = np.flatnonzero(env.stage >= 0)
                env.step(agent(env, rows))
                if reporter is not None:
                    reporter(env)
            if reporter is not None:
                reporter(env, final=True)
            summary = summarize(env.eval_results())
    report = dict(belief=args.belief, diagnostic_only=args.belief == 'oracle',
                  config=report_config,
                  **summary, search=agent.statistics(), elapsed_seconds=perf_counter()-start)
    interval = lambda v: f'{v[0]:.2%}–{v[1]:.2%}'
    md = ('| Opponent | Games | Win rate (draw=½) | Wilson 95% | Pair bootstrap 95% |\n'
          '|---|---:|---:|---|---|\n'
          f"| {args.opponent} | {summary['games']} | {summary['win_rate']:.2%} | "
          f"{interval(summary['wilson_interval'])} | {interval(summary['bootstrap_interval'])} |\n\n"
          '| Search phase | Seconds / searched decision |\n|---|---:|\n' +
          ''.join(f'| {k} | {v:.6f} |\n' for k, v in report['search']['seconds_per_search'].items()) +
          '\n```json\n'+json.dumps(report['search'], indent=2)+'\n```\n')
    print(md, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    args.out.with_suffix('.md').write_text(md)


if __name__ == '__main__':
    main()
