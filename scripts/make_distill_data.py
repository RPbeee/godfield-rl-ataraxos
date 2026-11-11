#!/usr/bin/env python3
"""Both-seat search self-play. Default seed namespace starts at 900000.

Each completed shard consumes one eval_seed, with paired games within that seed.
Evaluation seed 74100 is never used. Resume verifies provenance before continuing.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl import VecEnv
from gfrl.distill import pack_records, save_shard, load_shard, outcome_target
from gfrl.evaluation import load_policy_model, _policy_rng
from gfrl.runtime import configure_precision
from gfrl.search import SearchAgent, load_search_belief
from gfrl.search_belief import variant_options
from eval_search import build_parser, validate_args


def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize_shards(paths, elapsed):
    games = examples = candidates = response = disagreement = 0
    entropy = seat_score = 0.
    counts, loads, seats = {}, {}, {}
    for path in paths:
        d = load_shard(path)
        meta, p, co = d['metadata'], d['probabilities'], d['candidate_offsets']
        games += meta['games']
        seat_score += meta['seat0_score']
        examples += len(d['game'])
        candidates += len(p)
        response += int(d['phase'].sum())
        entropy += float(-(p*p.clamp_min(1e-300).log()).sum())
        for a, b in zip(co[:-1], co[1:]):
            disagreement += int(p[a:b].argmax() != d['log_pi'][a:b].argmax())
        for dst, source in ((counts, meta['search']['counts']), (loads, meta['world_load_counts']),
                            (seats, meta['seat_counts'])):
            for k, v in source.items():
                dst[k] = dst.get(k, 0)+v
    size = sum(p.stat().st_size for p in paths)
    return dict(games=games, examples=examples, main_examples=examples-response, response_examples=response,
                mean_candidates=candidates/max(examples, 1), mean_search_entropy=entropy/max(examples, 1),
                policy_disagreement=disagreement/max(examples, 1), seat0_win_rate=seat_score/max(games, 1),
                search_counts=counts, world_load_counts=loads, seat_counts=seats,
                fallback_rate=counts.get('fallback_no_candidates', 0)/max(counts.get('searched', 0), 1),
                rejection_rate=counts.get('invalid_samples', 0)/max(counts.get('resamples', 0)+counts.get('searched', 0)*meta['provenance']['search_config']['worlds'], 1),
                bytes_written=size, bytes_per_example=size/max(examples, 1), elapsed_seconds=elapsed,
                seed_namespace='eval_seed >= 900000 by default; one new seed per shard; paired games',
                belief_view='OBS_FIELDS of root actor_seat; no oracle belief allowed')


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(seed=900000, depth=3, beta=.1, search_response=True)
    parser.add_argument('--no-search-response', dest='search_response', action='store_false')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--games', type=int, default=1000000)
    parser.add_argument('--games-per-shard', type=int, default=512)
    parser.add_argument('--max-hours', type=float, default=2.5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = validate_args(parser, args)
    if args.belief == 'oracle':
        parser.error('oracle belief is forbidden for distillation')
    if min(args.games, args.games_per_shard) < 2 or args.games % 2 or args.games_per_shard % 2:
        parser.error('games and games-per-shard must be positive even numbers')
    if not np.isfinite(args.max_hours) or args.max_hours < 0:
        parser.error('max-hours must be finite and nonnegative')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.out_dir.glob('shard-*.pt.gz'))
    if paths and not args.resume:
        parser.error('output contains shards; use --resume')
    provenance = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_md5=file_md5(args.checkpoint),
                      belief=str(args.belief), belief_md5=file_md5(args.belief) if Path(args.belief).is_file() else None,
                      fit_md5=file_md5(args.fit) if args.fit else None,
                      belief_options=variant_options(args), search_config=asdict(config), base_seed=args.seed,
                      num_envs=args.num_envs, threads=args.threads, device=args.device, tf32=args.tf32)
    for i, path in enumerate(paths):
        metadata = load_shard(path)['metadata']
        if metadata['provenance'] != provenance or metadata['seed'] != args.seed+i:
            parser.error('resume provenance or seed stream mismatch')
    completed = sum(load_shard(p)['metadata']['games'] for p in paths)
    torch.set_num_threads(args.threads)
    configure_precision(args.device, args.tf32)
    model = load_policy_model(args.checkpoint, args.device)
    belief = load_search_belief(args.belief, args.fit, args.device, **variant_options(args))
    start, new_examples = perf_counter(), 0
    last_progress = start
    while completed < args.games and (perf_counter()-start < args.max_hours*3600 or not paths):
        seed = args.seed+len(paths)
        games = min(args.games_per_shard, args.games-completed)
        agent = SearchAgent(model, belief, args.device, config=config, seed=seed,
                            num_threads=args.threads, record=True)
        env = VecEnv(min(args.num_envs, games), args.threads, eval_pairs=games//2,
                     opponent='external', eval_seed=seed, track_projection=True)
        with _policy_rng(seed, (agent,)):
            while not env.eval_done():
                rows = np.flatnonzero(env.stage >= 0)
                env.step(agent(env, rows))
                now = perf_counter()
                if args.progress_interval and now-last_progress >= args.progress_interval:
                    done = completed+len(env.eval_results())
                    elapsed = now-start
                    count = new_examples+len(agent.records)
                    eta = elapsed*(args.games-done)/max(done, 1)
                    print(f'games={done}/{args.games} examples(new)={count} '
                          f'examples/s={count/max(elapsed, 1e-9):.2f} elapsed={elapsed:.1f}s ETA={eta:.1f}s '
                          '(in-flight batch)', flush=True)
                    last_progress = now
        results = {r['g']: r for r in env.eval_results()}
        for record in agent.records:
            result = results[record['game']]
            record['outcome'] = outcome_target(result['result'], record['seat'], result['agent_seat'], result['truncated'])
        score = sum((r['result']*(1 if r['agent_seat'] == 0 else -1)+1)/2 for r in results.values())
        metadata = dict(provenance=provenance, seed=seed, seed_range=[seed, seed+1], games=games,
                        seat0_score=score, search=agent.statistics(),
                        world_load_counts=dict(agent.world_load_counts),
                        seat_counts={str(s): sum(r['seat'] == s for r in agent.records) for s in (0, 1)})
        path = args.out_dir/f'shard-{len(paths):06d}.pt.gz'
        if not agent.records:
            raise RuntimeError('batch produced no searchable roots')
        save_shard(path, pack_records(agent.records, metadata))
        paths.append(path)
        completed += games
        new_examples += len(agent.records)
        elapsed = perf_counter()-start
        rate = new_examples/max(elapsed, 1e-9)
        eta = elapsed*(args.games-completed)/max(completed, 1)
        print(f'games={completed}/{args.games} examples(new)={new_examples} examples/s={rate:.2f} '
              f'elapsed={elapsed:.1f}s ETA={eta:.1f}s bytes/example={path.stat().st_size/len(agent.records):.1f}', flush=True)
        del agent, env
    summary = summarize_shards(paths, perf_counter()-start)
    summary['new_examples_per_second'] = new_examples/max(summary['elapsed_seconds'], 1e-9)
    summary['provenance'] = provenance
    (args.out_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
