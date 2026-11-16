#!/usr/bin/env python3
"""Both-seat search self-play. Default seed namespace starts at 900000.

Each completed shard consumes one eval_seed, with paired games within that seed.
Evaluation seed 74100 is never used. Resume verifies provenance before continuing.
"""
import hashlib
import json
import os
from dataclasses import asdict, replace
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


def shard_summary_part(data):
    p,co=data['probabilities'],data['candidate_offsets']
    disagreement=sum(int(p[a:b].argmax()!=data['log_pi'][a:b].argmax()) for a,b in zip(co[:-1],co[1:]))
    return (data['metadata'],len(data['game']),len(p),int(data['phase'].sum()),
            float(-(p*p.clamp_min(1e-300).log()).sum()),disagreement)


def summarize_shards(paths, elapsed, parts=None):
    games = examples = candidates = response = disagreement = 0
    entropy = seat_score = 0.
    counts, loads, seats, seconds = {}, {}, {}, {}
    if parts is None:parts=(shard_summary_part(load_shard(path)) for path in paths)
    for meta,n,nc,nr,ent,dis in parts:
        games += meta['games']
        seat_score += meta['seat0_score']
        examples += n;candidates += nc;response += nr;entropy += ent;disagreement += dis
        for dst, source in ((counts, meta['search']['counts']), (loads, meta['world_load_counts']),
                            (seats, meta['seat_counts']), (seconds, meta['search']['seconds'])):
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
                searched_positions_per_second=counts.get('searched', 0)/max(elapsed, 1e-9),
                games_per_hour=games*3600/max(elapsed, 1e-9), search_seconds=seconds,
                seed_namespace='eval_seed >= 900000 by default; one new seed per shard; paired games',
                belief_view='OBS_FIELDS of root actor_seat; no oracle belief allowed')


def generate_cuda_shard(model, belief, config, args, seed, games):
    """Only public records and final outcomes cross the dataset boundary."""
    from gfrl.gpu.search import GpuSearchAgent
    from gfrl.gpu.search_env import GpuSearchEnv
    agent = GpuSearchAgent(model, belief, args.device, config=config, seed=seed,
                           profile=args.profile, record=True, dtype=args.search_dtype,packed_record=args.cuda_record_format=='packed')
    try:
        with GpuSearchEnv(min(games,args.num_envs), games=games, eval_seed=seed, device=args.device, pooled=True) as env:
            done = torch.zeros(games, dtype=torch.bool, device=env.device)
            outcomes = torch.zeros((games,2), dtype=torch.int8, device=env.device)
            truncated = torch.zeros_like(done)
            start = last_progress = perf_counter()
            remaining = None
            with _policy_rng(seed, (agent,)):
                while not bool(done.all()):
                    before_games=env.episode_id.view(torch.int64).clone()
                    if os.environ.get('GFRL_GPU_SEARCH_ROUNDS', '0') == '1':
                        if remaining is None or not bool(remaining.any()):
                            remaining = env.stage >= 0
                        rows = torch.nonzero(remaining, as_tuple=True)[0]
                        choices = agent(env, rows)
                        choices = torch.where(remaining, choices, -1)
                        stepped=remaining.clone()
                    else:
                        stepped=env.stage>=0
                        choices = agent(env)
                    torch.cuda.synchronize()
                    since = perf_counter()
                    env.step(choices)
                    torch.cuda.synchronize()
                    agent.seconds['real_env_step'] += perf_counter()-since
                    ended = env.episode_done & stepped
                    ids=before_games[ended]
                    outcomes[ids] = env.outcome[ended]
                    truncated[ids] = env.truncated[ended]
                    done[ids]=True
                    if remaining is not None:
                        remaining &= ~env.decision_done & (env.stage >= 0)
                    now = perf_counter()
                    if args.progress_interval and now-last_progress >= args.progress_interval:
                        print(f'cuda games={int(done.sum())}/{games} examples={agent.record_count} '
                              f'elapsed={now-start:.1f}s', flush=True)
                        last_progress = now
            outcomes, truncated = outcomes.cpu().numpy(), truncated.cpu().numpy()
            if agent.packed_record:
                from gfrl.gpu.search_records import finish_outcomes
                finish_outcomes(agent.record_blocks,outcomes,truncated)
            for record in agent.records:
                g, seat = record['game'], record['seat']
                record['outcome'] = outcome_target(int(outcomes[g,seat]),seat,seat,bool(truncated[g]))
            score = float(((outcomes[:,0]+1)/2).sum())
        return agent, score
    finally:
        agent.close()


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
    parser.add_argument('--env-backend', choices=('cpu', 'cuda'), default='cpu',
                        help='CUDA simulates real games, sampled worlds and rollouts on device')
    parser.add_argument('--search-dtype', choices=('bf16','fp32'), default='bf16',
                        help='CUDA production inference; fp32 retains the CPU comparison mode')
    parser.add_argument('--cuda-record-format',choices=('packed','legacy'),default='packed',
                        help='CUDA keeps format v1 while batching retained public records')
    parser.add_argument('--shard-compression-workers',type=int,default=0,
                        help='gzip workers: 0 uses up to 8 for CUDA, 1 for CPU')
    args = parser.parse_args()
    if not 0<=args.shard_compression_workers<=args.threads:
        parser.error('shard-compression-workers must be 0..threads')
    compression_workers=args.shard_compression_workers or (min(8,args.threads) if args.env_backend=='cuda' else 1)
    config = validate_args(parser, args)
    if args.belief == 'oracle':
        parser.error('oracle belief is forbidden for distillation')
    if args.env_backend == 'cuda':
        for key in ('GFRL_GPU_SEARCH_ROUNDS','GFRL_GPU_SEARCH_BELIEF_CACHE','GFRL_GPU_SEARCH_BELIEF_KV','GFRL_GPU_SEARCH_BELIEF_GRAPH','GFRL_GPU_BELIEF_SELF'):
            os.environ.setdefault(key,'1')
        if args.device != 'cuda' or not torch.cuda.is_available():
            parser.error('--env-backend cuda requires --device cuda and a CUDA GPU')
        if args.num_envs < 2 or config.search_cache_device == 'host':
            parser.error('CUDA requires num-envs >= 2 and a device search cache')
        if getattr(args, 'belief_mix', 0) or getattr(args, 'belief_hybrid', False):
            parser.error('CUDA currently supports AR/independent/baseline beliefs without mixture/hybrid')
        if not config.root_batch_size:
            _,total_memory=torch.cuda.mem_get_info(args.device)
            root_batch=128 if total_memory>=40*1024**3 else 32
            config = replace(config, root_batch_size=root_batch)
            print(f'CUDA: effective root-batch-size={root_batch} (recorded in provenance)', flush=True)
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
    if args.env_backend == 'cuda':
        provenance['env_backend'] = 'cuda'
        provenance['search_dtype'] = args.search_dtype
        provenance['cuda_record_format']=args.cuda_record_format
        defaults=dict(GFRL_GPU_SEARCH_BELIEF_CACHE='1',GFRL_GPU_SEARCH_BELIEF_KV='1',
                      GFRL_GPU_SEARCH_BELIEF_GRAPH='1',GFRL_GPU_BELIEF_SELF='1',GFRL_GPU_SEARCH_ROUNDS='1',
                      GFRL_GPU_SEARCH_ENCODER_CHUNK='256',GFRL_GPU_SEARCH_LEAN='1',
                      GFRL_GPU_SEARCH_ARENA_BYTES='524288',GFRL_GPU_STACK_BYTES='32768')
        provenance['cuda_search']={k:os.environ.get(k,v) for k,v in defaults.items()}
    summary_parts=[] if args.env_backend=='cuda' else None
    for i, path in enumerate(paths):
        if summary_parts is None:metadata = load_shard(path)['metadata']
        else:
            previous=load_shard(path);metadata=previous['metadata']
            summary_parts.append(shard_summary_part(previous));del previous
        if metadata['provenance'] != provenance or metadata['seed'] != args.seed+i:
            parser.error('resume provenance or seed stream mismatch')
    completed = (sum(part[0]['games'] for part in summary_parts) if summary_parts is not None
                 else sum(load_shard(p)['metadata']['games'] for p in paths))
    torch.set_num_threads(args.threads)
    configure_precision(args.device, args.tf32)
    model = load_policy_model(args.checkpoint, args.device)
    belief = load_search_belief(args.belief, args.fit, args.device, **variant_options(args))
    start, new_examples = perf_counter(), 0
    initial_completed, new_searched, serialize_seconds, summary_seconds = completed, 0, 0., 0.
    last_progress = start
    while completed < args.games and (perf_counter()-start < args.max_hours*3600 or not paths):
        seed = args.seed+len(paths)
        games = min(args.games_per_shard, args.games-completed)
        if args.env_backend == 'cuda':
            agent, score = generate_cuda_shard(model, belief, config, args, seed, games)
        else:
            agent = SearchAgent(model, belief, args.device, config=config, seed=seed,
                                num_threads=args.threads, record=True, profile=args.profile)
            env = VecEnv(min(args.num_envs, games), args.threads, eval_pairs=games//2,
                         opponent='external', eval_seed=seed, track_projection=True)
            with _policy_rng(seed, (agent,)):
                while not env.eval_done():
                    rows = np.flatnonzero(env.stage >= 0)
                    since = perf_counter()
                    previous_search = agent.seconds['total']
                    choices = agent(env, rows)
                    agent.seconds['real_policy'] += perf_counter()-since-(agent.seconds['total']-previous_search)
                    since = perf_counter()
                    env.step(choices)
                    agent.seconds['real_env_step'] += perf_counter()-since
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
            del env
        metadata = dict(provenance=provenance, seed=seed, seed_range=[seed, seed+1], games=games,
                        seat0_score=score, search=agent.statistics(),
                        world_load_counts=dict(agent.world_load_counts),
                        seat_counts={str(s): (sum(int((b['seat']==s).sum()) for b in agent.record_blocks)
                            if args.env_backend=='cuda' and agent.packed_record
                            else sum(r['seat'] == s for r in agent.records)) for s in (0, 1)})
        path = args.out_dir/f'shard-{len(paths):06d}.pt.gz'
        recorded=agent.record_count if args.env_backend=='cuda' else len(agent.records)
        if not recorded:
            raise RuntimeError('batch produced no searchable roots')
        since = perf_counter()
        if args.env_backend=='cuda' and agent.packed_record:
            from gfrl.gpu.search_records import pack_blocks
            data=pack_blocks(agent.record_blocks,metadata)
        else:data=pack_records(agent.records,metadata)
        if compression_workers>1:
            from gfrl.gpu.shard import save_shard as parallel_save_shard
            parallel_save_shard(path,data,compression_workers)
        else:save_shard(path,data)
        serialize_seconds += perf_counter()-since
        if summary_parts is not None:
            since=perf_counter();summary_parts.append(shard_summary_part(data));summary_seconds+=perf_counter()-since
        del data
        paths.append(path)
        completed += games
        new_examples += recorded
        new_searched += agent.counts['searched']
        elapsed = perf_counter()-start
        rate = new_examples/max(elapsed, 1e-9)
        eta = elapsed*(args.games-completed)/max(completed, 1)
        print(f'games={completed}/{args.games} examples(new)={new_examples} examples/s={rate:.2f} '
              f'elapsed={elapsed:.1f}s ETA={eta:.1f}s bytes/example={path.stat().st_size/recorded:.1f}', flush=True)
        del agent
    since=perf_counter()
    summary = summarize_shards(paths, perf_counter()-start,summary_parts)
    summary_seconds+=perf_counter()-since
    summary['elapsed_seconds']=perf_counter()-start
    summary['summary_seconds']=summary_seconds
    summary['new_examples_per_second'] = new_examples/max(summary['elapsed_seconds'], 1e-9)
    summary['searched_positions_per_second'] = new_searched/max(summary['elapsed_seconds'], 1e-9)
    summary['games_per_hour'] = (completed-initial_completed)*3600/max(summary['elapsed_seconds'], 1e-9)
    summary['new_searched_positions'] = new_searched
    summary['serialize_seconds'] = serialize_seconds
    summary['compression_workers'] = compression_workers
    summary['provenance'] = provenance
    (args.out_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
