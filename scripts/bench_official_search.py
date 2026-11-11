#!/usr/bin/env python3
"""Offline single-move latency: fixed official fixture, no browser/network."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
from statistics import median
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import torch
from gfrl.actor import synchronize
from gfrl.evaluation import load_policy_model, _policy_rng
from gfrl.official import OfficialSession
from gfrl.official_advisor import advise
from gfrl.search import SearchAgent, SearchConfig, load_search_belief


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/play/m-2166.pt')
    p.add_argument('--belief', default='runs/play/belief-ar-m.pt')
    p.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    p.add_argument('--worlds', nargs='+', type=int, default=[8, 16, 32, 64])
    p.add_argument('--depth', nargs='+', type=int, default=[1, 3, 5])
    p.add_argument('--candidates', type=int, default=16)
    p.add_argument('--candidate-samples', type=int, default=64)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--position', choices=('fixture', 'wide', 'response'), default='fixture')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    metadata = dict(platform=platform.platform(), processor=platform.processor(), torch=torch.__version__,
                    device=args.device, position=args.position, threads=args.threads, candidates=args.candidates,
                    candidate_samples=args.candidate_samples, repeats=args.repeats,
                    mps_available=torch.backends.mps.is_available(), search_response=True, rows=[])
    if args.device == 'mps' and not metadata['mps_available']:
        metadata['unavailable'] = 'torch.backends.mps.is_available() is False'
        args.out.write_text(json.dumps(metadata, indent=2)+'\n')
        print(metadata['unavailable']); return
    torch.set_num_threads(args.threads)
    game = json.loads((Path(__file__).resolve().parents[1] /
        'third_party/goodfield/data/official-v2/wire-game-updates-v73.json').read_text())['updates'][0]['game']
    sequence = [game]
    if args.position in ('wide', 'response'):
        game['players'][0]['items'] = [dict(id=i+1, modelId=m) for i, m in enumerate(
            (19, 55, 60, 107, 130, 138, 149, 165, 176, 189, 190, 191, 198, 200, 202, 218, 3, 3))]
        # Synthetic main: diverse own public cards and resources, no private enemy inventory.
        game['players'][0].update(hp=70, mp=60, cp=30)
        game['events'] = [dict(action='startGame'), dict(action='advanceGF', playerId=1)]
    if args.position == 'response':
        game['attackTurnPlayerId'] = 2
        game['events'][-1]['playerId'] = 2
        pending = deepcopy(game)
        pending.update(updateCount=game['updateCount']+1,
            attacks=[dict(playerId=2, targetPlayerId=1, itemModelIds=[55])],
            events=[dict(action='useAttackItems', items=[dict(id=900, modelId=55)]),
                    dict(action='setTargetPlayer', playerId=1)])
        sequence.append(pending)
    model = load_policy_model(args.checkpoint, args.device)
    belief = load_search_belief(args.belief, None, args.device)
    for worlds in args.worlds:
        for depth in args.depth:
            config = SearchConfig(worlds=worlds, depth=depth, candidates=args.candidates,
                                  candidate_samples=args.candidate_samples, select='argmax', search_response=True)
            elapsed = []; reports = []
            # Fresh agent per repeat, one excluded warmup per budget. Same public root and seed.
            for repeat in range(args.repeats+1):
                session = OfficialSession(1)
                for received in sequence:
                    result = session.ingest(deepcopy(received))
                agent = SearchAgent(model, belief, args.device, config=config, seed=421, num_threads=args.threads)
                agent.actor.sample = False
                synchronize(agent.device); started = perf_counter()
                with _policy_rng(421, (agent,)):
                    advice = advise(result, agent, search=True)
                synchronize(agent.device); seconds = perf_counter()-started
                expected = 'response_public_worlds' if args.position == 'response' else 'main_public_worlds'
                if advice['search_status'] != expected or advice['search_worlds'] < 1:
                    raise RuntimeError('benchmark root was not searched')
                if repeat:
                    elapsed.append(seconds)
                    reports.append({k: advice[k] for k in ('search_status', 'search_worlds', 'search_candidates')})
            row = dict(worlds=worlds, depth=depth, seconds=elapsed, median=median(elapsed),
                       maximum=max(elapsed), diagnostics=reports)
            metadata['rows'].append(row)
            args.out.write_text(json.dumps(metadata, indent=2)+'\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
