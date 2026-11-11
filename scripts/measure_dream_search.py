#!/usr/bin/env python3
"""Offline fixed-policy coverage and official Dream public-root reproduction."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import torch
from gfrl import VecEnv, _core
from gfrl.actor import Actor
from gfrl.evaluation import _policy_rng
from gfrl.model import Model, ModelConfig
from gfrl.official import OfficialSession
from gfrl.search import SearchAgent, SearchConfig, load_search_belief, search_support


def dream_games(response, count, singleton=True, changed=False):
    path = Path(__file__).resolve().parents[1] / 'third_party/goodfield/data/official-v2/wire-game-updates-v73.json'
    first = json.loads(path.read_text())['updates'][0]['game']
    faces = [189, 58, 20, 91, 195, 63] if singleton else [6, 7, 55, 60, 130, 138]
    first['players'][0].update(hp=22, mp=15, cp=40, curses=['dream'])
    first['players'][1].update(hp=37, mp=30, cp=0, curses=[])
    first['players'][0]['items'] = [dict(id=990+i, modelId=(6 if changed else f), fakeModelId=f)
                                   for i, f in enumerate(faces[:count])]
    first['players'][0]['items'] += [dict(id=980+i, modelId=f) for i, f in enumerate([3, 4, 190])]
    if changed:
        first['players'][1]['items'] = [dict(id=800+i, modelId=f) for i, f in enumerate([94, 232, 209])]
    if not response:
        return [first]
    first['attackTurnPlayerId'] = 2
    first['events'] = [dict(action='startGame'), dict(action='advanceGF', playerId=2)]
    pending = deepcopy(first)
    pending.update(updateCount=2, attacks=[dict(playerId=2, targetPlayerId=1, itemModelIds=[19])],
                   events=[dict(action='useAttackItems', items=[dict(id=900, modelId=19)]),
                           dict(action='setTargetPlayer', playerId=1)])
    return [first, pending]


def official_result(response, count, singleton=True, changed=False):
    session = OfficialSession(1)
    for game in dream_games(response, count, singleton, changed):
        result = session.ingest(deepcopy(game))
    assert result['status'] == 'decision', result['status']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=300)
    parser.add_argument('--belief', default='uniform')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.games < 0 or args.games % 2:
        parser.error('games must be nonnegative and even')
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = Model(ModelConfig(d=16, heads=2, depth=1, ff=32)).eval()
    config = SearchConfig(worlds=4, candidates=4, candidate_samples=8, depth=1,
                          search_response=True, select='argmax', skip_single_candidate=True)
    belief = load_search_belief(args.belief)
    cases = []
    start = perf_counter()
    for singleton in (False, True):
        for response in (False, True):
            for count in range(2, 7):
                result = official_result(response, count, singleton)
                env = result['env']
                root = _core._official_search_root(result['public'], env.events, 64, env.restrictions)
                agent = SearchAgent(model, belief, config=config, seed=17)
                with _policy_rng(31, (agent,)):
                    selected = agent.search(root, [0])
                cases.append(dict(singleton_faces=singleton, response=response, disguised=count,
                    supported=int(root.determinize_support(np.array([0], np.int32))[0]) == 0,
                    selected=bool(selected), candidate_diagnostics=agent.last_candidate_diagnostics,
                    loads=dict(agent.world_load_counts), counts=dict(agent.counts)))
    totals = Counter()
    trajectory = hashlib.sha256()
    agent = SearchAgent(model, belief, config=config, seed=778)
    if args.games:
        env = VecEnv(16, 1, eval_pairs=args.games//2, opponent='external', eval_seed=20261111,
                     track_projection=True)
        actor = Actor(model, env, 'cpu')
        last = perf_counter()
        # Search has its own RNG and isolated Torch stream; only actor selects real actions.
        with _policy_rng(101, (actor,)):
            while not env.eval_done():
                rows = np.flatnonzero(env.stage >= 0).astype(np.int32)
                fresh = rows[env.substep[rows] == 0]
                roots = fresh[search_support(env, fresh, True) == 0]
                if len(roots):
                    with _policy_rng(203, (agent,)):
                        selected = agent.search(env, roots)
                    for r in roots:
                        view = env.play_view(int(r), int(env.actor_seat[r]))
                        own = next(p for p in view['players'] if p['id'] == view['actor'])
                        dream = 'dream' in own['status']['curses']
                        for group in ('overall', 'dream' if dream else 'non_dream'):
                            totals[group+'_supported'] += 1
                            totals[group+'_zero_candidates'] += int(int(r) not in selected)
                choices = actor(env, rows)
                trajectory.update(choices.tobytes())
                env.step(choices)
                if perf_counter()-last > 30:
                    print(f'{len(env.eval_results())}/{args.games} games: {dict(totals)}', flush=True)
                    last = perf_counter()
        totals['completed_games'] = len(env.eval_results())
    report = dict(config=asdict(config), belief=args.belief, seed=20261111,
                  policy='fixed seeded tiny model; both seats; real actions never use search',
                  cases=cases, trajectory_sha256=trajectory.hexdigest(), selfplay=dict(totals), loads=dict(agent.world_load_counts),
                  counts=dict(agent.counts), elapsed_seconds=perf_counter()-start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(selfplay=report['selfplay'], cases=len(cases),
                         zero_cases=sum(not c['selected'] for c in cases), seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
