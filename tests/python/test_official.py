"""Offline wire/native conformance and structural privacy tests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gfrl import _core
from gfrl.model import OBS_FIELDS
from gfrl.official import OfficialSession, OfficialDecision, project_game, read_games, DTYPES
from gfrl.official_advisor import advise

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'third_party/goodfield/data/official-v2'


def fixtures():
    for path in sorted(DATA.glob('wire-game-updates-*.json')):
        data = json.loads(path.read_text())
        yield path, data['roles']['viewer_player_id'], data['updates']


@pytest.fixture(scope='module')
def native():
    exe = ROOT / 'build/gfrl_official_native_dump'
    assert exe.exists(), 'Build gfrl_official_native_dump before pytest'
    run = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    return [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]


def signature(view):
    b = view.get('board', view)
    players = []
    for p in b['players']:
        s = deepcopy(p['status'])
        if s:
            s['guardian'] = s['guardian'] or 'none'
            s['curses'].sort()
        players.append((p['id'], p['dead'], p['fogged'], s, p.get('used', p.get('used_models'))))
    hand = sorted((h['face'], h['used'], h['disguised']) for h in b['hand'])
    return json.dumps([b['gf'], players, hand], sort_keys=True)


def token_arrays(n):
    return {key: np.asarray([t[col] for t in n['tokens'][:55]]) for key, col in
            [('obs_type', 0), ('obs_aux_card', 1), ('obs_cards', 2), ('obs_cat', 3), ('obs_num', 4)]}


def align_hand(p, native):
    """Native reconstructed scenes may use another physical inventory order."""
    remaining = list(p['hand']); result = deepcopy(p); result['hand'] = []
    for token in native['tokens'][3:21]:
        if token[0] == 0:
            continue
        code = token[2][0]
        index = next(i for i, h in enumerate(remaining) if
                     (h['face'], h['disguised'], h['used']) == (code, bool(token[4][24]), bool(token[4][25])))
        result['hand'].append(remaining.pop(index))
    assert not remaining
    return result


def test_wire_native_board_and_masks(native):
    index = {}
    for n in native:
        index.setdefault((n['viewer'], signature(n['view'])), []).append(n)
    total = matched = ordered = mask_positions = 0
    absent = []; reordered = []
    for path, viewer, updates in fixtures():
        session = OfficialSession(viewer)
        for update in updates:
            total += 1
            r = session.ingest(update['game']); p = r['public']
            candidates = index.get((viewer, signature(_core._official_board(p))), [])
            if not candidates:
                absent.append((path.name, p['update'])); continue
            # Restrict to the same public phase and actor, then compare every
            # current-board feature. Native external receipt decisions are not
            # wire updateCount, so its decision counter is supplied explicitly.
            candidates = [n for n in candidates if n['view']['actor'] == p['actor'] and
                          n['view']['phase'] == ('main', 'response', 'finished', 'purchase', 'grant')[p['phase']]]
            assert candidates, (path.name, p['update'])
            errors = []
            for n in candidates:
                pp = deepcopy(p); pp['decisions'] = n['view']['decisions']
                aligned = align_hand(pp, n)
                got = _core._official_observation(aligned, [], 0)
                want = token_arrays(n)
                if all(np.array_equal(np.asarray(got[k], dtype=DTYPES[k]), np.asarray(a, dtype=DTYPES[k])) for k, a in want.items()):
                    break
                errors.append((n, got, want))
            else:
                pytest.fail(f'{path.name}:{p["update"]} board differs from all native replay candidates')
            matched += 1
            original = _core._official_observation(pp, [], 0)
            if all(np.array_equal(np.asarray(original[k], dtype=DTYPES[k]), np.asarray(a, dtype=DTYPES[k])) for k, a in want.items()):
                ordered += 1
            else:
                reordered.append((path.name, p['update']))
            assert np.array_equal(got['obs_valid'], np.asarray(got['obs_type']) != 0)
            if p['actor'] == viewer and p['phase'] != 2:
                env = OfficialDecision(aligned, [], 0)
                for prefix in n['prefixes']:
                    assert int(env.stage[0]) == prefix['stage']
                    np.testing.assert_array_equal(env.mask[0], prefix['mask'])
                    env.step(np.array([prefix['choice']], np.int32))
                assert env.complete
                mask_positions += 1
    assert (total, matched, ordered, mask_positions) == (79, 76, 74, 72)
    assert set(reordered) == {('wire-game-updates-v64.json', 4), ('wire-game-updates-v91.json', 2)}
    assert set(absent) == {('wire-game-updates-v82-hell.json', 1),
                           ('wire-game-updates-v93.json', 2), ('wire-game-updates-v94.json', 3)}


class FirstLegal:
    def __call__(self, env, rows):
        return env.mask.argmax(1).astype(np.int32)


def test_every_wire_ignores_hidden_inventory_tokens_counts_and_identity():
    decisions = 0
    for path, viewer, updates in fixtures():
        a, b = OfficialSession(viewer), OfficialSession(viewer)
        for update in updates:
            original = update['game']; changed = deepcopy(original)
            for player in changed['players']:
                player['name'] = 'DO NOT EXPOSE'; player['userId'] = 'SECRET'
                if player['id'] != viewer:
                    # Keep publicly used faces, discard every unused true copy,
                    # then invent a completely different count, IDs and values.
                    player['items'] = [i for i in player.get('items', []) if i.get('used')]
                    player['items'] += [dict(id=99999+j, modelId=209 if j % 2 else 207,
                                             fakeModelId=27, used=False) for j in range(17)]
                else:
                    for item in player['items']:
                        if item.get('fakeModelId'):
                            item['modelId'] = 296  # own hidden actual value too
            assert project_game(original, viewer) == project_game(changed, viewer)
            ra, rb = a.ingest(original), b.ingest(changed)
            assert ra['status'] == rb['status']
            assert ra['unsupported'] == rb['unsupported']
            assert a.events == b.events
            if ra['status'] == 'decision':
                decisions += 1
                for key in OBS_FIELDS + ('mask',):
                    np.testing.assert_array_equal(getattr(ra['env'], key), getattr(rb['env'], key))
                assert advise(ra, FirstLegal()) == advise(rb, FirstLegal())
    assert decisions == 75


def test_fog_discards_hidden_status_and_true_used_models():
    _, viewer, updates = next(fixtures())
    g = deepcopy(updates[0]['game'])
    own = next(x for x in g['players'] if x['id'] == viewer)
    own['curses'] = ['fog']
    a = project_game(g, viewer)
    other = next(x for x in g['players'] if x['id'] != viewer)
    other.update(hp=99, mp=0, cp=1, guardian='moon', curses=['flash'], confusionCount=100)
    assert project_game(g, viewer) == a
    assert next(x for x in a['players'] if x['id'] != viewer)['status'] is None


def test_duplicates_gaps_invalids_and_midgame():
    path, viewer, updates = next(fixtures())
    s = OfficialSession(viewer)
    first = s.ingest(updates[0]['game'])
    assert s.ingest(updates[0]['game'])['status'] == 'duplicate'
    assert s.ingest(dict(updates[0]['game'], gf=999))['status'] == 'unsupported'
    mid = OfficialSession(viewer).ingest(updates[-1]['game'])
    assert not mid['history_complete']
    bad = deepcopy(updates[0]['game']); bad['events'].append(dict(action='SECRET-UNKNOWN'))
    r = OfficialSession(viewer).ingest(bad)
    assert r['status'] == 'unsupported' and 'SECRET' not in str(r)
    bad = deepcopy(updates[0]['game']);bad['players'] *= 2
    assert OfficialSession(viewer).ingest(bad)['status'] == 'unsupported'


def test_prefix_and_jsonl_inputs(tmp_path):
    _, viewer, updates = next(fixtures())
    games = [u['game'] for u in updates[:2]]
    path = tmp_path / 'decoded.jsonl'
    path.write_text('\n'.join(json.dumps(g) for g in games))
    assert list(read_games(path, viewer)) == games
    def value(x):
        if isinstance(x, dict):return {'mapValue': {'fields': {k: value(v) for k,v in x.items()}}}
        if isinstance(x, list):return {'arrayValue': {'values': list(map(value,x))}}
        if type(x) is bool:return {'booleanValue': x}
        if type(x) is int:return {'integerValue': str(x)}
        if x is None:return {'nullValue': None}
        return {'stringValue': x}
    frame=json.dumps([{'documentChange': {'document': {'name': 'private', 'fields': {'game': value(games[0])}}}}])
    path.write_bytes(f"{len(frame.encode('utf-16-le'))//2}\n{frame}".encode()+b'100\n{')
    assert list(read_games(path,viewer)) == games[:1]


def test_synthetic_purchase_and_response_read_only():
    _, viewer, updates = next(fixtures())
    p=project_game(updates[0]['game'],viewer);p['decisions']=0
    other=next(x['id'] for x in p['players'] if x['id']!=viewer)
    p.update(actor=viewer,phase=3,pending=None,
             purchase=dict(origin=viewer,buyer=viewer,seller=other,model=23))
    env=OfficialDecision(p)
    assert np.flatnonzero(env.mask[0]).tolist()==[7,8]
    env.step(np.array([8],np.int32));assert env.complete
    action=env.semantic_action()
    assert action['kind']=='decline' and action['operations'][-1]==dict(op='confirm')


@pytest.mark.parametrize('version',['v73','v87'])
def test_start_to_end_history_against_native_accepted_records(native,version):
    data=json.loads((DATA/f'wire-game-updates-{version}.json').read_text())
    session=OfficialSession(1)
    for u in data['updates']:
        r=session.ingest(u['game']);p=r['public']
        assert r['history_complete']
        expected=[n for n in native if n['viewer']==1 and signature(n['view'])==signature(_core._official_board(p))
                  and n['policy_decisions']==p['decisions']]
        assert expected
        got=_core._official_observation(p,session.events,64)
        for n in expected:
            if all(np.array_equal(np.asarray(got[key],dtype=DTYPES[key])[55:],np.asarray([t[col] for t in n['tokens'][55:]],dtype=DTYPES[key]))
                   for key,col in [('obs_type',0),('obs_aux_card',1),('obs_cards',2),('obs_cat',3),('obs_num',4)]):
                break
        else:
            pytest.fail(f'{version} update{p["update"]} public history differs')


def test_unsupported_execution_and_missing_pending_metadata():
    _,viewer,updates=next(fixtures())
    g=deepcopy(updates[0]['game'])
    own=next(p for p in g['players'] if p['id']==viewer)
    own['items']=[dict(id=7,modelId=94),dict(id=19,modelId=232)]
    p=project_game(g,viewer);p['decisions']=0;p.update(actor=viewer,phase=0,pending=None)
    env=OfficialDecision(p)
    while not env.complete:
        if env.stage[0]==0:c=0
        elif env.stage[0]==1:
            c=next((i for i in (0,1) if env.mask[0,i]),18)
        else:c=int(np.flatnonzero(env.mask[0])[0])
        env.step(np.array([c],np.int32))
    assert env.action_support()==['danger_wide_requires_unobservable_mortar_absence']
    # A reversed active head without its observed launch must never invent an
    # origin or reflection count from the current source alone.
    g['attacks']=[dict(playerId=viewer,targetPlayerId=next(p['id'] for p in g['players'] if p['id']!=viewer),
                       itemModelIds=[184],atk=22,reversed=True)]
    r=OfficialSession(viewer).ingest(g)
    assert 'redirect_requires_history' in r['unsupported']


def test_ema_actor_and_main_search_from_public_root():
    import torch
    from gfrl.model import Model, ModelConfig
    from gfrl.actor import Actor
    from gfrl.search import SearchAgent, SearchConfig, load_search_belief
    torch.set_num_threads(1);torch.manual_seed(19)
    data=json.loads((DATA/'wire-game-updates-v73.json').read_text())
    result=OfficialSession(1).ingest(data['updates'][0]['game'])
    model=Model(ModelConfig(d=32,depth=1,heads=2,ff=64)).eval()
    root=_core._official_search_root(result['public'],[],64)
    for field in OBS_FIELDS:
        np.testing.assert_array_equal(getattr(root,field),getattr(result['env'],field))
    actor=Actor(model,result['env'],'cpu',sample=False)
    plain=advise(result,actor)
    assert plain['status']=='advice'
    result=OfficialSession(1).ingest(data['updates'][0]['game'])
    search=SearchAgent(model,load_search_belief('uniform',None,'cpu'),'cpu',
                       config=SearchConfig(worlds=2,candidates=2,depth=1,select='argmax'),seed=0,num_threads=1)
    searched=advise(result,search,search=True)
    assert searched['search_status']=='main_public_worlds'
    assert searched['action']['operations'][-1]==dict(op='confirm')
