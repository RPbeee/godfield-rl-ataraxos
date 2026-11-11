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


@pytest.mark.parametrize('atk', [13, 47])
def test_trade_received_mask_matches_native_match(atk):
    output = subprocess.run([str(ROOT/'build/gfrl_official_native_dump'), '--trades'],
                            capture_output=True, text=True, check=True).stdout
    records = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
    responders = [n for n in records if n['viewer'] == 7]
    assert len(responders) == 2
    for n in responders:
        view = n['view']; pending = view['pending']; board = view['board']
        # Copy the native response scene into the received-game schema. The
        # operation's wire atk semantics are intentionally not asserted.
        players = []
        for p in board['players']:
            s = p['status']
            players.append(dict(id=p['id'], dead=p['dead'], hp=s['hp'], mp=s['mp'], cp=s['cp'],
                                curses=s['curses'], guardian=s['guardian'], items=[]))
        next(p for p in players if p['id']==7)['items'] = [dict(id=h['token'], modelId=h['face'], used=h['used'])
                                                        for h in board['hand']]
        game = dict(players=players, gf=board['gf'], updateCount=1, attackTurnPlayerId=view['owner'],
                    events=[], attacks=[dict(playerId=0, targetPlayerId=7, itemModelIds=pending['models'], atk=atk)])
        projected = project_game(game, 7)
        assert 'base_override' not in projected['pending']
        result = OfficialSession(7).ingest(game)
        assert result['status'] == 'decision'
        env = result['env']
        assert env.public['pending'] == projected['pending']
        for prefix in n['prefixes']:
            assert int(env.stage[0]) == prefix['stage']
            np.testing.assert_array_equal(env.mask[0], prefix['mask'])
            if prefix['stage'] == 1:
                assert not any(env.mask[0, :3])  # all three armor copies
                assert env.mask[0, 3] and env.mask[0, 18]  # 190 or empty
            env.step(np.array([prefix['choice']], np.int32))
        fresh = OfficialSession(7).ingest(game)
        advice = advise(fresh, FirstLegal())
        assert '取引 /' in advice['board'] and '威力 ' not in advice['board']


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
    assert plain['status']=='unsupported'
    assert plain['unsupported']==['ui_kind_unobserved:discard']
    from gfrl.official_advisor import advise_supported
    result=OfficialSession(1).ingest(data['updates'][0]['game'])
    supported=advise_supported(result,actor)
    assert supported['status']=='advice' and actor.sample is False
    assert supported['mask_exclusions']==[dict(reason='ui_kind_unobserved:discard',kind='discard'),
        dict(reason='self_target_of_enemy_default_unobserved',target='self')]
    result=OfficialSession(1).ingest(data['updates'][0]['game'])
    search=SearchAgent(model,load_search_belief('uniform',None,'cpu'),'cpu',
                       config=SearchConfig(worlds=2,candidates=2,depth=1,select='argmax'),seed=0,num_threads=1)
    searched=advise(result,search,search=True)
    assert searched['search_status']=='main_public_worlds'
    assert searched['action']['operations'][-1]==dict(op='confirm')


def received_scene(n, update, events):
    view = n['view']; board = view['board']; players = []
    for p in board['players']:
        s = p['status']
        used = p.get('used', p.get('used_models', []))
        items = [dict(id=100+i, modelId=m, fakeModelId=0, used=True) for i, m in enumerate(used)]
        if p['id'] == n['viewer']:
            items = [dict(id=h['token'], modelId=h['face'], fakeModelId=0, used=h['used'])
                     for h in board['hand']]
        players.append(dict(id=p['id'], hp=s['hp'], mp=s['mp'], cp=s['cp'],
                            dead=p['dead'], curses=s['curses'], guardian=s['guardian'] or None, items=items))
    attacks = []
    if view['pending']:
        a = view['pending']
        attacks = [dict(playerId=a['source'], targetPlayerId=a['target'], itemModelIds=a['models'])]
    return dict(players=players, gf=board['gf'], updateCount=update,
                attackTurnPlayerId=view['owner'], events=events, attacks=attacks)


@pytest.mark.parametrize("implicit_recipient", [False, True])
def test_received_removals_native_observation_masks_and_history(implicit_recipient):
    from gfrl.official import parser_module
    output = subprocess.run([str(ROOT/'build/gfrl_official_native_dump'), '--removals'],
                            capture_output=True, text=True, check=True).stdout
    records = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
    assert len(records) == 12
    for start, model in ((0, 205), (6, 206)):
        for viewer in (0, 7):
            initial, pending, final = [n for n in records[start:start+6] if n['viewer'] == viewer]
            session = OfficialSession(viewer)
            session.ingest(parser_module().sanitize_game(received_scene(initial, 1, [dict(action='startGame')]), viewer))
            game = received_scene(pending, 2, [dict(action='useAttackItems', items=[dict(id=0, modelId=model)]),
                                              dict(action='setTargetPlayer', playerId=7)])
            assert session.ingest(parser_module().sanitize_game(game, viewer))['status'] != 'unsupported'
            removed = final['view']['last_removal']['removed_models']
            event = dict(action='removeItems' if model == 205 else 'removeUsedMiracles', playerId=7,
                         items=[dict(id=200+i, modelId=m, fakeModelId=0, used=model==206)
                                for i, m in enumerate(removed)])
            if implicit_recipient and model == 205:
                event.pop('playerId')
            game = received_scene(final, 3, [dict(action='useDefenseItems', items=[]), event,
                                            dict(action='advanceGF', playerId=7)])
            clean = parser_module().sanitize_game(game, viewer)
            other = deepcopy(session)
            result = session.ingest(clean)
            assert result['status'] != 'unsupported', result
            assert result['history_complete']
            p = align_hand(result['public'], final)
            assert _core._official_board(p)['removal'] == dict(target=7, models=removed)
            got = _core._official_observation(p, session.events, 64)
            for key, col in [('obs_type', 0), ('obs_aux_card', 1), ('obs_cards', 2), ('obs_cat', 3), ('obs_num', 4)]:
                np.testing.assert_array_equal(np.asarray(got[key], dtype=DTYPES[key]),
                                              np.asarray([t[col] for t in final['tokens']], dtype=DTYPES[key]))
            if result['status'] == 'decision':
                env = OfficialDecision(p, session.events)
                for prefix in final['prefixes']:
                    assert int(env.stage[0]) == prefix['stage']
                    np.testing.assert_array_equal(env.mask[0], prefix['mask'])
                    env.step(np.array([prefix['choice']], np.int32))
                assert env.complete
            changed = deepcopy(clean)
            foreign = next(x for x in changed['players'] if x['id'] != viewer)
            foreign['items'] = [x for x in foreign['items'] if x['used']]
            foreign['items'] += [dict(id=1000+i, modelId=296, fakeModelId=5, used=False) for i in range(12)]
            assert project_game(clean, viewer) == project_game(changed, viewer)
            hidden_result = other.ingest(changed)
            assert result['status'] == hidden_result['status']
            assert result['public'] == hidden_result['public']
            assert session.events == other.events
            if result['status'] == 'decision':
                for key in OBS_FIELDS + ('mask',):
                    np.testing.assert_array_equal(getattr(result['env'], key), getattr(hidden_result['env'], key))


@pytest.mark.parametrize('buyer', [0, 7])
def test_received_purchase_buyer_decides_seller_waits(buyer):
    from gfrl.official import parser_module
    game = dict(players=[dict(id=i, hp=40, mp=10, cp=20, items=[]) for i in (0, 7)],
                gf=1, updateCount=1, attackTurnPlayerId=buyer, events=[],
                attacks=[dict(playerId=buyer, targetPlayerId=7-buyer, itemModelIds=[5], buyingItemModelId=23)])
    clean = parser_module().sanitize_game(game, 7)
    projected = project_game(clean, 7)
    assert projected['phase'] == 3 and projected['actor'] == buyer and projected['pending'] is None
    assert projected['purchase'] == dict(origin=buyer, buyer=buyer, seller=7-buyer, model=23)
    result = OfficialSession(7).ingest(clean)
    assert result['status'] == ('decision' if buyer == 7 else 'waiting')
    if buyer == 7:
        assert np.flatnonzero(result['env'].mask[0]).tolist() == [7, 8]
        for choice, kind in ((7, 'accept'), (8, 'decline')):
            env = OfficialSession(7).ingest(clean)['env']
            env.step(np.array([choice], np.int32))
            assert env.complete and env.semantic_action()['kind'] == kind


def test_hit_matches_miss_and_midgame_removal_is_incomplete():
    from gfrl.official import parser_module
    _, viewer, updates = next(fixtures())
    game = deepcopy(updates[0]['game'])
    other = next(x['id'] for x in game['players'] if x['id'] != viewer)
    outcomes = []
    for action in ('miss', 'hit'):
        g = deepcopy(game); g['events'].append(dict(action=action, playerId=other))
        session = OfficialSession(viewer)
        result = session.ingest(parser_module().sanitize_game(g, viewer))
        assert result['status'] == 'decision' and result['history_complete']
        outcomes.append((result, session.events))
    assert outcomes[0][1] == outcomes[1][1]
    for key in OBS_FIELDS + ('mask',):
        np.testing.assert_array_equal(getattr(outcomes[0][0]['env'], key), getattr(outcomes[1][0]['env'], key))
    game['events'] = [dict(action='removeItems', playerId=other,
                           items=[dict(id=9, modelId=23, fakeModelId=6, used=False)])]
    result = OfficialSession(viewer).ingest(parser_module().sanitize_game(game, viewer))
    assert result['status'] == 'decision' and not result['history_complete']
    assert result['public']['events'][0] == dict(action='removeItems', player=other, models=[23], faces=[6])


def test_remove_guardian_public_departure_and_history_token():
    from gfrl.official import parser_module
    game = json.loads((Path(__file__).parent/'fixtures/official_training.json').read_text())[0]
    clean = parser_module().sanitize_game(game, 1)
    public = project_game(clean, 1)
    public['players'][0]['status']['guardian'] = 'neptune'
    session = OfficialSession(1)
    first = session.ingest_public(public)
    assert first['history_complete']
    after = deepcopy(public)
    pid = after['players'][0]['id']
    after['update'] += 1
    after['events'] = [dict(action='useDefenseItems', faces=[], reveals=[]),
                       dict(action='removeGuardian', player=pid)]
    after['players'][0]['status']['guardian'] = 'none'
    result = session.ingest_public(after)
    assert result['history_complete'] and not result['public']['unsupported']
    change = dict(player=pid, before='neptune', after='none', reason='departure')
    assert session.events[-1]['after']['guardian_changes'] == [change]
    obs = _core._official_observation(result['public'], session.events, 64)
    # Native History emits a separate public GuardianChange after Decision,
    # with enum values +1 and departure reason +1, as well as the status delta.
    indices = [i for i,t in enumerate(obs['obs_type']) if t == 17]
    assert len(indices) == 1
    cat = obs['obs_cat'][indices[0]]
    assert cat[0] == 1 and cat[11] == 2 and cat[13] == 1
    assert cat[12] > 1
    follow = deepcopy(after); follow['update'] += 1
    follow['events'] = [dict(action='useDefenseItems', faces=[], reveals=[])]
    again = session.ingest_public(follow)
    obs = _core._official_observation(again['public'],session.events,64)
    assert obs['obs_type'].count(17) == 1  # retained journal is not emitted twice


def test_remove_guardian_unknown_before_state_marks_history_incomplete():
    from gfrl.official import parser_module
    game = json.loads((Path(__file__).parent/'fixtures/official_training.json').read_text())[0]
    game['events'] = [dict(action='removeGuardian',playerId=1)]
    clean = parser_module().sanitize_game(game,1)
    public = project_game(clean,1)
    assert public['events'] == [dict(action='removeGuardian',player=1)]
    result = OfficialSession(1).ingest(clean)
    assert result['status'] == 'decision' and not result['history_complete']
    assert 'unknown_event' not in result['public']['unsupported']


def test_dynamic_pending_native_values_and_missing_value_policy():
    output = subprocess.run([str(ROOT/'build/gfrl_official_native_dump'), '--dynamic'],
                            capture_output=True, text=True, check=True).stdout
    records = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
    grouped = {model: [] for model in (202,58,94)}
    for n in records:
        if n['viewer'] != 7 or n['view']['phase'] != 'response' or n['view']['actor'] != 7:
            continue
        model = n['view']['pending']['models'][0]
        grouped[model].append(n)
        game = received_scene(n, 1, [])
        result = OfficialSession(7).ingest(game)
        if model == 58:
            assert result['status'] == 'unsupported'
            assert 'dynamic_attack_value_unavailable' in result['unsupported']
        elif model == 202:
            assert result['status'] == 'decision'
            assert result['public']['pending']['power'] == 0
            assert result['public']['pending']['amount'] == -10
            assert result['public']['warnings'] == ['assumed_dynamic_value: 202']
            # Unknown sign changes amount only; armor remains unavailable.
            assert not result['env'].mask[0,0] and result['env'].mask[0,1]
            np.testing.assert_array_equal(result['env'].mask[0], n['prefixes'][0]['mask'])
            assert 'assumed_dynamic_value: 202' in advise(result, FirstLegal())['warnings']
        else:
            assert result['status'] == 'decision'
            public = result['public']; public['decisions'] = n['view']['decisions']
            got = _core._official_observation(public, [], 0)
            for key, want in token_arrays(n).items():
                np.testing.assert_allclose(got[key], want, atol=1e-7)
            env = result['env']
            for prefix in n['prefixes']:
                np.testing.assert_array_equal(env.mask[0], prefix['mask'])
                env.step(np.array([prefix['choice']],np.int32))
    for model, field, values in [(202,'amount',{-10,10}), (58,'power',{20,40}), (94,'power',{30})]:
        rows = grouped[model]
        assert {n['view']['pending'][field] for n in rows} == values
        num_col = _core.obs_schema(0)['num']['amount' if field == 'amount' else 'power_log1p']
        assert len({n['tokens'][21][4][num_col] for n in rows}) == len(values)
        assert all(n['prefixes'] for n in rows)
    signs = {n['view']['pending']['amount']: n for n in grouped[202]}
    assert signs[-10]['prefixes'] == signs[10]['prefixes']
    # Native's signed HP effect is non-attack at either sign: 130 cannot respond,
    # while 190 can, including when the sign is unavailable.
    assert not signs[-10]['prefixes'][0]['mask'][0]
    assert signs[-10]['prefixes'][0]['mask'][1]
    # Mortar launches have power 99 but never expose a response decision.
    mortar = [n for n in records if (n['view'].get('last_danger') or {}).get('mortar')]
    assert mortar and all(n['view']['phase'] != 'response' for n in mortar)
    assert all(n['view']['last_danger']['power'] == 99 for n in mortar)


@pytest.mark.parametrize('reject_count, expected_samples', [(2,2), (16,16), (100,16)])
@pytest.mark.parametrize('search', [False,True])
def test_bot_advice_fallback_samples_policy_at_same_position(monkeypatch, reject_count, expected_samples, search):
    import torch
    from gfrl.official_advisor import advise_supported
    from gfrl.evaluation import _policy_rng
    class Policy:
        sample = False
        def __call__(self, env, rows):
            legal = np.flatnonzero(env.mask[0])
            # Nonuniform policy, sampled using the existing torch policy RNG.
            weights = torch.arange(1,len(legal)+1,dtype=torch.float)
            index = int(torch.multinomial(weights,1)) if self.sample else len(legal)-1
            return np.array([legal[index]],np.int32)
    outcomes = []
    for _ in range(2):
        result = OfficialSession(1).ingest(json.loads((DATA/'wire-game-updates-v73.json').read_text())['updates'][0]['game'])
        original = deepcopy(result['public']); calls = []; actor = Policy()
        def support(env):
            calls.append((env.public, env.semantic_action()))
            return ['dream_actual_execution_not_verified_from_public_faces'] if len(calls)<=reject_count else []
        monkeypatch.setattr(OfficialDecision,'action_support',support)
        class SearchPolicy:
            def __init__(self): self.actor = actor
            def __call__(self, env, rows): return self.actor(env, rows)
        agent = SearchPolicy() if search else actor
        with _policy_rng(17, ()):
            got = advise_supported(result,agent,search=search)
        assert got['fallback_samples'] == expected_samples
        assert len(calls) == expected_samples+1
        assert all(p == original for p,_ in calls)
        assert actor.sample is False
        assert got['status'] == ('unsupported' if reject_count>16 else 'advice')
        outcomes.append(got['action'])
    assert outcomes[0] == outcomes[1]
