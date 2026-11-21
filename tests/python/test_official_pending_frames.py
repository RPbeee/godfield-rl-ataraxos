"""Synthetic offline regressions; no live receipts, names or transport data."""
from copy import deepcopy
import json
import subprocess

import numpy as np
import pytest

from gfrl import _core
from gfrl.official import DECISION_ONLY_UNSUPPORTED, OfficialSession, OfficialStreams, project_game
from gfrl.official_advisor import advise
from test_official import ROOT, FirstLegal, received_scene, token_arrays
from test_official_bot import capture_file
from test_official_live_fixes import base_game


@pytest.mark.parametrize('code', sorted(DECISION_ONLY_UNSUPPORTED))
@pytest.mark.parametrize('own_turn', [False, True])
def test_decision_only_reasons_wait_only_for_foreign_actor(code, own_turn):
    p = project_game(base_game(), 1)
    p['actor'] = 1 if own_turn else 2
    p['unsupported'] = [code]
    session = OfficialSession(1)
    r = session.ingest_public(p)
    assert r['status'] == ('unsupported' if own_turn else 'waiting')
    assert r['unsupported'] == [code]
    assert session.previous['update'] == p['update']


@pytest.mark.parametrize('code', ['team_game', 'sale_recipient_unavailable', 'unknown_event',
                                   'ambiguous_pending_actor', 'future_unknown_reason'])
def test_public_state_failures_stop_even_for_foreign_actor(code):
    p = project_game(base_game(), 1)
    p.update(actor=2, unsupported=[code])
    assert OfficialSession(1).ingest_public(p)['status'] == 'unsupported'


def test_foreign_dynamic_frame_does_not_latch_stream_stop(tmp_path):
    g = base_game()
    g['attacks'] = [dict(playerId=1, targetPlayerId=2, itemModelIds=[58])]
    streams = OfficialStreams(OfficialSession(1))
    r = streams.poll([capture_file(tmp_path, [g])])
    assert r['status'] == 'waiting' and not streams.failed
    assert r['unsupported'] == ['dynamic_attack_value_unavailable']
    n = deepcopy(g)
    n.update(updateCount=2, attacks=[], events=[])
    assert streams.poll([capture_file(tmp_path, [g, n], serial=2)])['status'] == 'decision'
    bad = deepcopy(n)
    bad.update(updateCount=3, attackTurnPlayerId=2, gf='invalid')
    assert streams.poll([capture_file(tmp_path, [g, n, bad], serial=3)])['status'] == 'unsupported'
    assert streams.failed


def test_multiple_frames_use_head_and_safe_diagnostics():
    g = base_game()
    g['attacks'] = [dict(playerId=2, targetPlayerId=1, itemModelIds=[6], atk=3),
                    dict(playerId=1, targetPlayerId=2, itemModelIds=[271], guardian='pluto',
                         privateName='MustNotEscape', opaque='MustNotEscape')]
    # The last event is about the tail; it must not overwrite head provenance.
    g['events'].append(dict(action='nextAttack', attack=g['attacks'][1]))
    r = OfficialSession(1).ingest(g)
    assert r['status'] == 'decision' and not r['history_complete']
    assert (r['public']['actor'], r['public']['pending']['source']) == (1, 2)
    assert r['public']['pending']['models'] == [6]
    assert advise(r, FirstLegal())['detail'] == r['detail'] == r['public']['detail']
    assert r['public']['detail'] == [dict(code='multiple_pending_frames', count=2, frames=[
        dict(playerId=2, targetPlayerId=1, itemModelIds=[6], has_atk=True, guardian='none'),
        dict(playerId=1, targetPlayerId=2, itemModelIds=[271], has_atk=False, guardian='pluto')])]
    assert 'MustNotEscape' not in json.dumps(r['public'])
    g['attacks'][0]['targetPlayerId'] = None
    stopped = OfficialSession(1).ingest(g)
    assert stopped['status'] == 'unsupported'
    assert any(d['code'] == 'multiple_pending_frames' and len(d['frames']) == 2 for d in stopped['detail'])


@pytest.mark.parametrize('mp,models,expected', [(17,[58],34), (0,[58],0), (10,[58,231],8),
                                             (10,[58,231,204],20)])
def test_58_uses_observed_launch_mp_and_native_composed_cost(mp, models, expected):
    g = base_game()
    g['attackTurnPlayerId'] = 2
    g['players'][1]['mp'] = mp
    s = OfficialSession(1)
    assert s.ingest(g)['status'] == 'waiting'
    n = deepcopy(g)
    n.update(updateCount=2, attacks=[dict(playerId=2,targetPlayerId=1,itemModelIds=models)],
             events=[dict(action='useAttackItems', items=[dict(modelId=m) for m in models]),
                     dict(action='setTargetPlayer',playerId=1)])
    n['players'][1]['mp'] = 0
    r = s.ingest(n)
    assert r['status'] == 'decision'
    assert r['public']['pending']['base_override'] == expected
    # A first receipt or a missing update cannot recover consumed launch MP.
    assert OfficialSession(1).ingest(n)['status'] == 'unsupported'
    n['updateCount'] = 4
    assert s.ingest(n)['status'] == 'unsupported'


def test_58_wire_atk_zero_is_authoritative():
    g = base_game()
    g['attacks'] = [dict(playerId=2,targetPlayerId=1,itemModelIds=[58],atk=0)]
    r = OfficialSession(1).ingest(g)
    assert r['status'] == 'decision'
    assert r['public']['pending']['base_override'] == 0


def test_58_does_not_guess_mp_across_unknown_event():
    g = base_game()
    g['attackTurnPlayerId'] = 2
    s = OfficialSession(1)
    s.ingest(g)
    n = deepcopy(g)
    n.update(updateCount=2, attacks=[dict(playerId=2,targetPlayerId=1,itemModelIds=[58])],
             events=[dict(action='unknownEvent'),
                     dict(action='useAttackItems',items=[dict(modelId=58)]),
                     dict(action='setTargetPlayer',playerId=1)])
    n['players'][1]['mp'] = 0
    r = s.ingest(n)
    assert r['status'] == 'unsupported'
    assert r['unsupported'] == ['dynamic_attack_value_unavailable']


def test_58_reconstructed_observation_and_mask_match_native():
    output = subprocess.check_output([str(ROOT/'build/gfrl_official_native_dump'), '--dynamic'], text=True)
    rows = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
    for mp in (10,20):
        n = next(n for n in rows if n['viewer'] == 7 and n['view'].get('pending')
                 and n['view']['pending']['models'] == [58] and n['view']['pending']['power'] == 2*mp)
        final = received_scene(n, 2, [dict(action='useAttackItems',items=[dict(modelId=58)]),
                                      dict(action='setTargetPlayer',playerId=7)])
        initial = deepcopy(final)
        initial.update(updateCount=1,attacks=[],events=[dict(action='startGame')])
        initial['players'][0]['mp'] = mp
        s = OfficialSession(7)
        assert s.ingest(initial)['status'] == 'waiting'
        r = s.ingest(final)
        assert r['status'] == 'decision'
        p = r['public']; p['decisions'] = n['view']['decisions']
        got = _core._official_observation(p, [], 0)
        for key, want in token_arrays(n).items():
            np.testing.assert_allclose(got[key], want, atol=1e-7)
        for prefix in n['prefixes']:
            np.testing.assert_array_equal(r['env'].mask[0], prefix['mask'])
            r['env'].step(np.array([prefix['choice']],np.int32))


@pytest.mark.parametrize('with_atk', [False, True])
def test_202_signed_amount_and_response_masks_match_native(with_atk):
    output = subprocess.check_output([str(ROOT/'build/gfrl_official_native_dump'), '--dynamic'], text=True)
    rows = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
    signs = set()
    for n in rows:
        pending = n['view'].get('pending')
        if n['viewer'] != 7 or not pending or pending['models'] != [202]:
            continue
        signs.add(pending['amount'])
        game = received_scene(n, 1, [])
        if with_atk:
            game['attacks'][0]['atk'] = pending['amount']
        r = OfficialSession(7).ingest(game)
        assert r['status'] == 'decision'
        p = r['public']
        assert p['pending']['power'] == pending['power'] == 0
        assert 'base_override' not in p['pending']
        assert p.get('warnings', []) == ([] if with_atk else ['assumed_dynamic_value: 202'])
        assert p['pending']['amount'] == (pending['amount'] if with_atk else -10)
        # With no sign the fallback observation matches the negative native
        # branch. With a received sign every observation matches its own branch.
        if with_atk or pending['amount'] < 0:
            p['decisions'] = n['view']['decisions']
            got = _core._official_observation(p, [], 0)
            for key, want in token_arrays(n).items():
                np.testing.assert_allclose(got[key], want, atol=1e-7)
        env = r['env']
        assert not env.mask[0, 0]  # ordinary armor 130
        assert env.mask[0, 1]      # reflect-anything 190
        for prefix in n['prefixes']:
            np.testing.assert_array_equal(env.mask[0], prefix['mask'])
            env.step(np.array([prefix['choice']], np.int32))
        assert env.complete
    assert signs == {-10, 10}


def test_received_attack_base_values_match_native_catalog_and_modifiers():
    output = subprocess.check_output([str(ROOT/'build/gfrl_official_native_dump'), '--attack-values'], text=True)
    rows = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
    checked = set()
    composed = set()
    for n in rows:
        pending = n['view'].get('pending')
        if n['viewer'] != 7 or not pending or n['view']['actor'] != 7:
            continue
        models = pending['models']
        model = models[0]
        checked.add(model)
        if len(models) > 1:
            composed.add(tuple(models))
        game = received_scene(n, 1, [])
        # Wire atk is the dynamic BASE value, before ordered modifiers.
        game['attacks'][0]['atk'] = 60 - 2 * _core._official_magic_cost(models) if model == 58 else _core.play_catalog()[model]['atk']
        r = OfficialSession(7).ingest(game)
        assert r['status'] == 'decision', models
        p = r['public']; p['decisions'] = n['view']['decisions']
        got = _core._official_observation(p, [], 0)
        for key, want in token_arrays(n).items():
            np.testing.assert_allclose(got[key], want, atol=1e-7, err_msg=str(models))
        env = r['env']
        for prefix in n['prefixes']:
            np.testing.assert_array_equal(env.mask[0], prefix['mask'], err_msg=str(models))
            env.step(np.array([prefix['choice']], np.int32))
        assert env.complete
    assert checked == {
        6, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 21, 23, 24, 26, 27, 29, 30,
        32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50,
        52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64, 65, 67, 69, 70, 71, 72,
        73, 75, 76, 77, 79, 80, 81, 82, 85, 86, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        111, 112, 211, 212, 213, 214, 215, 216, 218, 224, 225, 226, 227, 228, 229, 230}
    assert len(composed) == 12


def test_received_counter_values_match_native():
    output = subprocess.check_output([str(ROOT/'build/gfrl_official_native_dump'), '--counter-values'], text=True)
    checked = set()
    for line in output.splitlines():
        if not line.startswith('{'):
            continue
        n = json.loads(line); pending = n['view'].get('pending')
        if not pending or n['view']['actor'] != n['viewer']:
            continue
        model = pending['models'][0]; checked.add(model)
        game = received_scene(n, 1, [])
        game['attacks'][0]['atk'] = pending['power'] or pending['amount']
        game['events'] = [dict(action='nextAttack', attack=deepcopy(game['attacks'][0]))]
        r = OfficialSession(n['viewer']).ingest(game)
        assert r['status'] == 'decision', model
        p = r['public']; p['decisions'] = n['view']['decisions']
        got = _core._official_observation(p, [], 0)
        for key, want in token_arrays(n).items():
            np.testing.assert_allclose(got[key], want, atol=1e-7, err_msg=str(model))
        for prefix in n['prefixes']:
            np.testing.assert_array_equal(r['env'].mask[0], prefix['mask'], err_msg=str(model))
            r['env'].step(np.array([prefix['choice']], np.int32))
    assert checked == set(range(181, 189))


@pytest.mark.parametrize('atk', [-47, 0, 47])
def test_202_wire_magnitude_never_enables_ordinary_armor(atk):
    game = base_game()
    game['attacks'] = [dict(playerId=2, targetPlayerId=1, itemModelIds=[202], atk=atk)]
    game['players'][0]['items'] = [dict(id=i, modelId=m) for i, m in enumerate((130, 190))]
    r = OfficialSession(1).ingest(game)
    assert r['status'] == 'decision'
    assert r['public']['pending']['amount'] == atk
    assert r['public']['pending']['power'] == 0
    assert not r['public'].get('warnings')
    assert not r['env'].mask[0, 0] and r['env'].mask[0, 1]


@pytest.mark.parametrize('redirect', ['reflect', 'bounce'])
def test_attack_by_guardian_redirect_tracks_launch_after_pray(redirect):
    s = OfficialSession(1)
    first = base_game()
    first.update(updateCount=1, attackTurnPlayerId=1, events=[dict(action='pray')], attacks=[])
    assert s.ingest(first)['status'] == 'decision'
    second = deepcopy(first)
    second.update(
        updateCount=2,
        events=[
            dict(action='attackByGuardian',
                 attack=dict(playerId=1, targetPlayerId=2, itemModelIds=[265], guardian='mars')),
            dict(action='useDefenseItems', items=[dict(modelId=190)]),
            dict(action=redirect),
        ],
        attacks=[dict(playerId=2, targetPlayerId=1, itemModelIds=[265], guardian='mars', reversed=True)],
    )
    r = s.ingest(second)
    assert r['status'] == 'decision'
    assert 'redirect_requires_history' not in r['unsupported']
    pending = r['public']['pending']
    assert pending['origin'] == 1
    assert pending['reflections' if redirect == 'reflect' else 'bounces'] == 1
    assert not pending['counter']
    assert not r['history_complete']


def test_next_attack_updates_target_for_same_update_reflect():
    s = OfficialSession(1)
    first = base_game()
    first.update(updateCount=1, attackTurnPlayerId=2,
                 attacks=[dict(playerId=2, targetPlayerId=1, itemModelIds=[6], atk=3)],
                 events=[dict(action='useAttackItems', items=[dict(modelId=6)]),
                         dict(action='setTargetPlayer', playerId=1)])
    assert s.ingest(first)['status'] == 'decision'
    second = deepcopy(first)
    second.update(
        updateCount=2,
        events=[
            dict(action='nextAttack', attack=dict(playerId=1, targetPlayerId=2, itemModelIds=[184], atk=22)),
            dict(action='useDefenseItems', items=[dict(modelId=190)]),
            dict(action='reflect'),
        ],
        attacks=[dict(playerId=2, targetPlayerId=1, itemModelIds=[184], atk=22, reversed=True)],
    )
    r = s.ingest(second)
    assert r['status'] == 'decision'
    pending = r['public']['pending']
    assert (pending['origin'], pending['source'], pending['target']) == (1, 2, 1)
    assert pending['reflections'] == 1 and pending['counter'] is True
