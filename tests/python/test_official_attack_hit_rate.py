"""Synthetic attack DTOs; auto54 supplied no received body to replay."""
from copy import deepcopy

import numpy as np
import pytest

from gfrl import _core
from gfrl.model import OBS_FIELDS
from gfrl.official import OfficialSession, parser_module
from test_official_live_fixes import base_game


def response(model, rate, *, target=1, guardian=None, atk=None):
    game = base_game()
    attack = dict(playerId=3-target, targetPlayerId=target,
                  itemModelIds=[model], hitRate=rate)
    if guardian:
        attack['guardian'] = guardian
    if atk is not None:
        attack['atk'] = atk
    game['attacks'] = [attack]
    return parser_module().sanitize_game(game, 1)


@pytest.mark.parametrize('model', [6, 95, 106, 112, 225])
@pytest.mark.parametrize('guardian', [None, 'earth', 'moon'])
@pytest.mark.parametrize('fallback', [False, True])
def test_catalog_or_zero_rate_preserves_observation_and_all_masks(model, guardian, fallback):
    if guardian and not _core._official_guardian_payload_supported(guardian, [model]):
        return
    game = response(model, 0 if fallback else _core.play_catalog()[model]['hit_rate'],
                    guardian=guardian)
    absent = deepcopy(game)
    del absent['attacks'][0]['hitRate']
    a, b = (OfficialSession(1).ingest(g) for g in (game, absent))
    assert a['status'] == b['status'] == 'decision'
    assert a['public'] == b['public']
    for key in OBS_FIELDS:
        np.testing.assert_array_equal(getattr(a['env'], key), getattr(b['env'], key))
    # Compare each selection prefix through confirmation, not only kind masks.
    while not a['env'].complete:
        np.testing.assert_array_equal(a['env'].mask, b['env'].mask)
        choice = np.array([np.flatnonzero(a['env'].mask[0])[0]], np.int32)
        a['env'].step(choice)
        b['env'].step(choice)
    assert b['env'].complete


@pytest.mark.parametrize('model,rate,guardian', [
    (6, 75, None), (106, 50, None), (112, 75, None),
    (6, 75, 'earth'), (225, 100, 'moon')])
@pytest.mark.parametrize('target', [1, 2])
def test_dynamic_rate_uses_policy_response_or_requires_death_power_then_resumes(model, rate, guardian, target):
    game = response(model, rate, target=target, guardian=guardian)
    session = OfficialSession(1)
    result = session.ingest(game)
    assert result['status'] == (('unsupported' if model == 112 else 'decision') if target == 1 else 'waiting')
    assert ('env' in result) == (target == 1 and model != 112)
    if model != 112:
        assert 'dynamic_hit_rate_unavailable' in result['public']['_policy_only']
    detail = next(d for d in result['detail'] if d['code'] == 'dynamic_hit_rate_unavailable')
    assert detail == dict(code='dynamic_hit_rate_unavailable', models=[model],
        hit_rate=rate, catalog_hit_rate=_core.play_catalog()[model]['hit_rate'],
        reason='dynamic_rate_requires_policy_response')
    game.update(updateCount=2, attacks=[], events=[], attackTurnPlayerId=1)
    assert session.ingest(game)['status'] == 'decision'


@pytest.mark.parametrize('target', [1, 2])
@pytest.mark.parametrize('rate', [0, 25, 75])
def test_death_bow_power_does_not_imply_native_death_state(target, rate):
    game = response(112, rate, target=target, atk=30)
    result = OfficialSession(1).ingest(game)
    assert result['status'] == ('decision' if target == 1 else 'waiting')
    assert 'death_attack_decision_unavailable' in result['public']['_policy_only']
    assert result['public']['pending']['death'] is False
    assert ('env' in result) == (target == 1)


def test_auto54_synthetic_receive_variants():
    # Read-only stop state: GF35, own HP7/MP0/CP2, enemy HP11/MP15/CP0.
    # The last logged action was model44 toward seat2. The missing body cannot
    # establish whether the next attack was 106, ordinary112, or a death bow.
    game = base_game()
    game.update(gf=35, updateCount=35, events=[])
    for p, resources in zip(game['players'], [(7, 0, 2), (11, 15, 0)]):
        p.update(zip(('hp', 'mp', 'cp'), resources))
        p.update(curses=[], guardian=None, items=[])
    models = [44, 3, 69, 98, 106, 112, 198, 199, 219, 237]
    game['players'][0]['items'] = [dict(id=i+100, modelId=m, fakeModelId=0, used=False)
                                 for i, m in enumerate(models)]
    session = OfficialSession(1)
    assert session.ingest(parser_module().sanitize_game(game, 1))['status'] == 'decision'
    for model, atk, rate, expected in [(106, 6, 75, 'decision'),
                                     (112, 1, 25, 'decision'),
                                     (112, 30, 75, 'decision')]:
        for target in (1, 2):
            received = deepcopy(game)
            received.update(updateCount=36, attacks=[dict(playerId=3-target,
                targetPlayerId=target, itemModelIds=[model], atk=atk, hitRate=rate)])
            result = deepcopy(session).ingest(parser_module().sanitize_game(received, 1))
            assert result['status'] == (expected if target == 1 else 'waiting')


def test_future_attack_rate_does_not_block_current_head():
    game = response(106, 75)
    game['attacks'].append(dict(playerId=2, targetPlayerId=1,
                               itemModelIds=[112], atk=30, hitRate=75))
    assert OfficialSession(1).ingest(game)['status'] == 'decision'


def test_finished_dynamic_attack_does_not_request_decision():
    game = response(112, 75, atk=30)
    game['isOver'] = True
    assert OfficialSession(1).ingest(game)['status'] == 'finished'


@pytest.mark.parametrize('target', [1, 2])
def test_lenient_dying_event_cannot_hide_active_death_bow(target):
    game = response(112, 75, target=target, atk=30)
    # This event remains outside the strict event schema. Replacing its opaque
    # body must not discard the authoritative active attack's rate/provenance.
    game['events'] = [dict(action='attackDyingly')]
    checked = parser_module().sanitize_game(game, 1, lenient_events=True)
    assert checked['events'] == [dict(action='unknownEvent')]
    result = OfficialSession(1).ingest(checked)
    assert result['status'] == ('decision' if target == 1 else 'waiting')
    assert 'death_attack_decision_unavailable' in result['public']['_policy_only']
    assert not result['history_complete']
