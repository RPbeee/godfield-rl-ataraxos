"""Synthetic death-processing metadata; no private captures or inventories."""
from copy import deepcopy

import numpy as np
import pytest

from gfrl.model import OBS_FIELDS
from gfrl.official import OfficialSession, parser_module, project_game
from test_official_live_fixes import base_game


@pytest.mark.parametrize('response', [False, True])
def test_empty_dying_array_has_identical_projection_observation_and_mask(response):
    game = base_game()
    if response:
        game['attacks'] = [dict(playerId=2, targetPlayerId=1, itemModelIds=[6])]
    empty = dict(game, dyingPlayerIds=[])
    assert project_game(empty, 1) == project_game(game, 1)
    plain = OfficialSession(1).ingest(game)
    received = OfficialSession(1).ingest(empty)
    assert plain['status'] == received['status'] == 'decision'
    for key in (*OBS_FIELDS, 'mask'):
        np.testing.assert_array_equal(getattr(plain['env'], key), getattr(received['env'], key))


@pytest.mark.parametrize('response', [False, True])
@pytest.mark.parametrize('own_decision', [False, True])
def test_nonempty_dying_stops_only_at_own_decision(response, own_decision):
    game = base_game()
    actor = 1 if own_decision else 2
    game['attackTurnPlayerId'] = actor
    if response:
        game['attacks'] = [dict(playerId=3-actor, targetPlayerId=actor, itemModelIds=[112], atk=30)]
    game['dyingPlayerIds'] = [2]
    game['players'][1]['hp'] = 0
    game = parser_module().sanitize_game(game, 1)
    session = OfficialSession(1)
    result = session.ingest(game)
    assert result['status'] == ('unsupported' if own_decision else 'waiting')
    assert 'env' not in result
    assert result['public']['dying_players'] == [2]
    assert result['public']['players'][1]['dead'] is False
    expected = [dict(code='dying_players_decision_unavailable',
        phase=int(response), dying_players=[2], reason='native_death_state_not_reconstructible')]
    if response:
        expected.insert(0, dict(code='death_attack_decision_unavailable', models=[112],
                               reason='native_death_state_not_reconstructible'))
    assert result['detail'] == expected
    assert session.ingest(game)['status'] == 'duplicate'
    # A later authoritative update clears the transitional metadata; do not
    # carry a sticky unsupported reason into the next usable decision.
    next_game = deepcopy(game)
    next_game.update(updateCount=2, events=[], dyingPlayerIds=[], attacks=[], attackTurnPlayerId=1)
    next_game['players'][1]['hp'] = 10
    resumed = session.ingest(next_game)
    assert resumed['status'] == 'decision'
    assert 'dying_players' not in resumed['public']
    assert not resumed['unsupported']


def test_finished_dying_metadata_does_not_request_a_decision():
    game = dict(base_game(), dyingPlayerIds=[2], isOver=True)
    result = OfficialSession(1).ingest(game)
    assert result['status'] == 'finished'
    assert 'env' not in result
