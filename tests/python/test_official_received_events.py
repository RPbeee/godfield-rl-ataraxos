"""Synthetic receipts and native witnesses for the second event intake fix."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gfrl import _core
from gfrl.official import OfficialSession, parser_module, project_game
from test_official import received_scene, align_hand
from test_official_live_fixes import base_game

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('scenario', ['revival', 'heaven'])
def test_dark_revival_and_heaven_history_match_native(scenario):
    run = subprocess.run([str(ROOT/'build/gfrl_official_native_dump'), '--received-effects', scenario],
                         capture_output=True, text=True, check=True)
    records = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
    for viewer in (0, 7):
        session = OfficialSession(viewer)
        scenes = [n for n in records if n['viewer'] == viewer]
        for i, n in enumerate(scenes):
            if i == 0:
                events = [dict(action='startGame')]
            elif scenario == 'heaven':
                events = [dict(action='pray'), dict(action='upgradeHeaven', playerId=0),
                          dict(action='dealDamage', playerId=0, damage=40),
                          dict(action='die', playerId=0), dict(action='endGame')]
            elif i == 1:
                events = [dict(action='useAttackItems', items=[dict(id=0, modelId=64)]),
                          dict(action='setTargetPlayer', playerId=7)]
            else:
                events = [dict(action='useDefenseItems', items=[]),
                          # Deliberately differs from HP: client ignores damage.
                          dict(action='dealDarkDamage', damage=1),
                          dict(action='revive', playerId=7,
                               item=dict(id=0, modelId=208, fakeModelId=0, used=False)),
                          dict(action='boostHP', playerId=7, hp=10),
                          dict(action='advanceGF', playerId=7)]
            g = received_scene(n, i+1, events)
            g['isOver'] = n['view']['board']['is_over']
            replaced = []
            clean = parser_module().sanitize_game(g, viewer, lenient_events=True, replaced_events=replaced)
            assert replaced == []
            result = session.ingest(clean)
            assert result['status'] != 'unsupported', result
            assert result['history_complete']
            p = align_hand(result['public'], n)
            obs = _core._official_observation(p, session.events, 64)
            for key, col in [('obs_type',0), ('obs_aux_card',1), ('obs_cards',2), ('obs_cat',3), ('obs_num',4)]:
                np.testing.assert_allclose(np.asarray(obs[key]), np.asarray([t[col] for t in n['tokens']]),
                                           atol=1e-7, err_msg=f'{scenario}/{viewer}/{i}/{key}')


def test_response_markers_do_not_create_curse_damage_or_counter_decisions():
    g = base_game(); session = OfficialSession(1); session.ingest(g)
    n = deepcopy(g); n['updateCount'] = 2
    n['events'] = [dict(action='useDefenseItems', items=[dict(modelId=130)]),
                   dict(action='counterAttack', index=0), dict(action='selfCurse', index=0)]
    clean = parser_module().sanitize_game(n, 1)
    r = session.ingest(clean)
    assert r['history_complete']
    assert len(session.events) == 1 and session.events[0]['decision']['kind'] == 1
    assert session.events[0]['before']['players'] == session.events[0]['after']['players']
    assert r['public']['pending'] is None


def test_collect_cp_is_known_but_cannot_fabricate_external_native_decision():
    g = base_game(); session = OfficialSession(1); session.ingest(g)
    n = deepcopy(g); n['updateCount'] = 2
    n['players'][0]['cp'] = 40; n['players'][1]['cp'] = 0
    n['events'] = [dict(action='collectCPOfEverybody', playerId=1)]
    replaced = []
    clean = parser_module().sanitize_game(n, 1, lenient_events=True, replaced_events=replaced)
    r = session.ingest(clean)
    assert replaced == [] and not r['history_complete'] and r['status'] == 'decision'
    assert r['public']['events'] == [dict(action='collectCPOfEverybody', player=1)]
    assert session.events == [] and r['public']['last_decision'] is None


def test_revive_releases_only_consumed_item_and_never_invents_hp_or_alive():
    g = base_game(); g['players'][1]['hp'] = 0; g['players'][1]['dead'] = True
    g['events'] = [dict(action='revive', playerId=2,
                        item=dict(id=998, modelId=208, fakeModelId=130, used=False))]
    public = project_game(g, 1)
    for secret in (6, 296):
        changed = deepcopy(g)
        changed['events'][0]['item']['id'] = secret
        changed['players'][1]['items'] = [dict(id=secret, modelId=secret, used=False)]*9
        assert project_game(changed, 1) == public
    assert public['events'] == [dict(action='revive', player=2, model=208, face=130)]
    r = OfficialSession(1).ingest(g)
    assert not r['history_complete']  # Fake Reveal/decision anchor unavailable.
    other = r['public']['players'][1]
    assert other['dead'] and other['status']['hp'] == 0


def test_null_overflow_is_no_eviction_and_foreign_unused_truth_stays_private():
    g = base_game(); g['events'] = [dict(action='setTargetPlayer', playerId=2),
        dict(action='sell', item=dict(id=1, modelId=6, fakeModelId=0, used=False), overflowItem=None)]
    p = project_game(parser_module().sanitize_game(g, 1), 1)
    assert p['events'][-1] == dict(action='sell', buyer=2, seller=1)
    g['events'][-1]['overflowItem'] = dict(id=2, modelId=218, fakeModelId=0, used=False)
    p = project_game(parser_module().sanitize_game(g, 1), 1)
    g['events'][-1]['overflowItem']['modelId'] = 296
    g['events'][-1]['item']['modelId'] = 296
    assert project_game(parser_module().sanitize_game(g, 1), 1) == p
    assert p['events'][-1]['overflow_unobserved']
