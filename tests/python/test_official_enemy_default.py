"""Offline regressions for auto37's self BUY and target-only recovery."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gfrl import _core
from gfrl.official import OfficialDecision
from gfrl.official_advisor import advise_supported
from test_official_bot import results


REASON = 'self_target_of_enemy_default_unobserved'


def board(model, kind):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=99, face=model, used=False, disguised=False)]
    if kind == 5:
        p['hand'].append(dict(id=100, face=6, used=False, disguised=False))
    return p


class PreferSelf:
    def __init__(self, kind, sample=False):
        self.kind = kind
        self.sample = sample
        self.modes = []

    def __call__(self, env, rows):
        self.modes.append(self.sample)
        stage = int(env.stage[0])
        mask = env.mask[0]
        choice = self.kind if stage == 0 else 1 if stage == 2 else 18 if stage == 1 else 0
        if not mask[choice]:
            choice = int(np.flatnonzero(mask)[0])
        return np.array([choice], np.int32)


def finish(env, kind, target):
    policy = PreferSelf(kind)
    while not env.complete:
        choice = target if int(env.stage[0]) == 2 else int(policy(env, [0])[0])
        env.step([choice])
    return env


@pytest.mark.parametrize('model,kind', [(4, 5), (5, 6), (6, 0), (58, 0),
                                       (205, 0), (206, 0), (219, 0)])
def test_enemy_default_self_rejected_enemy_preserved(model, kind):
    p = board(model, kind)
    assert _core.play_catalog()[model]['targets_enemy_by_default']
    for target in (1, 2):
        env = finish(OfficialDecision(p), kind, target)
        assert env.semantic_action()['target'] == target
        assert env.action_support() == ([REASON] if target == 1 else [])


@pytest.mark.parametrize('model', [198, 201, 202])
@pytest.mark.parametrize('target', [1, 2])
def test_self_default_tools_keep_both_targets(model, target):
    env = finish(OfficialDecision(board(model, 0), exclude_enemy_default_self=True), 0, target)
    assert not _core.play_catalog()[model]['targets_enemy_by_default']
    assert env.semantic_action()['target'] == target
    assert env.action_support() == []


@pytest.mark.parametrize('model', [95, 112, 224])
def test_chance_attacks_follow_upstream_default_hint(model):
    assert not _core.play_catalog()[model]['targets_enemy_by_default']
    for filtered in (False, True):
        env = finish(OfficialDecision(board(model, 0), exclude_enemy_default_self=filtered), 0, 1)
        assert env.action_support() == []


@pytest.mark.parametrize('model,kind', [(4, 5), (5, 6), (6, 0), (205, 0), (219, 0)])
@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('search', [False, True])
def test_recovery_prunes_self_target_keeps_cards_and_policy_mode(model, kind, sample, search):
    p = board(model, kind)
    original = deepcopy(p)
    actor = PreferSelf(kind, sample)
    agent = actor
    if search:
        class Search:
            def __init__(self): self.actor = actor
            def __call__(self, env, rows): return actor(env, rows)
        agent = Search()
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False), agent, search=search)
    assert advice['status'] == 'advice'
    assert advice['action']['kind'] == ('sell' if kind == 5 else 'buy' if kind == 6 else 'play')
    assert advice['action']['cards'][0] == dict(model=model, item_id=99)
    assert advice['action']['target'] == 2
    assert advice['mask_exclusions'] == [dict(reason=REASON, target='self')]
    assert 'fallback_samples' not in advice
    assert actor.sample == sample and all(mode == sample for mode in actor.modes)
    assert p == original


@pytest.mark.parametrize('model,ability', [(281, 'boostCPToEnemy'), (282, 'takeCP')])
def test_cp_enemy_default_support_uses_catalog(model, ability):
    # Guardian faces cannot be selected as hand cards, but share the predicate.
    env = OfficialDecision(board(6, 0))
    env.public['hand'][0]['face'] = model
    env.cursor = SimpleNamespace(command=lambda: dict(kind=0, target=1, slots=[0]))
    assert _core.play_catalog()[model]['ability'] == ability
    assert env.action_support() == [REASON]


def test_target_exclusion_combines_with_card_exclusion_and_preserves_observation():
    from gfrl.model import OBS_FIELDS
    p = board(6, 0)
    p['hand'].append(dict(id=100, face=7, used=False, disguised=False))
    regular = OfficialDecision(p)
    masked = OfficialDecision(p, excluded_ids=[99], exclude_enemy_default_self=True)
    for key in OBS_FIELDS:
        np.testing.assert_array_equal(getattr(regular, key), getattr(masked, key))
    env = finish(masked, 0, 2)
    assert env.semantic_action()['cards'] == [dict(model=7, item_id=100)]
    assert env.action_support() == []


def test_self_target_filter_uses_player_identity_with_reversed_roster():
    p = board(5, 6)
    p['players'].reverse()
    for filtered in (False, True):
        env = OfficialDecision(p, exclude_enemy_default_self=filtered)
        env = finish(env, 6, 2 if filtered else 1)
        assert env.semantic_action()['target'] == (2 if filtered else 1)
        assert env.action_support() == ([] if filtered else [REASON])
