"""Offline Dream recovery; all public states and probes are synthetic."""
from copy import deepcopy
import json

import numpy as np
import pytest

from gfrl.model import OBS_FIELDS
from gfrl.official import OfficialDecision, OfficialSession
from gfrl.official_advisor import DREAM_REASON, advise_supported
from gfrl.official_bot import Journal, private_directory, run
from gfrl.official_ui import UIUnsupported, prepare
from test_official_bot import FakeProbe, GAMES, advisor, results, visible
from test_official_live_fixes import PreferFirst
from test_official_unobserved_kinds import PreferUnobserved


@pytest.mark.parametrize('response', [False, True])
@pytest.mark.parametrize('sample', [False, True])
def test_legal_cardless_move_survives_all_dream_card_exclusions(response, sample):
    p = deepcopy(results()[1 if response else 2]['public'])
    # A shield can be discarded or defended with, but does not forbid prayer.
    p['hand'] = [dict(id=99, face=130, used=False, disguised=True)]
    actor = PreferUnobserved(); actor.sample = sample
    original = OfficialDecision(p)
    advice = advise_supported(dict(env=original, history_complete=False), actor)
    assert advice['status'] == 'advice'
    assert advice['action']['kind'] == ('defend' if response else 'pray')
    assert advice['action']['cards'] == []
    assert not advice.get('dream_last_resort') and DREAM_REASON not in advice['warnings']
    if not response:
        assert dict(reason='ui_kind_unobserved:discard', kind='discard') in advice['mask_exclusions']
    # Rebuilding masks preserves the real hand, observation, and rule legality.
    masked = OfficialDecision(p, excluded_ids=[99], excluded_kinds=['discard'])
    assert masked.complete and masked.semantic_action() == advice['action']
    for field in OBS_FIELDS:
        np.testing.assert_array_equal(getattr(original, field), getattr(masked, field))
    assert actor.sample == sample and all(mode == sample for mode in actor.modes)


@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('search', [False, True])
def test_dream_restores_cards_only_after_safe_moves_are_exhausted(sample, search):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=99, face=6, used=False, disguised=True)]
    actor = PreferFirst(); actor.sample = sample
    class Search:
        def __init__(self): self.actor = actor
        def __call__(self, env, rows): return actor(env, rows)
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False),
                              Search() if search else actor, search=search)
    assert advice['status'] == 'advice' and advice['unsupported'] == []
    assert advice['dream_last_resort'] and DREAM_REASON in advice['warnings']
    assert advice['action']['kind'] == 'play'
    assert advice['action']['cards'] == [dict(item_id=99, model=6)]
    assert dict(reason=DREAM_REASON, item_ids=[99]) in advice['mask_exclusions']
    assert not advice.get('selection_exhausted')
    assert actor.sample == sample and all(mode == sample for mode in actor.modes)
    s = visible(p)
    with pytest.raises(UIUnsupported, match='ambiguous_disguised_face'):
        prepare(s, p, advice['action'], 'SyntheticName')
    plan = prepare(s, p, advice['action'], 'SyntheticName', dream_last_resort=True)
    assert len(plan['cards']) == 1


def test_dream_never_restores_discard_even_when_policy_prefers_it():
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=99, face=6, used=False, disguised=True),
                 dict(id=100, face=158, used=False, disguised=True)]
    actor = PreferUnobserved()
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False), actor)
    assert advice['status'] == 'advice' and advice['dream_last_resort']
    # PreferUnobserved would choose discard first if it were restored.
    assert advice['action']['kind'] == 'play'
    assert dict(reason='ui_kind_unobserved:discard', kind='discard') in advice['mask_exclusions']
    assert any(stage == 0 and mask[0, 2] for stage, mask in actor.masks)


@pytest.mark.parametrize('reject', [False, True])
def test_dream_journal_acceptance_and_received_actual_result(tmp_path, reject):
    games = deepcopy(GAMES)
    # One apparent weapon with a different actual miracle. No non-card main
    # command is legal, so the bot must select the visible weapon as a last resort.
    games[2]['players'][0]['items'] = [dict(id=99, modelId=218, fakeModelId=6, used=False)]
    games[2]['players'][0].setdefault('curses', []).append('dream')
    class Probe(FakeProbe):
        def send(self, command):
            if (reject and self.index == 2 and command.get('y') == 193
                    and command.get('kind') == 'clickPoint'):
                self.commands.append(command)
                return dict(state=self.state())
            return super().send(command)
    probe = Probe(tmp_path, games=games)
    observed = []
    def advising(result):
        observed.append(result['public'])
        if result['public']['update'] == 3:
            return advise_supported(result, PreferFirst())
        return advisor(result)
    directory = private_directory(tmp_path/'log')
    journal = Journal(directory)
    report = run(probe, advising, journal, name='SyntheticName', timeout=.05, settle_timeout=0)
    journal.close()
    rows = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r.get('state', {}).get('update') == 3)
    assert row['dream_last_resort']
    if reject:
        assert report['reason'] == 'acceptance_timeout' and not row['accepted']
        assert DREAM_REASON in row['warnings']
        assert not any(p['update'] > 3 for p in observed)
    else:
        assert report['status'] == 'complete' and row['accepted']
        assert row['received_update'] == 4
        assert row['advice']['dream_last_resort'] and DREAM_REASON in row['advice']['warnings']
        assert row['advice']['action']['cards'] == [dict(item_id=99, model=6)]
        received = next(p for p in observed if p['update'] == 4)
        session = OfficialSession(1)
        expected = [session.ingest(g) for g in games[:4]][-1]['public']
        assert received == expected
        assert any(e['decision']['faces'] == [218] for e in session.events)
        assert any(h['face'] == 218 and h['used'] for h in received['hand'])


def test_dream_last_resort_preserves_other_card_and_target_exclusions():
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=99, face=94, used=False, disguised=True),
                 dict(id=100, face=232, used=False, disguised=True)]
    class PreferWide(PreferFirst):
        def __call__(self, env, rows):
            if int(env.stage[0]) == 1 and env.chosen_count[0] > 0 and env.mask[0, 1]:
                return np.array([1], np.int32)
            return super().__call__(env, rows)
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False), PreferWide())
    assert advice['status'] == 'advice' and advice['dream_last_resort']
    assert advice['action']['cards'] == [dict(item_id=99, model=94)]
    assert dict(reason='danger_wide_requires_unobservable_mortar_absence', item_ids=[100]) in advice['mask_exclusions']

    p['hand'] = [dict(id=99, face=6, used=False, disguised=True)]
    class PreferSelf(PreferFirst):
        def __call__(self, env, rows):
            if int(env.stage[0]) == 2 and env.mask[0, 1]:
                return np.array([1], np.int32)
            return super().__call__(env, rows)
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False), PreferSelf())
    assert advice['status'] == 'advice' and advice['dream_last_resort']
    assert advice['action']['target'] == 2
    assert dict(reason='self_target_of_enemy_default_unobserved', target='self') in advice['mask_exclusions']
