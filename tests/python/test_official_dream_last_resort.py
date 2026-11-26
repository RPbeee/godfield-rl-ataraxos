"""Offline Dream recovery; all public states and probes are synthetic."""
from copy import deepcopy
import json

import numpy as np
import pytest

from gfrl.model import OBS_FIELDS
from gfrl.official import OfficialDecision, OfficialSession, ambiguous_disguised_ids
from gfrl.official_advisor import AMBIGUOUS_REASON, DREAM_REASON, advise, advise_supported
from gfrl.official_bot import Journal, private_directory, run
from gfrl.official_ui import PATHS, UIUnsupported, execute, point, prepare
from test_official_bot import FakeProbe, GAMES, advisor, results, visible
from test_official_live_fixes import PreferFirst
from test_official_unobserved_kinds import PreferUnobserved


@pytest.mark.parametrize('dream', [None, 'allow', 'last-resort'])
def test_bot_cli_plumbs_dream_mode(tmp_path, monkeypatch, dream):
    from gfrl import actor, evaluation, official_bot as bot
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=99, face=200, used=False, disguised=True),
                 dict(id=100, face=6, used=False, disguised=False)]
    monkeypatch.setattr(evaluation, 'load_policy_model', lambda *a: None)
    monkeypatch.setattr(actor, 'Actor', lambda *a, **kw: PreferFirst())
    monkeypatch.setattr(bot, 'Probe', lambda *a: None)
    observed = []
    def running(probe, advisor, journal, **kwargs):
        observed.append(advisor(dict(env=OfficialDecision(p), history_complete=False)))
        return dict(status='complete')
    monkeypatch.setattr(bot, 'run', running)
    args = ['--checkpoint', 'unused', '--playwright', 'unused', '--out-dir', str(tmp_path/'log')]
    assert bot.main(args + ([] if dream is None else ['--dream', dream])) == 0
    assert len(observed) == 1
    advice = observed[0]
    assert advice['action']['cards'] == [dict(item_id=100, model=6) if dream == 'last-resort'
                                        else dict(item_id=99, model=200)]
    assert bool(advice.get('dream_played')) == (dream != 'last-resort')


def test_dream_allow_still_requires_selected_visible_face(tmp_path):
    games = deepcopy(GAMES)
    games[2]['players'][0]['items'] = [dict(id=99, modelId=218, fakeModelId=6, used=False)]
    class WrongSelection(FakeProbe):
        def send(self, command):
            response = super().send(command)
            if command.get('y') == 635 and self.selected:
                self.selected = [PATHS[7]]
                response['state'] = self.state()
            return response
    probe = WrongSelection(tmp_path, games=games)
    probe.phase = 'battle'; probe.index = 2
    result = probe.rs[2]; p = result['public']
    advice = advise_supported(result, PreferFirst())
    assert advice['dream_played']
    with pytest.raises(UIUnsupported, match='selected_cards_mismatch'):
        execute(probe, probe.state(), p, advice['action'], 'SyntheticName',
                allow_dream=True, settle_timeout=0)
    assert not any(c.get('y') == 193 for c in probe.commands)


def test_dream_defense_requires_next_response_stability_without_face_inference(tmp_path, monkeypatch):
    from gfrl import official_bot as bot
    from gfrl.official_ui import REFLECT_MODELS
    games = deepcopy(GAMES)
    games[3]['players'][0]['items'] = [dict(id=99, modelId=39, fakeModelId=130, used=False)]
    games[3]['attacks'][0]['itemModelIds'] = [6]
    games[3]['events'][0]['items'] = [dict(id=5, modelId=6)]
    assert 130 not in REFLECT_MODELS
    def advising(result):
        if result['public']['update'] == 4:
            advice = advise_supported(result, PreferFirst())
            assert advice['dream_played'] and advice['action']['cards'] == [dict(item_id=99, model=130)]
            return advice
        return advisor(result)
    observed = []
    def executing(probe, state, public, action, name, **kwargs):
        observed.append((public['update'], kwargs['previous_reflect']))
        return execute(probe, state, public, action, name, **kwargs)
    monkeypatch.setattr(bot, 'execute', executing)
    probe = FakeProbe(tmp_path, games=games)
    journal = Journal(private_directory(tmp_path/'log'))
    report = run(probe, advising, journal, name='SyntheticName', timeout=.05, settle_timeout=1)
    journal.close()
    assert report['status'] == 'complete'
    assert (5, True) in observed


@pytest.mark.parametrize('dream', [None, 'allow', 'last-resort'])
@pytest.mark.parametrize('search', [False, True])
def test_unique_face_dream_mode_preserves_or_excludes_sampled_action(dream, search):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=99, face=200, used=False, disguised=True),
                 dict(id=100, face=6, used=False, disguised=False)]
    position = lambda: dict(env=OfficialDecision(p), history_complete=False)
    actor = PreferFirst(); actor.sample = True
    class Search:
        def __init__(self): self.actor = actor
        def __call__(self, env, rows): return actor(env, rows)
    agent = Search() if search else actor
    initial = advise(position(), agent, search=search)
    calls = len(actor.modes)
    assert initial['unsupported'] == [DREAM_REASON]
    actor.modes.clear()
    advice = advise_supported(position(), agent, search=search,
                              **({} if dream is None else dict(dream=dream)))
    assert advice['status'] == 'advice' and advice['unsupported'] == []
    if dream == 'last-resort':
        assert advice['action']['cards'] == [dict(item_id=100, model=6)]
        assert dict(reason=DREAM_REASON, item_ids=[99]) in advice['mask_exclusions']
        assert not advice.get('dream_played')
    else:
        assert advice['action'] == initial['action']
        assert advice['selections'] == initial['selections']
        assert advice['search_status'] == initial['search_status']
        assert advice['dream_played'] is True and DREAM_REASON in advice['warnings']
        assert not advice.get('mask_exclusions') and not advice.get('dream_last_resort')
        assert len(actor.modes) == calls
        assert prepare(visible(p), p, advice['action'], 'SyntheticName', allow_dream=True)['cards']
    assert actor.sample and all(actor.modes)


@pytest.mark.parametrize('dream', ['allow', 'last-resort'])
@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('disguised_first', [False, True])
def test_mixed_face_advice_and_click(dream, search, disguised_first):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=18, face=6, used=False, disguised=disguised_first),
                 dict(id=99, face=6, used=False, disguised=not disguised_first)]
    assert ambiguous_disguised_ids(p['hand']) == set()
    position = lambda: dict(env=OfficialDecision(p), history_complete=False)
    initial = advise(position(), PreferFirst())
    assert initial['action']['cards'] == [dict(item_id=18, model=6)]
    assert initial['unsupported'] == ([DREAM_REASON] if disguised_first else [])
    actor = PreferFirst()
    class Search:
        def __init__(self): self.actor = actor
        def __call__(self, env, rows): return actor(env, rows)
    advice = advise_supported(position(), Search() if search else actor, search=search, dream=dream)
    assert advice['status'] == 'advice' and advice['unsupported'] == []
    excluded = disguised_first and dream == 'last-resort'
    dream_played = disguised_first and not excluded
    assert advice['action']['cards'] == [dict(item_id=99 if excluded else 18, model=6)]
    assert advice.get('mask_exclusions', []) == ([dict(reason=DREAM_REASON, item_ids=[18])] if excluded else [])
    assert bool(advice.get('dream_played')) == dream_played
    assert (DREAM_REASON in advice['warnings']) == dream_played
    assert AMBIGUOUS_REASON not in advice['warnings'] and not advice.get('dream_last_resort')
    s = visible(p)
    if disguised_first:
        with pytest.raises(UIUnsupported, match=AMBIGUOUS_REASON):
            prepare(s, p, initial['action'], 'SyntheticName')
    plan = prepare(s, p, advice['action'], 'SyntheticName', allow_dream=dream_played)
    assert plan['cards'] == [point(804 if excluded else 886, 635)]


@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('dream', ['allow', 'last-resort'])
@pytest.mark.parametrize('with_real', [False, True])
def test_duplicate_disguised_face(dream, search, with_real):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=18, face=6, used=False, disguised=True),
                 dict(id=99, face=6, used=False, disguised=True)]
    if with_real:
        p['hand'].append(dict(id=100, face=6, used=False, disguised=False))
    assert ambiguous_disguised_ids(p['hand']) == {18, 99}
    actor = PreferFirst()
    class Search:
        def __init__(self): self.actor = actor
        def __call__(self, env, rows): return actor(env, rows)
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False),
                              Search() if search else actor, search=search, dream=dream)
    assert advice['status'] == 'advice' and advice['unsupported'] == []
    dream_played = dream == 'allow' or not with_real
    last_resort = dream == 'last-resort' and not with_real
    assert advice['action']['cards'] == [dict(item_id=18 if dream_played else 100, model=6)]
    assert bool(advice.get('dream_played')) == dream_played
    assert bool(advice.get('dream_last_resort')) == last_resort
    assert (DREAM_REASON in advice['warnings']) == (AMBIGUOUS_REASON in advice['warnings']) == dream_played
    if dream == 'allow':
        assert not advice.get('mask_exclusions')
    else:
        excluded = {i for e in advice['mask_exclusions'] if e['reason'] == DREAM_REASON for i in e['item_ids']}
        assert excluded == {18, 99}
    plan = prepare(visible(p), p, advice['action'], 'SyntheticName',
                   allow_dream=dream == 'allow', dream_last_resort=last_resort)
    assert plan['cards'] == [point(804 if dream_played else 722, 635)]


def test_disguised_duplicates_group_by_face_and_used_state():
    hand = [dict(id=1, face=218, used=False, disguised=True),
            dict(id=2, face=218, used=False, disguised=False),
            dict(id=3, face=218, used=True, disguised=True),
            dict(id=4, face=218, used=True, disguised=True)]
    assert ambiguous_disguised_ids(hand) == {3, 4}


@pytest.mark.parametrize('dream', ['allow', 'last-resort'])
def test_real_core_protector_can_defend_with_two_disguised_copies(dream):
    p = deepcopy(results()[1]['public'])
    p['players'][0]['status']['curses'] = ['dream']
    p['pending'].update(models=[6], power=1, element=None, category='weapons')
    p['hand'] = [dict(id=18, face=176, used=False, disguised=True),
                 dict(id=99, face=176, used=False, disguised=True),
                 dict(id=100, face=176, used=False, disguised=False)]
    class ChooseOneCard(PreferFirst):
        def __call__(self, env, rows):
            if int(env.stage[0]) == 1 and env.chosen_count[0] >= 1:
                finish = int(np.flatnonzero(env.mask[0])[-1])
                return np.array([finish], np.int32)
            return super().__call__(env, rows)
    if dream == 'allow':
        p['hand'].insert(0, p['hand'].pop())
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False), ChooseOneCard(), dream=dream)
    assert advice['status'] == 'advice' and advice['action']['kind'] == 'defend'
    assert advice['action']['cards'] == [dict(item_id=100, model=176)]
    assert not advice.get('dream_played') and not advice.get('dream_last_resort')
    assert DREAM_REASON not in advice['warnings'] and AMBIGUOUS_REASON not in advice['warnings']
    assert prepare(visible(p), p, advice['action'], 'SyntheticName')['cards'] == [
        point(886 if dream == 'allow' else 722, 635)]
    if dream == 'allow':
        multi = advise_supported(dict(env=OfficialDecision(p), history_complete=False), PreferFirst(), dream='allow')
        assert multi['status'] == 'advice' and multi['dream_played'] and not multi.get('dream_last_resort')
        assert [c['item_id'] for c in multi['action']['cards']] == [100, 18, 99]
        assert prepare(visible(p), p, multi['action'], 'SyntheticName', allow_dream=True)['cards'] == [
            point(886, 635), point(722, 635), point(804, 635)]


@pytest.mark.parametrize('response', [False, True])
@pytest.mark.parametrize('sample', [False, True])
def test_legal_cardless_move_survives_all_dream_card_exclusions(response, sample):
    p = deepcopy(results()[1 if response else 2]['public'])
    # A shield can be discarded or defended with, but does not forbid prayer.
    p['hand'] = [dict(id=99, face=130, used=False, disguised=True)]
    actor = PreferUnobserved(); actor.sample = sample
    original = OfficialDecision(p)
    advice = advise_supported(dict(env=original, history_complete=False), actor, dream='last-resort')
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
                              Search() if search else actor, search=search, dream='last-resort')
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


@pytest.mark.parametrize('dream', ['allow', 'last-resort'])
def test_dream_never_restores_discard_even_when_policy_prefers_it(dream):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=99, face=6, used=False, disguised=True),
                 dict(id=100, face=158, used=False, disguised=True)]
    actor = PreferUnobserved()
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False), actor, dream=dream)
    assert advice['status'] == 'advice' and advice['dream_played']
    # PreferUnobserved would choose discard first if it were restored.
    assert advice['action']['kind'] == 'play'
    assert dict(reason='ui_kind_unobserved:discard', kind='discard') in advice['mask_exclusions']
    assert any(stage == 0 and mask[0, 2] for stage, mask in actor.masks)


@pytest.mark.parametrize('reject', [False, True])
@pytest.mark.parametrize('ambiguous', [False, True])
@pytest.mark.parametrize('dream', ['allow', 'last-resort'])
def test_dream_journal_acceptance_and_received_actual_result(tmp_path, reject, ambiguous, dream):
    games = deepcopy(GAMES)
    # One apparent weapon with a different actual miracle. No non-card main
    # command is legal, so last-resort mode must restore the visible weapon.
    games[2]['players'][0]['items'] = [dict(id=99, modelId=218, fakeModelId=6, used=False)]
    if ambiguous:
        games[2]['players'][0]['items'].insert(0, dict(id=18, modelId=218, fakeModelId=6, used=False))
    reason = AMBIGUOUS_REASON if ambiguous else DREAM_REASON
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
            return advise_supported(result, PreferFirst(), dream=dream)
        return advisor(result)
    directory = private_directory(tmp_path/'log')
    journal = Journal(directory)
    # The replay includes a later reflected response requiring one confirm read.
    report = run(probe, advising, journal, name='SyntheticName', timeout=.05, settle_timeout=.5)
    journal.close()
    rows = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r.get('state', {}).get('update') == 3)
    assert row['dream_played'] is True
    assert bool(row.get('dream_last_resort')) == (dream == 'last-resort')
    assert reason in row['warnings']
    if reject:
        assert report['reason'] == 'acceptance_timeout' and not row['accepted']
        assert reason in row['warnings']
        if dream == 'last-resort':
            assert dict(reason=reason, item_ids=[18 if ambiguous else 99]) in row['mask_exclusions']
        else:
            assert not row.get('mask_exclusions')
        assert not any(p['update'] > 3 for p in observed)
    else:
        assert report['status'] == 'complete' and row['accepted']
        assert row['received_update'] == 4
        assert row['advice']['dream_played'] is True and reason in row['advice']['warnings']
        assert row['advice']['unsupported'] == []
        assert row['advice']['action']['cards'] == [dict(item_id=18 if ambiguous else 99, model=6)]
        received = next(p for p in observed if p['update'] == 4)
        session = OfficialSession(1)
        expected = [session.ingest(g) for g in games[:4]][-1]['public']
        assert received == expected
        assert any(e['decision']['faces'] == [218] for e in session.events)
        assert any(h['face'] == 218 and h['used'] for h in received['hand'])


@pytest.mark.parametrize('dream', ['allow', 'last-resort'])
def test_dream_preserves_other_card_and_target_exclusions(dream):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=99, face=94, used=False, disguised=True),
                 dict(id=100, face=232, used=False, disguised=True)]
    class PreferWide(PreferFirst):
        def __call__(self, env, rows):
            if int(env.stage[0]) == 1 and env.chosen_count[0] > 0 and env.mask[0, 1]:
                return np.array([1], np.int32)
            return super().__call__(env, rows)
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False), PreferWide(), dream=dream)
    assert advice['status'] == 'advice' and advice['dream_played']
    assert advice['action']['cards'] == [dict(item_id=99, model=94)]
    # 94+232 is forbidden by the gfrl selection mask, not by a UI exclusion.
    assert not any(e.get('reason') == 'danger_wide_requires_unobservable_mortar_absence'
                   for e in advice.get('mask_exclusions', []))

    p['hand'] = [dict(id=99, face=6, used=False, disguised=True)]
    class PreferSelf(PreferFirst):
        def __call__(self, env, rows):
            if int(env.stage[0]) == 2 and env.mask[0, 1]:
                return np.array([1], np.int32)
            return super().__call__(env, rows)
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False), PreferSelf(), dream=dream)
    assert advice['status'] == 'advice' and advice['dream_played']
    assert advice['action']['target'] == 2
    assert dict(reason='self_target_of_enemy_default_unobserved', target='self') in advice['mask_exclusions']
