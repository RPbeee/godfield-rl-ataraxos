"""Synthetic regressions for unobserved UI kinds; no network or live artifacts."""
from copy import deepcopy
import json

import numpy as np
import pytest

from gfrl.model import OBS_FIELDS
from gfrl.official import KINDS, OBSERVED_UI_KINDS, OfficialDecision
from gfrl.official_advisor import advise_supported
from gfrl.official_bot import Journal, private_directory, run
from gfrl.official_ui import UIUnsupported, prepare
from test_official_bot import KindProbe, action, advisor, results, visible
from test_official_live_fixes import PreferFirst


class PreferUnobserved(PreferFirst):
    def __call__(self, env, rows):
        if int(env.stage[0]) == 0:
            for k in (2, 9):
                if env.mask[0, k]:
                    self.modes.append(self.sample)
                    self.masks.append((0, env.mask.copy()))
                    return np.array([k], np.int32)
        return super().__call__(env, rows)


def scene(kind):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=114, face=114, used=False, disguised=False)]
    p['tiebreak_gf'] = p['gf'] if kind == 'sacrifice' else None
    return p


@pytest.mark.parametrize('kind', sorted(set(KINDS) - OBSERVED_UI_KINDS))
@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('search', [False, True])
def test_unobserved_kind_pruned_and_policy_mode_preserved(kind, sample, search):
    p = scene(kind)
    actor = PreferUnobserved(); actor.sample = sample
    class Search:
        def __init__(self):
            self.actor = actor
        def __call__(self, env, rows):
            return actor(env, rows)
    agent = Search() if search else actor
    advice = advise_supported(dict(env=OfficialDecision(p), history_complete=False), agent, search=search)
    assert advice['status'] == 'advice' and advice['action']['kind'] == 'pray'
    assert advice['mask_exclusions'] == [dict(reason=f'ui_kind_unobserved:{kind}', kind=kind)]
    assert 'fallback_samples' not in advice
    assert actor.sample == sample and all(mode == sample for mode in actor.modes)


@pytest.mark.parametrize('kind', ['discard', 'sacrifice'])
def test_kind_mask_preserves_public_cards_observations_and_legality(kind):
    p = scene(kind)
    regular = OfficialDecision(p)
    regular.step(np.array([KINDS.index(kind)], np.int32))
    while not regular.complete:
        regular.step(np.array([np.flatnonzero(regular.mask[0])[0]], np.int32))
    assert regular.action_support() == [f'ui_kind_unobserved:{kind}']
    masked = OfficialDecision(p, excluded_kinds=[kind])
    assert masked.complete and masked.semantic_action()['kind'] == 'pray'
    for key in OBS_FIELDS:
        np.testing.assert_array_equal(getattr(regular, key), getattr(masked, key))
    with pytest.raises(UIUnsupported, match='unobserved_ui_kind') as error:
        prepare(visible(p), p, regular.semantic_action(), 'SyntheticName')
    assert error.value.detail == dict(kind=kind)
    # Excluding every legal kind must fail before forced selection advances.
    with pytest.raises((ValueError, RuntimeError)):
        OfficialDecision(p, excluded_kinds=[kind, 'pray'])


@pytest.mark.parametrize('stop', [False, True])
def test_moves_journal_keeps_kind_exclusions(tmp_path, stop):
    probe = KindProbe(tmp_path, target=None)
    exclusion = dict(reason='ui_kind_unobserved:discard', kind='discard')
    def advising(r):
        if not stop:
            return dict(advisor(r), mask_exclusions=[exclusion])
        return dict(status='unsupported', unsupported=['ui_kind_unobserved:discard'],
                    action=action(r['public'], 'discard', [3]),
                    mask_exclusions=[exclusion], selection_exhausted=True)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advising, journal, name='SyntheticName', dry_run=not stop)
    journal.close()
    rows = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    if stop:
        assert report['reason'] == 'unsupported_advice'
        assert rows[-1]['selection_exhausted']
        assert rows[-1]['mask_exclusions'] == [exclusion]
    else:
        assert report['status'] == 'dry_run'
        assert rows[-1]['advice']['mask_exclusions'] == [exclusion]
