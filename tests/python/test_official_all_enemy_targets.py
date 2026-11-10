"""Synthetic, offline regressions for match1's all-enemy confirmation."""
from copy import deepcopy

import pytest

from gfrl.official_ui import (
    PATHS, UIUnsupported, execute, point, same_confirmation,
    verify_target,
)
from test_official_bot import action, results, visible


def scene(models=(52, 232)):
    public = deepcopy(results()[2]['public'])
    public['gf'] = 18 if models == (104,) else 10
    public['hand'] = [dict(id=i, face=m, used=False, disguised=False)
                      for i, m in enumerate(models, 99)]
    state = visible(public)
    # Observed card geometry, with synthetic names and public resources only.
    for image, x in zip(state['images'], (764,) if models == (104,) else (518, 846)):
        image.update(x=x, y=493)
    command = action(public, 'play', models)
    return public, state, command


def destination(state, text, x=690):
    state['texts'].append(dict(text=text, x=x-30, y=56, w=60, h=23))
    state['targetArrow'] = 'right' if x == 690 else 'left'


@pytest.mark.parametrize('pane', [350, 690])
@pytest.mark.parametrize('roster_reversed', [False, True])
@pytest.mark.parametrize('display', ['opponent', 'self', 'absent'])
@pytest.mark.parametrize('models', [(52, 232)] + [(m,) for m in (*range(95, 113), *range(224, 231))])
def test_targetless_accepts_opponent_or_absent_band(pane, roster_reversed, display, models):
    p, s, a = scene()
    a['cards'] = [dict(model=m) for m in models]
    original = deepcopy(a)
    if roster_reversed:
        p['players'].reverse()
    s['texts'][0]['x'] = pane-30
    if display != 'absent':
        destination(s, 'SyntheticOpponent' if display == 'opponent' else 'SyntheticName', 1040-pane)
    if display != 'self':
        verify_target(s, p, a, 'SyntheticName', pane)
    else:
        with pytest.raises(UIUnsupported, match='^visible_target_mismatch$') as error:
            verify_target(s, p, a, 'SyntheticName', pane)
        assert error.value.detail['expected_target'] == 'none'
    assert a == original


@pytest.mark.parametrize('models', [(52, 232), (104,)])
@pytest.mark.parametrize('display', ['opponent', 'self', 'absent'])
def test_targetless_execute_confirms_opponent_or_absent(display, models):
    p, s, a = scene(models)
    clicks = [point(r['x']+40, r['y']+40) for r in s['images']]

    class Probe:
        def __init__(self): self.commands = []
        def send(self, command):
            self.commands.append(command)
            if command in clicks:
                i = clicks.index(command)
                s['images'].append(dict(path=PATHS[models[i]], x=205, y=103+100*i, w=80, h=80))
                if i == len(models)-1 and display != 'absent':
                    destination(s, 'SyntheticOpponent' if display == 'opponent' else 'SyntheticName')
            return dict(state=deepcopy(s))

    probe = Probe()
    if display != 'self':
        plan = execute(probe, deepcopy(s), p, a, 'SyntheticName', settle_timeout=0)
        assert probe.commands == clicks + [point(350, 193)]
        assert plan['commands'] == probe.commands
        assert not plan.get('target_switched')
        assert same_confirmation(s, p, a, 'SyntheticName', plan)
        altered = deepcopy(s)
        if display == 'opponent':
            altered['texts'].pop()
        else:
            destination(altered, 'SyntheticOpponent')
        assert same_confirmation(altered, p, a, 'SyntheticName', plan)
        if display == 'opponent':
            destination(altered, 'SyntheticName')
        else:
            altered['texts'][-1]['text'] = 'SyntheticName'
        assert not same_confirmation(altered, p, a, 'SyntheticName', plan)
    else:
        # Duplicate own names are stopped before confirmation.
        with pytest.raises(UIUnsupported, match='^own_name_band_not_unique$'):
            execute(probe, deepcopy(s), p, a, 'SyntheticName', settle_timeout=0)
        assert probe.commands == clicks


def test_other_targetless_actions_accept_both_displays():
    for kind, models in [('play', [52]), ('play', [207]), ('play', [94]),
                         ('pray', []), ('sell', [4, 232])]:
        p, s, a = scene()
        a.update(kind=kind, cards=[dict(model=m) for m in models])
        verify_target(s, p, a, 'SyntheticName', 350)
        destination(s, 'SyntheticOpponent')
        verify_target(s, p, a, 'SyntheticName', 350)
