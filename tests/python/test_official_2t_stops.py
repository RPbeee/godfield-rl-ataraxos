"""Synthetic regressions for selection timing and room completion; offline only."""
from copy import deepcopy
import json

import numpy as np
import pytest

from gfrl.official import OfficialDecision
from gfrl.official_bot import Journal, private_directory, run
from gfrl.official_ui import PATHS, UIUnsupported, execute, point, prepare, same_selection
from test_official_bot import (RoomProbe, action, advisor, results, room_clock,
                               settle_clock, visible)


def response_scene():
    p = deepcopy(results()[1]['public'])
    p['pending'].update(models=[97], power=3, element='water', category='weapons')
    p['hand'] = [dict(id=10, face=126, used=False, disguised=False)]
    s = visible(p)
    a = action(p, 'defend', [126])
    return p, s, a


@pytest.mark.parametrize('model, allowed', [(126, True), (127, False)])
def test_flame_boots_against_fog_fan_native_mask(model, allowed):
    p, _, _ = response_scene()
    p['hand'][0]['face'] = model
    env = OfficialDecision(p)
    assert bool(env.mask[0, 0]) == allowed
    if not allowed:
        return
    env.step(np.array([0], np.int32))  # first hand slot, Flame Boots
    assert env.complete
    assert env.semantic_action()['cards'] == [dict(model=126, item_id=10)]


@pytest.mark.parametrize('ignored', [1, 2])
def test_unchanged_selection_retries_once(settle_clock, ignored):
    p, s, a = response_scene()
    plan = prepare(s, p, a, 'SyntheticName')
    click = plan['cards'][0]
    class Probe:
        clicks = 0
        def send(self, command):
            if command == click:
                self.clicks += 1
                if self.clicks > ignored:
                    s['images'].append(dict(path=PATHS[126], x=545, y=103, w=80, h=80))
            return dict(state=deepcopy(s))
    probe = Probe(); commands = []
    if ignored == 2:
        with pytest.raises(UIUnsupported, match='selected_cards_mismatch') as e:
            execute(probe, s, p, a, 'SyntheticName', commands=commands, settle_timeout=1)
        assert len(e.value.detail['clicks']) == 2
        assert commands == [click, click]
    else:
        actual = execute(probe, s, p, a, 'SyntheticName', commands=commands, settle_timeout=1)
        assert actual['selection_retries'] == [0]
        assert commands == [click, click, point(690, 193)]
    assert probe.clicks == 2 and settle_clock.now >= 1


@pytest.mark.parametrize('change', ['hand', 'attack', 'selection', 'pane', 'gf', 'resources', 'arrow'])
def test_changed_selection_never_retried(change):
    p, s, a = response_scene()
    plan = prepare(s, p, a, 'SyntheticName'); altered = deepcopy(s)
    if change == 'hand': altered['images'][0]['x'] += 1
    if change == 'attack': altered['images'][-1]['path'] = PATHS[98]
    if change == 'selection': altered['images'].append(dict(path=PATHS[126], x=545,y=103,w=80,h=80))
    if change == 'pane': altered['texts'][0]['x'] -= 340
    if change == 'gf': altered['body'] = altered['body'].replace(f"G.F.{p['gf']}", 'G.F.999')
    if change == 'resources': altered['body'] = altered['body'].replace('HP', 'XP')
    if change == 'arrow': altered['targetArrow'] = 'left'
    assert not same_selection(altered, s, p, 'SyntheticName', plan, {})


@pytest.mark.parametrize('win', [False, True])
@pytest.mark.parametrize('screen', ['room', 'result', 'transition'])
def test_room_last_game_requires_no_ui_after_finished(tmp_path, room_clock, win, screen):
    class Probe(RoomProbe):
        def send(self, command):
            if self.phase == 'battle' and self.rs[self.index]['status'] == 'finished':
                # The endGame receipt is sufficient, even if the result UI has
                # disappeared or is still animating. No further UI is needed.
                assert command['kind'] == 'capture'
            return super().send(command)
        def state(self):
            if self.phase == 'battle' and self.rs[self.index]['status'] == 'finished':
                return dict(body={'room':'隠れ乱闘 個人戦 チーム戦 終末の時',
                                  'result':'G.F.10 戦いを終わる', 'transition':''}[screen], images=[], texts=[])
            return super().state()
    probe = Probe(tmp_path, start='bot')
    if win:
        final = probe.games[-1]['players']
        final[0]['dead'], final[1]['dead'] = False, True
        final[0]['hp'], final[1]['hp'] = 10, 0
        probe.rs = results(probe.games)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room',
                 password='SyntheticPassword', room_start='bot', games=1)
    journal.close()
    assert report == dict(status='complete', games=1, wins=int(win), losses=int(not win), draws=0)
    rows = [json.loads(x) for x in (directory/'moves.jsonl').read_text().splitlines()]
    assert rows[-1]['status'] == 'finished' and probe.exits == 0


@pytest.mark.parametrize('advance', [False, True])
def test_selection_wait_observes_late_ui_or_new_receive(settle_clock, advance):
    from gfrl.official_bot import ReceiveAdvanced
    p, s, a = response_scene()
    click = prepare(s, p, a, 'SyntheticName')['cards'][0]
    class Probe:
        clicks = 0
        def send(self, command):
            if command == click:
                self.clicks += 1
            if not advance and command['kind'] == 'state' and self.clicks:
                s['images'] = [r for r in s['images'] if not (r['x'] == 545 and r['y'] == 103)]
                s['images'].append(dict(path=PATHS[126], x=545, y=103, w=80, h=80))
            return dict(state=deepcopy(s))
    probe = Probe()
    def refresh():
        if advance and settle_clock.now >= .5:
            raise ReceiveAdvanced(dict(public=dict(update=p['update']+1)))
    if advance:
        with pytest.raises(ReceiveAdvanced):
            execute(probe, s, p, a, 'SyntheticName', settle_timeout=1, refresh=refresh)
    else:
        plan = execute(probe, s, p, a, 'SyntheticName', settle_timeout=1, refresh=refresh)
        assert not plan.get('selection_retries')
    assert probe.clicks == 1


def test_room_returns_automatically_before_next_game(tmp_path, room_clock):
    class Probe(RoomProbe):
        can_return = False
        def send(self, command):
            terminal_capture = (command['kind'] == 'capture' and self.phase == 'battle'
                                and self.rs[self.index]['status'] == 'finished')
            reply = super().send(command)
            if terminal_capture:
                self.can_return = True
            return reply
        def state(self):
            if self.can_return and self.phase == 'battle' and self.rs[self.index]['status'] == 'finished':
                self.phase = 'room'; self.members = ['SyntheticOpponent']
                self.room_reads = 0; self.stream_epoch += 1; self.can_return = False
            return super().state()
    probe = Probe(tmp_path, start='bot')
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room',
                 password='SyntheticPassword', room_start='bot', games=2)
    journal.close()
    assert report['status'] == 'complete' and report['games'] == 2
    assert probe.joins == 2
    assert not any(c.get('text') == '戦いを終わる' for c in probe.commands)
