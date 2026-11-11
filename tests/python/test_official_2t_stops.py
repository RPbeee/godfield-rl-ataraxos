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


@pytest.mark.parametrize('trigger', ['reflections', 'bounces', 'previous', 'counter'])
@pytest.mark.parametrize('changes', [0, 2, None])
def test_redirect_response_settles_before_first_click(settle_clock, trigger, changes):
    p, s, a = response_scene()
    if trigger == 'counter':
        p['pending']['counter'] = True
    elif trigger != 'previous':
        p['pending'][trigger] = 1
    click = prepare(s, p, a, 'SyntheticName')['cards'][0]
    class Probe:
        reads = 0
        clicks = []
        def send(self, command):
            if command['kind'] == 'state':
                self.reads += 1
                if changes is None or self.reads <= changes:
                    s['images'][0]['x'] += 1
            else:
                self.clicks.append((command, self.reads, settle_clock.now))
                if command['y'] > 480:
                    assert changes is not None and self.reads == changes+1
                    assert command['x'] == click['x'] + changes
                    s['images'].append(dict(path=PATHS[126], x=545,y=103,w=80,h=80))
            return dict(state=deepcopy(s))
    probe = Probe()
    if changes is None:
        with pytest.raises(UIUnsupported, match='response_ui_unstable'):
            execute(probe,s,p,a,'SyntheticName',settle_timeout=2,previous_reflect=trigger=='previous')
        assert not probe.clicks and settle_clock.now == 2
    else:
        execute(probe,s,p,a,'SyntheticName',settle_timeout=2,previous_reflect=trigger=='previous')
        assert probe.clicks[0][1:] == (changes+1, .5*(changes+1))


@pytest.mark.parametrize('change', ['verify', 'hand_images', 'own_selected_images',
                                    'other_selected_images', 'band_texts', 'targetArrow'])
def test_refused_selection_retry_reports_comparisons(settle_clock, change):
    p,s,a = response_scene()
    before = deepcopy(s)
    class Probe:
        clicks = 0
        def send(self, c):
            if c['kind'] == 'clickPoint':
                self.clicks += 1
                if change == 'verify': s['body'] = s['body'].replace('HP', 'XP')
                if change == 'hand_images': s['images'][0]['x'] += 1
                if change == 'own_selected_images':
                    s['images'].append(dict(path=PATHS[127],x=545,y=103,w=80,h=80))
                if change == 'other_selected_images': s['images'][-1]['x'] += 1
                if change == 'band_texts': s['texts'].append(dict(text='PRIVATE',x=200,y=65,w=40,h=20))
                if change == 'targetArrow': s['targetArrow'] = 'left'
            return dict(state=deepcopy(s))
    probe=Probe()
    # A verify failure may itself be the terminal code rather than a selection
    # mismatch, so separately exercise its comparison diagnostic below.
    if change == 'verify':
        from gfrl.official_ui import selection_checks
        altered=deepcopy(before); altered['body']=altered['body'].replace('HP','XP')
        detail=selection_checks(altered,before,p,'SyntheticName',prepare(before,p,a,'SyntheticName'),{})
    else:
        with pytest.raises(UIUnsupported, match='selected_cards_mismatch') as caught:
            execute(probe,s,p,a,'SyntheticName',settle_timeout=1)
        assert probe.clicks == 1
        detail=caught.value.detail['selection_check']
    assert detail['checks'][change] is False
    assert 'PRIVATE' not in json.dumps(detail)
    assert all(type(v) is bool for v in detail['checks'].values())


@pytest.mark.parametrize('change', ['notice', 'name', 'arrow', 'description'])
def test_only_transient_response_bands_allow_retry(settle_clock, change):
    p,s,a=response_scene()
    click=prepare(s,p,a,'SyntheticName')['cards'][0]
    class Probe:
        clicks=0
        def send(self,c):
            if c == click:
                self.clicks += 1
                if self.clicks == 1:
                    if change == 'notice': s['texts'].append(dict(text='反射された',x=250,y=300,w=100,h=20))
                    if change == 'name': s['texts'].append(dict(text='反射された',x=250,y=60,w=100,h=20))
                    if change == 'description': s['texts'].append(dict(text='反射された攻撃',x=250,y=300,w=100,h=20))
                    if change == 'arrow': s['targetArrow']='left'
                else:
                    s['images'].append(dict(path=PATHS[126],x=545,y=103,w=80,h=80))
            return dict(state=deepcopy(s))
    probe=Probe()
    if change == 'notice':
        plan=execute(probe,s,p,a,'SyntheticName',settle_timeout=1)
        assert plan['selection_retries'] == [0] and probe.clicks == 2
        assert settle_clock.now == 1.5
    else:
        with pytest.raises(UIUnsupported, match='selected_cards_mismatch'):
            execute(probe,s,p,a,'SyntheticName',settle_timeout=1)
        assert probe.clicks == 1


def test_selection_diagnostic_whitelist():
    from gfrl.official_bot import selection_check_detail
    raw=dict(checks=dict(verify=False,band_texts='PRIVATE',unknown=True),
             counts=dict(band_texts=dict(before=2,after=1,text='PRIVATE'),verify=dict(before=4)))
    assert selection_check_detail(raw) == dict(checks=dict(verify=False),
                                              counts=dict(band_texts=dict(before=2,after=1)))


def test_selection_diagnostics_reach_stop_log(tmp_path, settle_clock, capsys):
    from test_official_bot import FakeProbe
    class Probe(FakeProbe):
        def send(self, c):
            if self.phase == 'battle' and self.index == 0 and c == point(230,533):
                self.commands.append(c)
                self.stale = True
                return dict(state=self.state())
            return super().send(c)
        def state(self):
            s = super().state()
            if self.stale and self.phase == 'battle':
                s['texts'].append(dict(text='PRIVATE',x=200,y=65,w=60,h=20))
            return s
    probe=Probe(tmp_path)
    journal=Journal(private_directory(tmp_path/'log'))
    report=run(probe,advisor,journal,name='SyntheticName',settle_timeout=1)
    journal.close()
    row=json.loads((journal.directory/'moves.jsonl').read_text())
    assert report['reason'] == row['reason'] == 'selected_cards_mismatch'
    assert report['detail']['selection_check'] == row['detail']['selection_check']
    assert row['detail']['selection_check']['checks']['band_texts'] is False
    assert 'PRIVATE' not in (journal.directory/'moves.jsonl').read_text()
    assert 'PRIVATE' not in capsys.readouterr().err
