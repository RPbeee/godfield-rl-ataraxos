"""Offline training replay and fail-closed UI tests. Never launch real probe."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest

from gfrl.official import OfficialSession, OfficialStreams, parser_module, project_game
from gfrl.official_ui import PATHS, UIUnsupported, prepare, execute, point, allocation_steps, ready
from gfrl.official_bot import Journal, Probe, private_directory, run

GAMES = json.loads((Path(__file__).parent/'fixtures/official_training.json').read_text())


def firestore(x):
    if isinstance(x, dict):
        return {'mapValue': {'fields': {k: firestore(v) for k, v in x.items()}}}
    if isinstance(x, list):
        return {'arrayValue': {'values': list(map(firestore, x))}}
    if type(x) is bool:
        return {'booleanValue': x}
    if type(x) is int:
        return {'integerValue': str(x)}
    if x is None:
        return {'nullValue': None}
    return {'stringValue': x}


def frame(game):
    body = json.dumps([{'documentChange': {'document': {'name': 'synthetic', 'fields': {'game': firestore(game)}}}}])
    return f"{len(body.encode('utf-16-le'))//2}\n{body}".encode()


def capture_file(tmp_path, games, seq=1, serial=1, tail=b''):
    raw = b''.join(frame(g) for g in games)+tail
    path = tmp_path/f'fake-{serial:06}-stream-{seq:06}-prefix.bin'
    path.write_bytes(raw); path.chmod(0o600)
    return dict(file=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), capture_error=False)


def results(games=GAMES):
    session = OfficialSession(1)
    return [session.ingest(g) for g in games]


def action(public, kind, models=(), target=None, allocation=None):
    hand = list(public['hand']); cards = []
    for model in models:
        h = next(h for h in hand if h['face'] == model and not h['used'])
        hand.remove(h); cards.append(dict(model=model, item_id=h['id']))
    ops = [dict(op='kind', kind=kind)]+[dict(op='select_card', **c) for c in cards]
    ops += [dict(op='finish_cards'), dict(op='target', player_id=target)]
    if allocation:
        ops.append(dict(op='allocate', **allocation))
    ops.append(dict(op='confirm'))
    return dict(kind=kind, cards=cards, target=target, allocation=allocation, operations=ops,
                expected_update=public['update'], expected_gf=public['gf'], viewer=1)


def recorded_action(public):
    return {1: lambda: action(public, 'exchange', [3], allocation=dict(hp=35, mp=35, cp=0)),
            2: lambda: action(public, 'defend'),
            3: lambda: action(public, 'play', [218], target=2),
            4: lambda: action(public, 'defend', [189, 130, 146]),
            5: lambda: action(public, 'defend')}[public['update']]()


# Observed hand coordinates, independently of wire inventory order.
LAYOUT = {3: [(230,533),(312,533),(394,533)], 29: [(476,533)], 48: [(558,533)],
          130: [(640,533)], 146: [(722,533)], 200: [(804,533)], 218: [(230,635)],
          189: [(804,533)], 4: [(394,533)], 5: [(230,635)], 32: [(558,533)], 206: [(886,533)], 191: [(312,635)]}


def visible(public, name='SyntheticName'):
    x = 690 if public['update'] in (2,5,6) else 350
    images = []; counts = {}
    for h in public['hand']:
        if h['used']:
            images.append(dict(path=PATHS[h['face']], x=846, y=595, w=80, h=80))
            continue
        m = h['face']; n = counts.get(m,0); counts[m] = n+1
        coords = LAYOUT.get(m, [(886,635)])
        cx, cy = coords[min(n,len(coords)-1)]
        if public['update'] == 1 and m == 218:
            cx, cy = 886,533
        if public['update'] >= 3 and m == 200:
            cx, cy = 886,533
        images.append(dict(path=PATHS[m], x=cx-40, y=cy-40, w=80, h=80))
    resources = ''.join(f"\nHP\n{p['status']['hp']}\nMP\n{p['status']['mp']}\n¥\n{p['status']['cp']}" for p in public['players'])
    if public['phase'] == 1:
        images += [dict(path=PATHS[m], x=1040-x-145, y=103+100*i, w=80, h=80)
                   for i,m in enumerate(public['pending']['models'])]
    return dict(body=f"修行\nG.F.{public['gf']}\n{name}"+resources, images=images, inputs=[],
                texts=[dict(text=name, x=x-30, y=60, w=60, h=20)])


class FakeProbe:
    viewer = 1
    def __init__(self, directory, *, reject=False, stale=False, corrupt_preview=False, games=GAMES):
        self.directory = directory; self.games = deepcopy(games); self.rs = results(games)
        self.index = 0; self.phase = 'birth'; self.commands = []; self.serial = 0
        self.reject = reject; self.stale = stale; self.corrupt_preview = corrupt_preview
        self.selected = []; self.values = None; self.name = 'SyntheticName'

    def state(self):
        if self.phase == 'birth':
            return dict(body='生誕する', inputs=[{}], images=[])
        if self.phase == 'home':
            return dict(body='コンピュータと対戦\n修行', images=[])
        if self.phase == 'settings':
            return dict(body='修行者 2 人\n終末の時\nなし\n戦いを始める', images=[])
        p = self.rs[self.index]['public']; s = visible(p, self.name)
        x = 690 if p['update'] in (2,5,6) else 350
        s['images'] += [dict(path=path, x=x-145, y=103+100*i,w=80,h=80) for i,path in enumerate(self.selected)]
        if p['update'] == 3 and self.selected:
            s['texts'].append(dict(text='SyntheticOpponent', x=660, y=60, w=60, h=20))
        if self.values:
            # Two status panes followed by the independent exchange preview.
            def values(v):
                return f"HP\n{v['hp']}\nMP\n{v['mp']}\n¥\n{v['cp']}\n"
            s['body'] += '\n'+values(self.values)+'+10\n+1\n-1\n-10'
        if p['is_over']:
            s['body'] += '\n戦いを終わる'
        return s

    def send(self, c):
        self.commands.append(c)
        k = c['kind']
        if k == 'capture':
            self.serial += 1
            if self.phase != 'battle':
                return dict(capture=dict(captures=[]))
            # Reconnect stream overlaps update 2, exposing merger duplicates.
            captures = [capture_file(self.directory,self.games[:min(self.index+1,2)],serial=self.serial)]
            if self.index >= 2:
                captures.append(capture_file(self.directory,self.games[1:self.index+1],seq=2,serial=self.serial))
            return dict(capture=dict(captures=captures))
        if k == 'fillName':
            self.name = c['text']
        elif k == 'clickText':
            text = c['text']
            if text == '戦いを始める':
                self.index = 0
            self.phase = {'生誕する':'home','修行':'settings','戦いを始める':'battle','戦いを終わる':'settings',
                          '戦いをあきらめる':'home'}.get(text,self.phase)
        elif k == 'clickPoint' and not self.stale:
            x,y = c['x'],c['y']
            p = self.rs[self.index]['public']
            if y in (533,635):
                r = next(r for r in self.state()['images'] if r['x']+40==x and r['y']+40==y)
                self.selected.append(r['path'])
                if r['path'] == PATHS[3]:
                    self.values = dict(hp=40,mp=10,cp=20)
            elif x in (620,760) and y in (123,183,303,363) and self.values:
                delta = {123:10,183:1,303:-1,363:-10}[y]
                self.values['mp' if x==620 else 'cp'] += delta
                self.values['hp'] -= delta
                if self.corrupt_preview:
                    self.values['hp'] += 1
            elif y in (193,250) and x in (350,690) and not self.reject:
                self.index += 1; self.selected=[]; self.values=None
        elif k == 'screenshot':
            return dict(screenshot='private-synthetic-screenshot')
        elif k == 'close':
            return dict(closed=True)
        return dict(state=self.state())

    def close(self):
        self.send(dict(kind='close'))


def advisor(result):
    return dict(status='advice', action=recorded_action(result['public']))


def test_stream_merge_sell_history_and_privacy(tmp_path):
    s=OfficialSession(1); streams=OfficialStreams(s)
    a=capture_file(tmp_path,GAMES[:2])
    assert streams.poll([a])['public']['update']==2
    assert len(s.events)==2
    b=capture_file(tmp_path,GAMES[1:],seq=2)
    r=streams.poll([b,a])
    assert r['status']=='finished' and not r['unsupported']
    assert streams.poll([a,b])==r
    sale=next(e for e in s.events if e['decision']['phase']==1)
    assert sale['after']['players'][0]['status']['mp']==20
    assert sale['after']['players'][1]['status']['cp']==35
    assert streams.seen[3]['events'][1]==dict(action='sell',buyer=1,seller=2,face=189)
    changed=deepcopy(GAMES)
    for g in changed:
        g['players'][1]['items'] += [dict(id=99, modelId=209),dict(id=100,modelId=207,fakeModelId=20)]
    t=OfficialStreams(OfficialSession(1))
    t.poll([capture_file(tmp_path,changed,seq=3)])
    assert t.seen==streams.seen
    assert t.session.events==s.events
    # When the other player receives the commodity, drop every item field.
    before=deepcopy(streams.context[3]);before['pending'].update(source=1,target=2)
    g=deepcopy(GAMES[2]);g['events']=g['events'][:2]
    p=project_game(g,1,prior=before)
    g['events'][1]['item']=dict(id=999,modelId=207,fakeModelId=296)
    assert project_game(g,1,prior=before)==p
    assert p['events'][1]==dict(action='sell',buyer=2,seller=1)


def test_stream_unknown_conflict_partial_and_new_game_baseline(tmp_path):
    s=OfficialStreams(OfficialSession(1))
    first=capture_file(tmp_path,GAMES[:1],tail=b'100\n{')
    assert s.poll([first])['status']=='decision'
    bad=deepcopy(GAMES[0]);bad['hp']='unknown-schema'
    r=s.poll([capture_file(tmp_path,[bad],serial=2)])
    assert r['status']=='unsupported' and 'unknown-schema' not in str(r)
    assert s.poll([])['status']=='unsupported'
    old=capture_file(tmp_path,GAMES)
    s=OfficialStreams(OfficialSession(1),[old])
    new=capture_file(tmp_path,GAMES+[GAMES[0]],serial=3)
    assert s.poll([new])['public']['update']==1
    s=OfficialStreams(OfficialSession(1))
    bad=deepcopy(GAMES[0]);bad['gf']=2
    assert s.poll([capture_file(tmp_path,[GAMES[0],bad],serial=4)])['status']=='unsupported'
    # Unknown opponent event stops even while waiting/at terminal.
    bad=deepcopy(GAMES[-1]);bad['events'].append(dict(action='PRIVATE_UNKNOWN'))
    assert OfficialSession(1).ingest(bad)['status']=='unsupported'


def test_recorded_equivalent_clicks_and_operations():
    rs=results()
    for r in rs[:-1]:
        p=r['public'];a=recorded_action(p);plan=prepare(visible(p),p,a,'SyntheticName')
        if p['update']==1:
            assert plan['cards']==[point(230,533)]
            assert plan['confirm']==point(350,250)
            assert plan['allocation'][-1][1]==dict(hp=35,mp=35,cp=0)
            assert len(plan['allocation'])==9
        elif p['update']==3:
            assert plan['commands']==[point(230,635),point(350,193)]
        elif p['update']==4:
            assert plan['commands']==[point(804,533),point(640,533),point(722,533),point(350,193)]
        else:
            assert plan['commands']==[point(690,193)]
        assert point(350,423) not in plan['commands'] and point(690,280) not in plan['commands']
        changed=deepcopy(GAMES[p['update']-1]);changed['players'][1]['items']=[dict(id=123,modelId=207)]
        other=OfficialSession(1)
        for g in GAMES[:p['update']-1]:other.ingest(g)
        pp=other.ingest(changed)['public']
        assert prepare(visible(pp),pp,a,'SyntheticName')==plan


@pytest.mark.parametrize('mutation,reason',[
    (lambda s,p,a:s.pop('texts'), 'own_name_band_not_unique'),
    (lambda s,p,a:s['images'].clear(), 'card_image_missing'),
    (lambda s,p,a:a.update(expected_update=999), 'stale_advice'),
    (lambda s,p,a:a.update(target=99), 'unknown_target'),
    (lambda s,p,a:a['operations'].append(dict(op='guess')), 'unknown_operations'),
    (lambda s,p,a:next(h for h in p['hand'] if h['face']==218).update(disguised=True), 'ambiguous_disguised_face'),
])
def test_ui_validation_stops_before_any_click(mutation,reason):
    p=results()[2]['public'];s=visible(p);a=recorded_action(p)
    mutation(s,p,a)
    with pytest.raises(UIUnsupported,match=reason):prepare(s,p,a,'SyntheticName')


@pytest.mark.parametrize('option,reason',[('stale','selected_cards_mismatch'),('corrupt_preview','exchange_value_mismatch'),('reject','acceptance_timeout')])
def test_bot_stops_saves_and_never_retries_submission(tmp_path,option,reason):
    probe=FakeProbe(tmp_path,**{option:True});journal=Journal(private_directory(tmp_path/'log'))
    r=run(probe,advisor,journal,name='SyntheticName',timeout=.01,settle_timeout=.01)
    journal.close()
    assert r['status']=='stopped' and r['reason']==reason
    assert probe.commands[-1]['kind']=='screenshot'
    submits=[c for c in probe.commands if c==point(350,250)]
    assert len(submits)==(1 if option=='reject' else 0)
    text=(tmp_path/'log/moves.jsonl').read_text()
    assert 'SyntheticName' not in text and 'raw' not in text


def test_bot_birth_start_five_decisions_acceptance_end_and_repeat(tmp_path):
    probe=FakeProbe(tmp_path);directory=private_directory(tmp_path/'log');journal=Journal(directory)
    report=run(probe,advisor,journal,name='SyntheticName')
    journal.close()
    assert report==dict(status='complete',games=1,wins=0,losses=1,draws=0)
    rows=[json.loads(x) for x in (directory/'moves.jsonl').read_text().splitlines()]
    assert [x['received_update'] for x in rows[:-1]]==[2,3,4,5,6]
    assert all(x['accepted'] for x in rows[:-1])
    assert probe.commands[-2]['text']=='戦いを終わる'
    assert stat.S_IMODE(directory.stat().st_mode)==0o700
    assert stat.S_IMODE((directory/'moves.jsonl').stat().st_mode)==0o600
    assert 'SyntheticName' not in (directory/'moves.jsonl').read_text()


def test_two_games_reenter_settings_and_close(tmp_path):
    probe=FakeProbe(tmp_path);journal=Journal(private_directory(tmp_path/'log'))
    report=run(probe,advisor,journal,name='SyntheticName',games=2)
    probe.close();journal.close()
    assert report==dict(status='complete',games=2,wins=0,losses=2,draws=0)
    assert sum(c.get('text')=='戦いを始める' for c in probe.commands)==2
    assert probe.commands[-1]['kind']=='close'


def test_session_sell_duplicate_and_hostile_types():
    s=OfficialSession(1)
    for g in GAMES[:3]:
        result=s.ingest(g)
        assert result['status']!='unsupported'
    assert s.ingest(GAMES[2])['status']=='duplicate'
    for value in (None,[],{'players':[None]}, {'players':[dict(id=1),dict(id=2)],'events':[None]}):
        assert OfficialSession(1).ingest(value)['status']=='unsupported'


def test_dry_run_no_decision_clicks(tmp_path):
    probe=FakeProbe(tmp_path);journal=Journal(private_directory(tmp_path/'log'))
    report=run(probe,advisor,journal,name='SyntheticName',dry_run=True)
    journal.close()
    assert report['status']=='dry_run'
    assert not any(c['kind']=='clickPoint' for c in probe.commands)
    assert probe.index==0


def test_probe_jsonl_close_two_responses_and_failure(tmp_path):
    script=tmp_path/'probe.py'
    script.write_text('''import sys,json
print(json.dumps({'ready':True,'state':{}}),flush=True)
for line in sys.stdin:
 c=json.loads(line)
 if c['kind']=='close':
  print(json.dumps({'capture':{'captures':[]}}),flush=True)
  print(json.dumps({'closed':True}),flush=True)
  break
 print(json.dumps({'state':{'body':'synthetic','images':[]}}),flush=True)
''')
    p=Probe([sys.executable,str(script)])
    assert p.send(dict(kind='state'))['state']['body']=='synthetic'
    p.close();assert p.process.poll() is not None
    script.write_text('import time\ntime.sleep(2)\n')
    with pytest.raises(UIUnsupported,match='probe_failure'):
        Probe([sys.executable,str(script)],timeout=.01)


@pytest.mark.parametrize('cause,reply', [
    ('timeout', None), ('eof', ''),
    ('probe_error', '{"error":"action_failed","raw":"PRIVATE_AUTH"}'),
    ('invalid_json', 'PRIVATE_AUTH {'), ('not_object', '["PRIVATE_AUTH"]'),
])
def test_probe_failure_at_decision_reaches_journal_and_summary(tmp_path, capsys, cause, reply):
    script = tmp_path/'transport.py'
    script.write_text(
        'import sys,time\nprint(\'{"ready":true}\',flush=True)\n'
        'sys.stdin.readline()\n' +
        ('time.sleep(10)\n' if reply is None else
         f'print({reply!r},flush=True)\nsys.exit(7)\n' if reply else 'sys.exit(7)\n'))
    transport = Probe([sys.executable, str(script)])
    transport.timeout = .02
    probe = FakeProbe(tmp_path)
    original_send = probe.send
    armed = False
    def send(command):
        if armed:
            return transport.send(command)
        return original_send(command)
    probe.send = send
    def advising(result):
        nonlocal armed
        armed = True
        return advisor(result)
    directory = private_directory(tmp_path/'log')
    journal = Journal(directory)
    try:
        report = run(probe, advising, journal, name='SyntheticName')
        assert transport.broken
        with pytest.raises(UIUnsupported, match='probe_stopped'):
            transport.send(dict(kind='capture'))
    finally:
        journal.close()
        transport.close()
    row, = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    detail = report['detail']
    assert report['reason'] == row['reason'] == 'probe_failure'
    assert detail == row['detail'] == json.loads(capsys.readouterr().err)['detail']
    assert detail['cause'] == cause and detail['command'] == 'state'
    assert type(detail['elapsed_seconds']) is float and detail['elapsed_seconds'] >= 0
    assert detail['exit_code'] == (-15 if cause == 'timeout' else 7)
    assert set(detail) == {'cause', 'command', 'elapsed_seconds', 'exit_code'} | (
        {'probe_error'} if cause == 'probe_error' else set())
    if cause == 'probe_error':
        assert detail['probe_error'] == 'action_failed'
    assert row['clicks'] == [] and not row['accepted']
    assert 'PRIVATE_AUTH' not in json.dumps(report) + json.dumps(row)


@pytest.mark.parametrize('kind', ['state', 'capture', 'fillName', 'clickText', 'clickPoint', 'screenshot', 'close',
                                 'PRIVATE_AUTH', [], {}, 1, True, None])
def test_probe_failure_queue_tracks_command_and_elapsed(monkeypatch, kind):
    import io
    import queue
    from types import SimpleNamespace
    from gfrl import official_bot as bot
    probe = Probe.__new__(Probe)
    probe.broken = False
    probe.timeout = 40
    probe.lines = queue.Queue()
    probe.lines.put('{"error":"action_failed"}')
    probe.process = SimpleNamespace(stdin=io.StringIO(), wait=lambda **kw: 9, poll=lambda: 9)
    ticks = iter([100., 102.5])
    monkeypatch.setattr(bot.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(UIUnsupported, match='probe_failure') as raised:
        probe.send(dict(kind=kind, text='PRIVATE_AUTH'))
    expected = dict(cause='probe_error', probe_error='action_failed', exit_code=9, elapsed_seconds=2.5)
    if isinstance(kind, str) and kind in bot.PROBE_COMMANDS:
        expected['command'] = kind
    assert raised.value.detail == expected


@pytest.mark.parametrize('code', ['startup_failed', 'a'*64, 'a'*65, 'PRIVATE_AUTH', 'bad-code',
                                 'bad\n', '', None, False, [], {}])
def test_probe_failure_detail_whitelists_at_log_boundary(tmp_path, capsys, code):
    from gfrl.official_bot import probe_failure_detail
    unsafe = dict(cause='probe_error', command='capture', probe_error=code, exit_code=3,
                  elapsed_seconds=1.25, raw='PRIVATE_AUTH', stderr='PRIVATE_AUTH')
    safe = dict(cause='probe_error', command='capture', exit_code=3, elapsed_seconds=1.25)
    if code in ('startup_failed', 'a'*64):
        safe['probe_error'] = code
    assert probe_failure_detail(unsafe) == safe
    probe = FakeProbe(tmp_path)
    def advising(result):
        raise UIUnsupported('probe_failure', unsafe)
    journal = Journal(private_directory(tmp_path/'log'))
    report = run(probe, advising, journal, name='SyntheticName')
    journal.close()
    row = json.loads((journal.directory/'moves.jsonl').read_text())
    assert row['detail'] == report['detail'] == safe
    assert json.loads(capsys.readouterr().err)['detail'] == safe
    assert probe_failure_detail(dict(cause=[], command={}, probe_error=[], exit_code=True,
                                    elapsed_seconds=float('nan'), raw='secret')) == {}
    assert probe_failure_detail(dict(elapsed_seconds=float('inf'))) == {}
    assert probe_failure_detail(dict(elapsed_seconds=-1)) == {}
    assert probe_failure_detail(dict(elapsed_seconds=10**1000)) == {}


def test_probe_failure_exit_wait_is_bounded_without_exit_code():
    import queue
    import subprocess
    from types import SimpleNamespace
    probe = Probe.__new__(Probe)
    probe.timeout = 40
    probe.command = 'capture'
    probe.sent_at = 0
    probe.lines = queue.Queue()
    probe.lines.put(None)
    waits = []
    terminations = []
    def wait(timeout):
        waits.append(timeout)
        raise subprocess.TimeoutExpired('private command', timeout)
    probe.process = SimpleNamespace(wait=wait, poll=lambda: None,
                                    terminate=lambda: terminations.append(True))
    with pytest.raises(UIUnsupported, match='probe_failure') as raised:
        probe.receive()
    assert waits == [.1, .1] and terminations == [True]
    assert probe.broken
    assert set(raised.value.detail) == {'cause', 'command', 'elapsed_seconds'}
    assert raised.value.detail['cause'] == 'eof'


def test_probe_startup_failure_reaches_main_journal_and_summary(tmp_path, monkeypatch, capsys):
    from gfrl import official_bot as bot, evaluation
    script = tmp_path/'startup.py'
    script.write_text('print(\'{"error":"startup_failed","raw":"PRIVATE_AUTH"}\',flush=True)\nraise SystemExit(4)\n')
    real_probe = Probe
    monkeypatch.setattr(bot, 'Probe', lambda *a, **kw: real_probe([sys.executable, str(script)]))
    monkeypatch.setattr(evaluation, 'load_policy_model', lambda *a: None)
    out = tmp_path/'log'
    assert bot.main(['--checkpoint', 'unused', '--playwright', 'unused', '--out-dir', str(out)]) == 1
    streams = capsys.readouterr()
    report = json.loads(streams.out)
    row = json.loads((out/'moves.jsonl').read_text())
    detail = row['detail']
    assert detail == report['detail'] == json.loads(streams.err)['detail']
    assert report['reason'] == row['reason'] == 'probe_failure'
    assert set(detail) == {'cause', 'probe_error', 'exit_code', 'elapsed_seconds'}
    assert detail['cause'] == 'probe_error' and detail['probe_error'] == 'startup_failed'
    assert detail['exit_code'] == 4
    assert 'PRIVATE_AUTH' not in streams.out + streams.err + json.dumps(row)


def test_parser_sell_exact_shape_and_unknown_remains_rejected():
    parser=parser_module();g=deepcopy(GAMES[0]);g['events']=[dict(action='sell',item=dict(id=7,modelId=189))]
    assert parser.parse_prefix(frame(g),1)['games'][0]['game']['events']==g['events']
    for event in [dict(action='sell'),dict(action='sell',item=dict(modelId=True)),
                  dict(action='sell',item=dict(modelId=189),playerId=1),dict(action='unknown')]:
        g['events']=[event]
        with pytest.raises(parser.ParseError):parser.parse_prefix(frame(g),1)


@pytest.fixture
def settle_clock(monkeypatch):
    # Exercise polling and deadlines without wall-clock sleeps or a browser.
    import gfrl.official_ui as ui
    class Clock:
        now = 0.
        sleeps = []
        def monotonic(self):
            return self.now
        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds
    clock = Clock()
    monkeypatch.setattr(ui.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(ui.time, 'sleep', clock.sleep)
    return clock


class LagProbe(FakeProbe):
    def __init__(self, directory, *, forever=False, advance=False, click_lag=False,
                 advance_after_click=False, mutation=None):
        super().__init__(directory)
        self.forever = forever; self.advance = advance; self.click_lag = click_lag
        self.advance_after_click = advance_after_click; self.mutation = mutation
        self.reads = 0; self.pending_state = None; self.pending_reads = 0

    def send(self, c):
        if c['kind'] == 'state' and self.phase == 'battle':
            self.reads += 1
            if self.advance and self.reads == 2:
                self.index = 1
            if self.pending_reads:
                self.commands.append(c); self.pending_reads -= 1
                return dict(state=deepcopy(self.pending_state))
            if self.index == 0 and not self.selected and (self.forever or self.reads <= 3):
                self.commands.append(c)
                s = self.state()
                if self.mutation:
                    self.mutation(s)
                else:
                    s.update(body=s['body'].replace('G.F.1', 'G.F.0')+
                             '\n預言者たちの戦いが\n今始まる！', texts=[], images=[])
                return dict(state=s)
        if c['kind'] == 'clickPoint' and c['y'] not in (193,250):
            before = deepcopy(self.state())
            result = super().send(c)
            if self.advance_after_click:
                self.advance_after_click = False
                self.index = 1; self.selected = []; self.values = None
            if self.click_lag:
                self.pending_state = before; self.pending_reads = 2
                return dict(state=before)
            return result
        if c['kind'] == 'screenshot':
            self.commands.append(c)
            path = self.directory/'synthetic-stop.png'
            path.write_bytes(b'synthetic screenshot')
            return dict(screenshot=str(path))
        return super().send(c)


def test_bot_waits_for_opening_and_each_click_without_reclicking(tmp_path, settle_clock):
    probe = LagProbe(tmp_path, click_lag=True)
    journal = Journal(private_directory(tmp_path/'log'))
    report = run(probe, advisor, journal, name='SyntheticName')
    journal.close()
    assert report == dict(status='complete', games=1, wins=0, losses=1, draws=0)
    expected = [c for r in results()[:-1]
                for c in prepare(visible(r['public']), r['public'],
                                 recorded_action(r['public']), 'SyntheticName')['commands']]
    assert [c for c in probe.commands if c['kind']=='clickPoint'] == expected
    # The initial navigation poll consumes the first of three opening states.
    assert len(settle_clock.sleeps) == 2 + 3*14


def test_bot_settle_timeout_saves_screenshot_without_clicks(tmp_path, settle_clock):
    probe = LagProbe(tmp_path, forever=True)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', settle_timeout=1.2)
    journal.close()
    assert report['status'] == 'stopped' and report['reason'] == 'visible_gf_changed'
    assert settle_clock.now == pytest.approx(1.2)
    assert not any(c['kind']=='clickPoint' for c in probe.commands)
    assert (directory/'stopped-screenshot.png').read_bytes() == b'synthetic screenshot'
    assert stat.S_IMODE((directory/'stopped-screenshot.png').stat().st_mode) == 0o600


@pytest.mark.parametrize('after_click', [False, True])
def test_bot_redecides_when_receive_advances_while_settling(tmp_path, settle_clock, after_click):
    probe = LagProbe(tmp_path, advance=not after_click, advance_after_click=after_click)
    journal = Journal(private_directory(tmp_path/'log')); updates = []
    def advising(result):
        updates.append(result['public']['update'])
        return advisor(result)
    report = run(probe, advising, journal, name='SyntheticName')
    journal.close()
    assert report['status'] == 'complete'
    assert updates == [1,2,3,4,5]
    assert point(350,250) not in probe.commands
    rows = [json.loads(line) for line in (tmp_path/'log/moves.jsonl').read_text().splitlines()]
    assert rows[0]['status'] == 'superseded' and rows[0]['received_update'] == 2
    assert rows[0]['clicks'] == ([point(230,533)] if after_click else [])


@pytest.mark.parametrize('mutation,reason', [
    (lambda s: s.update(texts=[]), 'own_name_band_not_unique'),
    (lambda s: s['images'].pop(), 'visible_hand_mismatch'),
    (lambda s: s.update(body=s['body']+'\n今始まる！'), 'visible_animation'),
    (lambda s: s.update(body=s['body'].replace('HP\n40', 'HP\n39')), 'visible_resources_mismatch'),
])
def test_bot_waits_for_all_scene_prerequisites(tmp_path, settle_clock, mutation, reason):
    p = results()[0]['public']; state = visible(p); mutation(state)
    with pytest.raises(UIUnsupported, match=reason):
        ready(state, p, 'SyntheticName')
    probe = LagProbe(tmp_path, mutation=mutation)
    journal = Journal(private_directory(tmp_path/'log'))
    report = run(probe, advisor, journal, name='SyntheticName', dry_run=True)
    journal.close()
    assert report['status'] == 'dry_run' and settle_clock.now == 1.
    assert not any(c['kind']=='clickPoint' for c in probe.commands)


def test_response_waits_for_opponent_attack_cards(tmp_path, settle_clock):
    from gfrl.official_ui import wait_visible
    p = results()[1]['public']; state = visible(p)
    state['images'] = [r for r in state['images'] if r['y'] != 103]
    with pytest.raises(UIUnsupported, match='visible_attack_mismatch'):
        ready(state, p, 'SyntheticName')
    probe = FakeProbe(tmp_path); probe.phase = 'battle'; probe.index = 1
    settled = wait_visible(probe, state, lambda s: ready(s,p,'SyntheticName'))
    assert settled == probe.state() and settle_clock.now == .5


class KindProbe(FakeProbe):
    def __init__(self, directory, *, target, mismatch=False, **kwargs):
        super().__init__(directory, **kwargs)
        self.target = target; self.mismatch = mismatch
        self.target_checks = []

    def state(self):
        s = super().state()
        if self.phase == 'battle' and self.index == 2:
            s['texts'] = [r for r in s['texts'] if r['text'] != 'SyntheticOpponent']
            if (self.selected or self.mismatch) and (self.target == 2) != self.mismatch:
                s['texts'].append(dict(text='SyntheticOpponent', x=660, y=60, w=60, h=20))
        return s

    def send(self, c):
        if self.phase == 'battle' and self.index == 2 and c == point(350,193):
            self.target_checks.append(deepcopy(self.state()['texts']))
        if c['kind'] == 'screenshot':
            self.commands.append(c)
            path = self.directory/'synthetic-stop.png'
            path.write_bytes(b'synthetic screenshot')
            return dict(screenshot=str(path))
        return super().send(c)


@pytest.mark.parametrize('kind,models,target', [
    ('buy', [5], 2), ('sell', [4,29], 2), ('pray', [], None),
    ('play', [191], 1), ('play', [191], None),
])
@pytest.mark.parametrize('mismatch', [False, True])
def test_new_kinds_select_target_submit_and_accept(tmp_path, kind, models, target, mismatch):
    games = deepcopy(GAMES)
    # Replace one visible hand card in the third decision, leaving the capture
    # and accepted-update path exercised by the real stream/session adapter.
    if models:
        next(h for h in games[2]['players'][0]['items'] if h['modelId']==218)['modelId'] = models[0]
    probe = KindProbe(tmp_path, games=games, target=target, mismatch=mismatch)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    def advising(r):
        if r['public']['update'] == 3:
            return dict(status='advice', action=action(r['public'],kind,models,target))
        return advisor(r)
    report = run(probe, advising, journal, name='SyntheticName', settle_timeout=.01)
    journal.close()
    rows = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r.get('state',{}).get('update') == 3)
    p = probe.rs[2]['public']; s = visible(p)
    a = action(p,kind,models,target)
    plan = prepare(s,p,a,'SyntheticName')
    if mismatch and target is not None:
        assert report['reason'] == ('visible_target_mismatch' if target is None else 'target_switch_failed')
        assert row['action'] == {k:a[k] for k in ('kind','cards','target')}
        assert row['detail']['expected_target'] in ('self','opponent','none')
        assert row['detail']['bands']
        assert row['clicks'] == plan['cards']
        assert not row['accepted'] and 'first_success_kind' not in row
        assert probe.target_checks == []
        saved = json.loads((directory/'stopped-state.json').read_text())
        assert saved == probe.state() and saved['images'] and saved['texts']
        assert (directory/'stopped-screenshot.png').read_bytes() == b'synthetic screenshot'
        assert stat.S_IMODE((directory/'stopped-state.json').stat().st_mode) == 0o600
    else:
        assert report['status'] == 'complete'
        assert row['clicks'] == plan['commands']
        assert row['accepted'] and row['received_update'] == 4
        assert row['first_success_kind'] == kind
        assert len(probe.target_checks) == 1
        assert len(row['clicks']) == len(models)+1


@pytest.mark.parametrize('kind', ['accept','decline','discard','unknown'])
def test_observation_stop_saves_state_and_image(tmp_path, kind):
    probe = KindProbe(tmp_path, target=None)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    def advising(r):
        return dict(status='advice', action=action(r['public'],kind))
    report = run(probe, advising, journal, name='SyntheticName')
    journal.close()
    assert report['reason'] == ('ui_phase_mismatch' if kind in ('accept','decline') else 'unobserved_ui_kind')
    assert not any(c['kind']=='clickPoint' for c in probe.commands)
    assert json.loads((directory/'stopped-state.json').read_text()) == probe.state()
    assert (directory/'stopped-screenshot.png').is_file()
    assert 'SyntheticName' not in (directory/'moves.jsonl').read_text()


def test_first_success_kind_once_across_games(tmp_path):
    probe = FakeProbe(tmp_path); directory = private_directory(tmp_path/'log')
    journal = Journal(directory)
    report = run(probe,advisor,journal,name='SyntheticName',games=2)
    journal.close()
    assert report['status'] == 'complete'
    rows = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    assert [r['first_success_kind'] for r in rows if 'first_success_kind' in r] == ['exchange','defend','play']


# Actual stopped-state geometry and text, with names replaced. Public resources,
# hand faces/order/used and pending models are transcribed from the accompanying
# moves.jsonl board; unselected item IDs are synthetic. No private capture needed.
UI_STOPS = json.loads((Path(__file__).parent/'fixtures/official_ui_stops.json').read_text())


@pytest.mark.parametrize('record', ['auto3', 'auto4', 'auto5', 'auto6'])
def test_recorded_false_stops_are_ready_and_selectable(record):
    r = deepcopy(UI_STOPS[record])
    s, p, a = r['state'], r['public'], r['action']
    ready(s, p, 'SyntheticName')
    plan = prepare(s, p, a, 'SyntheticName')
    assert plan['cards'] == {'auto3': [point(230,533)], 'auto4': [point(722,533)],
                             'auto5': [], 'auto6': [point(886,635)]}[record]
    if record == 'auto3':
        assert any(i['path'] == '/images/curses/small/cold.webp' for i in s['images'])
    # The same scene check also runs after each card selection.
    from gfrl.official_ui import verify
    verify(s, p, 'SyntheticName', plan, {})


@pytest.mark.parametrize('phrase', ['ダメージで夢', 'ダメージで地獄病', '反射した', '反射された', '跳ね返', 'の攻撃！'])
def test_card_description_words_do_not_identify_animation(phrase):
    r = deepcopy(UI_STOPS['auto4'])
    r['state']['body'] += '\n'+phrase
    ready(r['state'], r['public'], 'SyntheticName')


@pytest.mark.parametrize('effect', ['confused', 'hidden_status'])
@pytest.mark.parametrize('player', [0, 1])
def test_screen_changing_effects_still_stop(effect, player):
    r = deepcopy(UI_STOPS['auto3']); p = r['public']['players'][player]
    if effect == 'confused':
        p['status']['confused'] = True
    elif effect == 'fog':
        p['status']['curses'].append('fog')
    else:
        p.update(status=None, fogged=True)
    with pytest.raises(UIUnsupported, match='unobserved_screen_effect'):
        prepare(r['state'], r['public'], r['action'], 'SyntheticName')


@pytest.mark.parametrize('curse', ['cold', 'fever', 'hell', 'heaven', 'dream', 'flash', 'darkcloud'])
def test_small_status_icons_do_not_affect_card_detection(curse):
    r = deepcopy(UI_STOPS['auto3'])
    for p in r['public']['players']:
        p['status']['curses'] = [curse]
    r['state']['images'].append(dict(path=f'/images/curses/small/{curse}.webp', x=1054, y=155, w=30, h=16))
    ready(r['state'], r['public'], 'SyntheticName')
    assert prepare(r['state'], r['public'], r['action'], 'SyntheticName')['cards'] == [point(230,533)]


@pytest.mark.parametrize('use_used', [False, True])
def test_same_model_unused_and_used_choose_matching_item(use_used):
    r = deepcopy(UI_STOPS['auto6']); s,p,a = r['state'],r['public'],r['action']
    p['hand'].append(dict(id=999, face=218, used=False, disguised=False))
    s['images'].append(dict(path=PATHS[218], x=190, y=595, w=80, h=80))
    if not use_used:
        a['cards'][0]['item_id'] = 999
        a['operations'][1]['item_id'] = 999
    ready(s,p,'SyntheticName')
    assert prepare(s,p,a,'SyntheticName')['cards'] == [point(886 if use_used else 230,635)]
    # Matching image totals alone cannot establish which copy is used.
    next(i for i in s['images'] if i['x']==846 and i['y']==595)['x'] = 272
    ready(s,p,'SyntheticName')
    with pytest.raises(UIUnsupported, match='ambiguous_used_card_position'):
        prepare(s,p,a,'SyntheticName')


def test_used_miracle_must_still_be_visible():
    r = deepcopy(UI_STOPS['auto6'])
    r['state']['images'] = [i for i in r['state']['images'] if i['y'] != 595]
    with pytest.raises(UIUnsupported, match='visible_hand_mismatch'):
        ready(r['state'],r['public'],'SyntheticName')


@pytest.mark.parametrize('update', ['4', '5'])
def test_live1_used_waterfall_remains_in_hand(update):
    # Observed resp offsets 48/58 and the corresponding projected capture.
    # Offset 48's own selected pane is removed, as in LiveReplayProbe.
    fixtures = json.loads((Path(__file__).parent/'fixtures/official_ui_live_used.json').read_text())
    s,p = fixtures[update]['state'],fixtures[update]['public']
    ready(s,p,'SyntheticName')
    h = next(h for h in p['hand'] if h['face']==218)
    assert h['used']
    from gfrl.official_ui import hand_images
    assert next(i for i in hand_images(s) if i['path']==PATHS[218]) == dict(
        path=PATHS[218], x=846, y=595, w=80, h=80)


def test_used_waterfall_click_is_verified_before_confirm():
    r = deepcopy(UI_STOPS['auto6']); s,p,a = r['state'],r['public'],r['action']
    class UsedProbe:
        def __init__(self):
            self.commands = []
        def send(self, command):
            self.commands.append(command)
            if command == point(886,635):
                s['images'].append(dict(path=PATHS[218], x=205, y=103, w=80, h=80))
                s['texts'].append(dict(text='SyntheticOpponent',x=660,y=60,w=60,h=20))
            return dict(state=s)
    probe = UsedProbe()
    execute(probe,s,p,a,'SyntheticName',settle_timeout=0)
    assert probe.commands == [point(886,635),point(350,193)]


def test_multiple_used_miracles_with_mixed_face_stop_without_guessing():
    r = deepcopy(UI_STOPS['auto6']); s,p,a = r['state'],r['public'],r['action']
    p['hand'] += [dict(id=998,face=218,used=True,disguised=False),
                  dict(id=999,face=218,used=False,disguised=False)]
    s['images'] += [dict(path=PATHS[218],x=x,y=595,w=80,h=80) for x in (190,764)]
    ready(s,p,'SyntheticName')
    with pytest.raises(UIUnsupported, match='ambiguous_used_card_position'):
        prepare(s,p,a,'SyntheticName')


def test_receive_parser_details_and_hidden_inventory_invariance(tmp_path):
    details = []
    for model in (30, 296):
        g = deepcopy(GAMES[0])
        g['players'][0]['name'] = 'SyntheticName'
        g['players'][1]['name'] = 'SyntheticOpponent'
        g['players'][1]['items'] = [dict(id=987654321, modelId=model, used=False)]
        g['attacks'] = [dict(itemModelIds=[5], buyingItemModelId=987654321)]
        streams = OfficialStreams(OfficialSession(1))
        result = streams.poll([capture_file(tmp_path, [g], seq=model)])
        assert result['status'] == 'unsupported'
        details.append(result['detail'])
        assert streams.poll([])['detail'] == result['detail']
    assert details[0] == details[1] == dict(error='ParseError', code='unsupported_game_schema',
        parser=dict(path='game.attacks[].buyingItemModelId', value_type='int'))
    assert all(s not in json.dumps(details) for s in ('SyntheticName', 'SyntheticOpponent', '987654321'))


def test_stream_fixed_codes_and_unexpected_exception_location(tmp_path, monkeypatch):
    for capture, code in [([dict(capture_error='SyntheticName')], 'capture_failed'),
                          ([dict(capture_error=False, file=str(tmp_path/'missing'))], None)]:
        r = OfficialStreams(OfficialSession(1)).poll(capture)
        assert r['status'] == 'unsupported'
        if code:
            assert r['detail'] == dict(error='ValueError', code=code)
        else:
            assert r['detail']['error'] == 'FileNotFoundError'
            assert r['detail']['file'] == 'official.py'
            assert isinstance(r['detail']['line'], int)
        assert 'SyntheticName' not in json.dumps(r)
    c = capture_file(tmp_path, GAMES[:1]); c['bytes'] += 1
    assert OfficialStreams(OfficialSession(1)).poll([c])['detail']['code'] == 'capture_integrity'
    c = capture_file(tmp_path, GAMES[:1])
    g = deepcopy(GAMES[0]); g['players'][0]['hp'] -= 1
    streams = OfficialStreams(OfficialSession(1)); streams.poll([c])
    assert streams.poll([capture_file(tmp_path, [g], seq=2)])['detail']['code'] == 'conflicting_update'
    streams = OfficialStreams(OfficialSession(1))
    def broken(*args):
        raise KeyError('SyntheticName PRIVATE_AUTH')
    monkeypatch.setattr(streams.parser, 'parse_prefix', broken)
    d = streams.poll([c])['detail']
    assert d['error'] == 'KeyError' and d['file'] == 'official.py' and isinstance(d['line'], int)
    assert 'SyntheticName' not in json.dumps(d)


@pytest.mark.parametrize('case', ['unknown_event', 'ambiguous_pending_actor', 'dynamic_attack_value_unavailable',
                                  'earth_moon_payload', 'multiple_pending_frames', 'sale_recipient_unavailable'])
def test_projection_unsupported_details_do_not_depend_on_unused_foreign_cards(case):
    records = []
    for model in (30, 296):
        g = deepcopy(GAMES[0]); g['events'] = []
        g['players'][1]['items'] = [dict(id=999, modelId=model, used=False)]
        attack = dict(playerId=2, targetPlayerId=1, itemModelIds=[58])
        if case == 'unknown_event':
            g['events'] = [dict(action='cold')]
        elif case == 'sale_recipient_unavailable':
            g['events'] = [dict(action='sell', item=dict(modelId=30))]
        else:
            if case == 'ambiguous_pending_actor': attack['targetPlayerId'] = 0
            if case == 'earth_moon_payload': attack.update(itemModelIds=[30], guardian='moon')
            g['attacks'] = [attack] * (2 if case == 'multiple_pending_frames' else 1)
        r = OfficialSession(1).ingest(g)
        assert r['status'] == 'unsupported'
        if case == 'multiple_pending_frames':
            # The queue itself is supported; missing head-58 launch MP still stops.
            assert r['unsupported'] == ['dynamic_attack_value_unavailable']
        else:
            assert case in r['unsupported']
        records.append(r['detail'])
    assert records[0] == records[1]
    fact = next(d for d in records[0] if d['code'] == case)
    if case == 'unknown_event': assert fact['action'] == 'cold'
    if case == 'ambiguous_pending_actor': assert fact['target_missing'] is True
    if case == 'dynamic_attack_value_unavailable': assert fact['models'] == [58]


@pytest.mark.parametrize('failure', ['hand', 'attack', 'resources', 'own_band', 'target_band', 'preview'])
def test_ui_safe_details(failure):
    from gfrl.official_ui import verify_resources, pane_x, verify_target, preview
    p = results()[1]['public']; s = visible(p)
    if failure == 'hand':
        s['images'] = [i for i in s['images'] if i['y'] == 103]
        check = lambda: ready(s, p, 'SyntheticName')
    elif failure == 'attack':
        s['images'] = [i for i in s['images'] if i['y'] != 103]
        check = lambda: ready(s, p, 'SyntheticName')
    elif failure == 'resources':
        s['body'] = s['body'].replace('HP\n40', 'HP\n39')
        check = lambda: verify_resources(s, p)
    elif failure == 'own_band':
        s['texts'] = []
        check = lambda: pane_x(s, 'SyntheticName')
    elif failure == 'target_band':
        s['texts'].extend([dict(text='SyntheticOpponent', x=660,y=60,w=60,h=20)]*2)
        check = lambda: verify_target(s,p,dict(target=2),'SyntheticName',690)
    else:
        check = lambda: preview(s)
    with pytest.raises(UIUnsupported) as caught: check()
    d = caught.value.detail
    assert d
    if failure in ('hand','attack','resources','preview'):
        assert 'expected' in d and 'actual' in d
    else:
        assert d['count'] == len(d['x'])
    assert caught.value.args == (str(caught.value),)
    assert all(n not in json.dumps(d) for n in ('SyntheticName','SyntheticOpponent'))


def test_selected_mismatch_has_each_attempted_path_and_coordinates():
    p = results()[3]['public']; s = visible(p); a = recorded_action(p)
    plan = prepare(s,p,a,'SyntheticName')
    class MissingSecondCard:
        def __init__(self): self.clicks = 0
        def send(self, c):
            self.clicks += 1
            if self.clicks == 1:
                s['images'].append(dict(path=PATHS[189], x=plan['pane']-145, y=103,w=80,h=80))
            return dict(state=s)
    with pytest.raises(UIUnsupported, match='selected_cards_mismatch') as caught:
        execute(MissingSecondCard(),s,p,a,'SyntheticName',settle_timeout=0)
    d = caught.value.detail
    assert d['expected'] == {PATHS[189]:1, PATHS[130]:1}
    assert d['actual'] == {PATHS[189]:1}
    assert d['clicks'] == [dict(path=PATHS[m], x=c['x'], y=c['y'], appeared=(i==0))
                          for i,(m,c) in enumerate(zip([189,130],plan['cards']))]


@pytest.mark.parametrize('failure', ['receive', 'advice', 'exception', 'ui'])
def test_stopped_rows_and_stderr_contain_safe_details(tmp_path, capsys, failure):
    probe = FakeProbe(tmp_path, stale=failure=='ui')
    if failure == 'receive':
        probe.games = deepcopy(probe.games)
        probe.games[0]['attacks'] = [dict(buyingItemModelId=987654321)]
    def advising(r):
        if failure == 'exception': raise KeyError('SyntheticName SyntheticOpponent PRIVATE_AUTH')
        if failure == 'advice':
            # An unexpected advisor payload must not be journalled wholesale.
            return dict(status='unsupported', board='SyntheticName', detail=dict(raw='SyntheticOpponent'))
        return advisor(r)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advising, journal, name='SyntheticName', settle_timeout=0)
    journal.close()
    assert report['status'] == 'stopped'
    rows = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    stopped = [r for r in rows if r['status']=='stopped']
    assert len(stopped) == 1 and 'detail' in stopped[0]
    stderr = capsys.readouterr().err
    assert len(stderr.splitlines()) == 1
    assert json.loads(stderr) == {k: stopped[0][k] for k in ('reason','detail')}
    for secret in ('SyntheticName', 'SyntheticOpponent', 'PRIVATE_AUTH', '987654321'):
        assert secret not in json.dumps(stopped) and secret not in stderr
    if failure == 'receive': assert stopped[0]['detail']['parser'] == dict(path='game.attacks[].buyingItemModelId', value_type='int')
    if failure == 'exception':
        assert stopped[0]['detail']['error'] == 'KeyError'
        assert stopped[0]['detail']['file'] == 'test_official_bot.py'
    if failure == 'ui': assert stopped[0]['detail']['clicks'][0]['appeared'] is False


def test_viewer_inference_parser_failure_is_also_safe(tmp_path, capsys):
    probe = FakeProbe(tmp_path)
    probe.viewer = None
    probe.games = deepcopy(probe.games)
    probe.games[0]['attacks'] = [dict(buyingItemModelId=987654321)]
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName')
    journal.close()
    row = json.loads((directory/'moves.jsonl').read_text())
    assert report['reason'] == 'unsupported_receive'
    assert row['detail']['parser'] == dict(path='game.attacks[].buyingItemModelId', value_type='int')
    assert 'SyntheticName' not in json.dumps(row)
    assert json.loads(capsys.readouterr().err)['detail'] == row['detail']


def test_exchange_read_values_are_attached(tmp_path):
    p = results()[0]['public']; s = visible(p)
    class BadPreview:
        def send(self, c):
            if c['y'] == 533:
                s['images'].append(dict(path=PATHS[3], x=205, y=103, w=80, h=80))
                s['body'] += '\nHP 39 MP 10 ¥ 20\n+10'
            return dict(state=s)
    with pytest.raises(UIUnsupported, match='exchange_initial_mismatch') as caught:
        execute(BadPreview(),s,p,recorded_action(p),'SyntheticName',settle_timeout=0)
    assert caught.value.detail == dict(expected=dict(hp=40, mp=10, cp=20),
                                       actual=dict(hp=39, mp=10, cp=20))


class RoomProbe(FakeProbe):
    def __init__(self, directory, *, start='human', members=1, duplicate=False,
                 stuck_end=False, match_names=1, game_players=2, end_change=False):
        games = deepcopy(GAMES)
        for g in games:
            for p in g['players']:
                p['isBot'] = False
                p['name'] = 'SyntheticName' if p['id'] == 1 else 'SyntheticOpponent'
            if match_names == 0:
                g['players'][0]['name'] = 'SyntheticOther'
            elif match_names == 2:
                g['players'][1]['name'] = 'SyntheticName'
            if game_players == 3:
                g['players'].append(dict(g['players'][1], id=3, name='SyntheticThird'))
        super().__init__(directory, games=games)
        if game_players == 3:
            self.rs = results(GAMES)  # visible fake remains a two-pane scene
        self.start = start; self.members = ['SyntheticOpponent'] * members
        self.duplicate = duplicate; self.stuck_end = stuck_end
        self.end_change = end_change
        self.end = 'G.F.100'; self.menu = False; self.room_reads = 0
        self.joins = 0; self.exits = 0
        self.stream_epoch = 0

    def state(self):
        if self.phase == 'home':
            return dict(body='修行\n隠れ乱闘\n真剣タイマン', inputs=[], images=[], texts=[])
        if self.phase == 'password':
            return dict(body='部屋の合言葉\n唱える', inputs=[dict(value='SyntheticPassword')], images=[], texts=[])
        if self.phase == 'room':
            own = 'SyntheticName' in self.members
            self.room_reads += int(own)
            if self.end_change and self.room_reads > 1:
                self.end = 'G.F.50'
            if self.start == 'human' and own and self.room_reads >= 4:
                self.phase = 'battle'; self.index = 0
                return self.state()
            body = '隠れ乱闘\n教典\nSyntheticName\n全体にお告げ\nお告げの記録\n個人戦\nチーム戦\n終末の時\n'+self.end+'\nチームをシャッフル\n戦いを始める'
            if self.menu: body += '\nなし\nG.F.1\nG.F.50\nG.F.75\nG.F.100\nG.F.150'
            texts = [dict(text=n,x=600,y=143+35*i,w=80,h=20) for i,n in enumerate(self.members)]
            # Occupants and header contain repeated own names, outside slots.
            texts += [dict(text='SyntheticName',x=1190,y=676,w=80,h=20),
                      dict(text='SyntheticName',x=650,y=10,w=80,h=20)]
            body += '\n'+'\n'.join(self.members)+'\nSyntheticName'
            return dict(body=body, texts=texts, inputs=[], images=[])
        return super().state()

    def send(self, c):
        k = c['kind']; text = c.get('text')
        if k == 'capture':
            reply = super().send(c)
            if self.stream_epoch:
                for entry in reply['capture']['captures']:
                    path = Path(entry['file'])
                    # Each synthetic game opens fresh streams, just as a
                    # reconnect may; never shrink an existing stream prefix.
                    target = path.with_name(path.name.replace('-stream-00000', f'-stream-{self.stream_epoch:05}'))
                    path.rename(target); entry['file'] = str(target)
            return reply
        if k == 'screenshot':
            self.commands.append(c)
            path = self.directory/'synthetic-stop.png'; path.write_bytes(b'synthetic screenshot')
            return dict(screenshot=str(path))
        if (k == 'clickText' and text in ('隠れ乱闘','唱える','終末の時','なし','戦いを終わる')) or (
                k == 'clickPoint' and (c['x'],c['y']) in ((400,173),(240,23))):
            self.commands.append(c)
            if text == '隠れ乱闘': self.phase = 'password'
            elif text == '唱える': self.phase = 'room'
            elif text == '終末の時': self.menu = True
            elif text == 'なし':
                self.menu = False
                if not self.stuck_end: self.end = 'なし'
            elif text == '戦いを終わる':
                self.phase = 'room'; self.members = ['SyntheticOpponent']; self.room_reads = 0
                self.stream_epoch += 1
            elif c.get('x') == 400:
                assert 'SyntheticName' not in self.members
                self.members.append('SyntheticName'); self.joins += 1
                if self.duplicate: self.members.append('SyntheticName')
            elif c.get('x') == 240:
                self.exits += 1; self.phase = 'home'
            return dict(state=self.state())
        if k == 'fillName' and self.phase == 'password':
            self.commands.append(c)
            assert c['text'] == 'SyntheticPassword'
            return dict(state=self.state())
        return super().send(c)


class GuestRoomProbe(RoomProbe):
    """Synthetic non-host text and rectangles from the reported room screen."""
    def __init__(self, directory, **options):
        super().__init__(directory, **options)
        self.end = 'なし'

    def state(self):
        state = super().state()
        if self.phase == 'room':
            state['body'] = ('隠れ乱闘\n教典\n個人戦\nチーム戦\n終末の時\n'+self.end+
                             '\n全体にお告げ\nお告げの記録\n'+'\n'.join(self.members)+
                             '\nSyntheticOpponent\nSyntheticName\nSyntheticName')
            state['texts'] = [dict(text='終末の時', x=355, y=385, w=89, h=28),
                              dict(text=self.end, x=383, y=415, w=33, h=21)]
            state['texts'] += [dict(text=n, x=671, y=141+35*i, w=97, h=23)
                               for i, n in enumerate(self.members)]
            state['texts'] += [dict(text=n, x=1152, y=655+22*i, w=75, h=18)
                               for i, n in enumerate(['SyntheticOpponent', 'SyntheticName'])]
        return state

    def send(self, command):
        assert command.get('text') not in ('終末の時', 'なし', '戦いを始める')
        return super().send(command)


def transition_state():
    return dict(body='←\n隠れ乱闘\n教典\nSyntheticName\n全体にお告げ\nお告げの記録\nSyntheticOpponent',
                texts=[], inputs=[dict(value='')], images=[])


@pytest.fixture
def room_clock(monkeypatch):
    import gfrl.official_bot as bot
    clock = [0.0]
    monkeypatch.setattr(bot.time, 'monotonic', lambda: clock[0])
    def sleep(seconds):
        clock[0] += seconds
    monkeypatch.setattr(bot.time, 'sleep', sleep)
    return clock


@pytest.mark.parametrize('probe_type', [RoomProbe, GuestRoomProbe])
@pytest.mark.parametrize('start', ['human', 'bot'])
def test_room_transitions_on_start_and_return(tmp_path, room_clock, probe_type, start):
    class TransitionProbe(probe_type):
        def __init__(self):
            # A guest's host starts independently of its own room_start option.
            super().__init__(tmp_path, start=start if probe_type is RoomProbe else 'human')
            self.hidden = {}; self.hiding = False

        def state(self):
            state = super().state()
            key = (self.phase, self.stream_epoch)
            self.hiding = ((self.phase == 'battle' and self.index == 0
                            or self.phase == 'room' and self.stream_epoch > 0)
                           and self.hidden.get(key, 0) < 4)
            if self.hiding:
                self.hidden[key] = self.hidden.get(key, 0)+1
                return transition_state()
            return state

        def send(self, command):
            if command['kind'] == 'capture' and self.hiding:
                self.commands.append(command)
                return dict(capture=dict(captures=[]))
            return super().send(command)

    probe = TransitionProbe()
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room',
                 password='SyntheticPassword', room_start=start, games=2,
                 settle_timeout=.01, room_wait=3)
    journal.close()
    assert report['status'] == 'complete' and report['games'] == 2
    assert probe.hidden == {('battle', 0): 4, ('room', 1): 4,
                            ('battle', 1): 4}
    assert probe.joins == 2 and probe.exits == 0
    assert sum(c.get('text') == '戦いを始める' for c in probe.commands) == (
        2 if probe_type is RoomProbe and start == 'bot' else 0)


@pytest.mark.parametrize('received', [False, True])
def test_room_start_transition_waits_for_receive_or_times_out(tmp_path, room_clock, received):
    from gfrl.official_bot import wait_room_start
    probe = RoomProbe(tmp_path, start='bot')
    probe.phase = 'room'; probe.end = 'なし'; probe.members.append('SyntheticName')
    actual_state = probe.state
    reads = []
    def state():
        reads.append(room_clock[0])
        return actual_state() if probe.phase == 'room' else transition_state()
    probe.state = state
    if received:
        assert wait_room_start(probe, 'SyntheticName', [], room_start='bot', room_wait=1) == 1
        assert probe.state()['body'] == transition_state()['body']  # receive precedes G.F.
    else:
        actual_send = probe.send
        def send(command):
            if command['kind'] == 'capture':
                return dict(capture=dict(captures=[]))
            return actual_send(command)
        probe.send = send
        with pytest.raises(UIUnsupported, match='^room_wait_timeout$'):
            wait_room_start(probe, 'SyntheticName', [], room_start='bot', room_wait=1)
        assert room_clock[0] >= 1 and len(reads) >= 4
    assert sum(c.get('text') == '戦いを始める' for c in probe.commands) == 1


@pytest.mark.parametrize('body', ['修行\n隠れ乱闘\n真剣タイマン', '修行\n教典',
                                  '教典\n隠れ乱闘', '隠れ乱闘\n個人戦'])
def test_room_unknown_screens_still_stop(tmp_path, room_clock, body):
    from gfrl.official_bot import wait_room_start, wait_room_return
    class UnknownProbe:
        def send(self, command):
            assert command == dict(kind='state')
            return dict(state=dict(body=body))
    for wait in (lambda: wait_room_start(UnknownProbe(), 'SyntheticName', [],
                                       room_start='bot', room_wait=1),
                 lambda: wait_room_return(UnknownProbe(), 1)):
        with pytest.raises(UIUnsupported, match='^room_start_screen_unknown$'):
            wait()
    assert room_clock[0] == 0


@pytest.mark.parametrize('games', [1, 2])
def test_room_return_transition_timeout(tmp_path, room_clock, games):
    class StuckReturn(RoomProbe):
        def state(self):
            if self.stream_epoch:
                return transition_state()
            return super().state()
    probe = StuckReturn(tmp_path)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room',
                 password='SyntheticPassword', games=games, settle_timeout=.01, room_wait=1)
    journal.close()
    assert report['games'] == 1 and probe.exits == 0
    if games == 1:
        assert report['status'] == 'complete'
    else:
        assert report['reason'] == 'room_wait_timeout' and room_clock[0] >= 1


@pytest.mark.parametrize('start', ['human', 'bot'])
def test_guest_room_lifecycle_waits_for_host(tmp_path, capsys, start):
    probe = GuestRoomProbe(tmp_path)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room',
                 password='SyntheticPassword', room_start=start, games=2, settle_timeout=.05, room_wait=3)
    journal.close()
    assert report['status'] == 'complete' and report['games'] == 2
    assert probe.joins == 2 and probe.exits == 0
    output = capsys.readouterr()
    assert len(output.err.splitlines()) == (2 if start == 'bot' else 0)
    assert start != 'bot' or '部屋主' in output.err
    assert all(s not in output.err+output.out+(directory/'moves.jsonl').read_text()
               for s in ('SyntheticName', 'SyntheticOpponent', 'SyntheticPassword'))


@pytest.mark.parametrize('clears', [False, True])
def test_guest_waits_for_host_to_clear_end_time(tmp_path, monkeypatch, clears):
    import gfrl.official_bot as bot
    probe = GuestRoomProbe(tmp_path)
    probe.end = 'G.F.100'
    clock = [0.0]; sleeps = []
    monkeypatch.setattr(bot.time, 'monotonic', lambda: clock[0])
    def sleep(seconds):
        clock[0] += seconds
        sleeps.append(seconds)
        if clears and clock[0] >= .4:
            probe.end = 'なし'  # The host changes the setting independently.
    monkeypatch.setattr(bot.time, 'sleep', sleep)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room',
                 password='SyntheticPassword', room_start='bot', settle_timeout=.01, room_wait=1)
    journal.close()
    assert sleeps and sum(sleeps) >= .4
    if clears:
        assert report['status'] == 'complete' and probe.joins == 1
    else:
        assert clock[0] >= 1
        assert report['reason'] == 'room_end_time_not_none' and probe.joins == 0
        assert probe.exits == 0


def test_guest_started_game_with_end_time_is_rejected(tmp_path):
    probe = GuestRoomProbe(tmp_path)
    for game in probe.games:
        game['tiebreakGF'] = 100
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room',
                 password='SyntheticPassword', room_start='bot', settle_timeout=.01, room_wait=2)
    journal.close()
    assert report['reason'] == 'room_game_end_time'
    assert probe.joins == 1 and probe.exits == 0


@pytest.mark.parametrize('start', ['human','bot'])
def test_room_lifecycle_and_privacy(tmp_path, capsys, start):
    probe = RoomProbe(tmp_path, start=start)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room',
                 password='SyntheticPassword', room_start=start, games=2, settle_timeout=.05, room_wait=3)
    journal.close()
    assert report['status'] == 'complete' and report['games'] == 2
    assert probe.joins == 2 and probe.exits == 0
    starts = [c for c in probe.commands if c.get('text') == '戦いを始める']
    assert len(starts) == (2 if start == 'bot' else 0)
    output = capsys.readouterr()
    assert all(s not in (directory/'moves.jsonl').read_text()+output.err+output.out
               for s in ('SyntheticName','SyntheticOpponent','SyntheticPassword'))


@pytest.mark.parametrize('options,reason', [
    (dict(stuck_end=True), 'visible_state_timeout'),
    (dict(members=2), 'room_participant_count'),
    (dict(duplicate=True), 'room_own_participant_not_unique'),
    (dict(match_names=0), 'viewer_not_unique'),
    (dict(match_names=2), 'viewer_not_unique'),
    (dict(game_players=3), 'room_game_not_two_players'),
    (dict(end_change=True), 'room_end_time_changed'),
])
def test_room_stops_without_start_or_exit_and_redacts_state(tmp_path, capsys, options, reason):
    probe = RoomProbe(tmp_path, **options)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room', password='SyntheticPassword',
                 room_wait=2, settle_timeout=.01)
    journal.close()
    assert report['reason'] == reason and probe.exits == 0
    assert not any(c.get('text') in ('戦いを始める','戦いをあきらめる') for c in probe.commands)
    assert (directory/'stopped-screenshot.png').is_file()
    recorded = (directory/'moves.jsonl').read_text()+(directory/'stopped-state.json').read_text()
    out = capsys.readouterr()
    for s in ('SyntheticName','SyntheticOpponent','SyntheticPassword'):
        assert s not in recorded+out.err+out.out


def test_room_dry_run_never_submits_move(tmp_path, capsys):
    probe = RoomProbe(tmp_path, start='bot')
    probe.viewer = 2  # Training's probe seat hint must not override name matching.
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room', password='SyntheticPassword',
                 room_start='bot', dry_run=True, settle_timeout=.01, room_wait=2)
    journal.close()
    assert report['status'] == 'dry_run'
    assert [c for c in probe.commands if c['kind']=='clickPoint'] == [point(400,173)]
    out = capsys.readouterr()
    assert all(s not in out.out+out.err+(directory/'moves.jsonl').read_text()
               for s in ('SyntheticName','SyntheticOpponent','SyntheticPassword'))


@pytest.mark.parametrize('extra', [ ['--on-stop','abandon'], [], ['--room-wait','0'], ['--turn-wait','nan'] ])
def test_room_argument_errors_are_private_before_initialization(tmp_path, capsys, extra):
    from gfrl.official_bot import main
    args = ['--mode','room','--checkpoint','unused','--out-dir',str(tmp_path/'out'), '--playwright','unused']
    if extra: args += ['--password','SyntheticPassword']+extra
    with pytest.raises(SystemExit): main(args)
    assert 'SyntheticPassword' not in capsys.readouterr().err
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('permissions', [0o600,0o644])
def test_password_file_private_errors(tmp_path, capsys, permissions):
    from gfrl.official_bot import main
    path = tmp_path/'secret'; path.write_text('SyntheticPassword\n'); path.chmod(permissions)
    args = ['--mode','room','--password-file',str(path),'--checkpoint','missing',
            '--out-dir',str(tmp_path/'out'),'--playwright','unused']
    if permissions == 0o600:
        assert main(args) == 1  # proceeds to the missing local model, never a probe
    else:
        with pytest.raises(SystemExit): main(args)
    out = capsys.readouterr()
    assert 'SyntheticPassword' not in out.err+out.out


@pytest.mark.parametrize('start,members', [('human',1),('bot',0)])
def test_room_timeout_never_starts_without_two_bot_participants(tmp_path, start, members):
    probe = RoomProbe(tmp_path, start='bot', members=members)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room', password='SyntheticPassword',
                 room_start=start, room_wait=.01, settle_timeout=.01)
    journal.close()
    assert report['reason'] == 'room_wait_timeout'
    assert not any(c.get('text') == '戦いを始める' for c in probe.commands)
    assert probe.exits == 0


def test_room_existing_participation_does_not_toggle(tmp_path):
    from gfrl.official_bot import enter_room
    probe = RoomProbe(tmp_path, start='bot')
    probe.phase = 'room'; probe.members.append('SyntheticName'); probe.end = 'なし'
    enter_room(probe, 'SyntheticName', password='SyntheticPassword', timeout=.01)
    assert not any(c['kind']=='clickPoint' for c in probe.commands)


def test_room_bot_start_is_never_retried(tmp_path):
    class RejectedStart(RoomProbe):
        def send(self, c):
            if c.get('text') == '戦いを始める':
                self.commands.append(c)
                return dict(state=self.state())
            return super().send(c)
    probe = RejectedStart(tmp_path, start='bot')
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room', password='SyntheticPassword',
                 room_start='bot', room_wait=.3, settle_timeout=.01)
    journal.close()
    assert report['reason'] == 'room_wait_timeout'
    assert sum(c.get('text') == '戦いを始める' for c in probe.commands) == 1


def test_room_name_checks_continue_after_viewer_selection(tmp_path, capsys):
    probe = RoomProbe(tmp_path)
    probe.games[1]['players'][1]['name'] = 'SyntheticName'
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room', password='SyntheticPassword',
                 room_wait=2, settle_timeout=.01)
    journal.close()
    assert report['reason'] == 'unsupported_receive'
    assert probe.exits == 0
    assert 'SyntheticName' not in (directory/'moves.jsonl').read_text()+capsys.readouterr().err


def test_room_only_documents_parse_as_no_games(tmp_path):
    from gfrl.official_bot import named_room_parser
    body = json.dumps([{'documentChange': {'document': {'name':'synthetic-room',
                       'fields':{'password':firestore('SyntheticPassword')}}}}])
    raw = f"{len(body.encode('utf-16-le'))//2}\n{body}".encode()
    parser, _ = named_room_parser('SyntheticName', [])
    assert parser.parse_prefix(raw)['games'] == []


def test_room_turn_wait_is_separate_from_send_acceptance(tmp_path, monkeypatch):
    import gfrl.official_bot as bot
    probe = RoomProbe(tmp_path)
    # An opponent-owned main scene between two own decisions. Release its next
    # received update only after the loop enters the opponent waiting branch.
    probe.games[1]['attacks'] = []
    probe.games[1]['attackTurnPlayerId'] = 2
    probe.games[2]['events'] = [e for e in probe.games[2]['events'] if e.get('action') != 'sell']
    probe.rs = results(probe.games)
    assert probe.rs[1]['status'] == 'waiting'
    actual = bot.accepted; waits = []
    def accepted(p, streams, update, timeout=20, **kwargs):
        waits.append((update,timeout))
        if update == 2:
            probe.index = 2
        return actual(p, streams, update, timeout, **kwargs)
    monkeypatch.setattr(bot, 'accepted', accepted)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room', password='SyntheticPassword',
                 timeout=.07, turn_wait=.8, room_wait=2, settle_timeout=.01)
    journal.close()
    assert report['status'] == 'complete'
    assert (2,.8) in waits
    assert all(timeout==.07 for update,timeout in waits if update != 2)


def test_room_password_dialog_stop_does_not_save_password(tmp_path, capsys):
    class PasswordStop(RoomProbe):
        value = ''
        def state(self):
            s = super().state()
            if self.phase == 'password':
                s['inputs'] = [dict(value=self.value)]
            return s
        def send(self, c):
            if self.phase == 'password' and c['kind'] == 'fillName':
                self.commands.append(c); self.value = c['text']
                return dict(state=self.state())
            if c.get('text') == '唱える':
                self.commands.append(c)
                return dict(state=self.state())  # no room arrives
            if c['kind'] == 'screenshot':
                assert self.value == '[redacted]'
            return super().send(c)
    probe = PasswordStop(tmp_path)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room', password='SyntheticPassword',
                 settle_timeout=.01)
    journal.close()
    assert report['status'] == 'stopped'
    assert (directory/'stopped-screenshot.png').is_file()
    output = capsys.readouterr()
    for path in directory.iterdir():
        assert b'SyntheticPassword' not in path.read_bytes()
    assert 'SyntheticPassword' not in output.out+output.err


def purchase_scene(kind='accept'):
    p = deepcopy(results()[0]['public'])
    p.update(phase=3, pending=None, purchase=dict(origin=1, buyer=1, seller=2, model=6))
    s = visible(p)
    # Measured text bounds inside the two large 205..495 buttons.
    s['texts'] += [dict(text='買う', x=303.828125, y=113, w=92.34375, h=59),
                   dict(text='買わない', x=257.65625, y=313, w=184.6875, h=59),
                   dict(text='買う', x=1040, y=574, w=40, h=23)]
    s['images'].append(dict(path=PATHS[6], x=545, y=203, w=80, h=80))
    return p, s, action(p, kind)


@pytest.mark.parametrize('kind,y', [('accept',142.5), ('decline',342.5)])
def test_purchase_uses_unique_position_and_offer_without_extra_confirm(kind, y):
    p,s,a = purchase_scene(kind)
    class Probe:
        commands = []
        def send(self, c):
            self.commands.append(c)
            return dict(state=s)
    probe = Probe()
    execute(probe,s,p,a,'SyntheticName',settle_timeout=0)
    assert probe.commands == [point(350,y)]


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'moved', 'offer', 'phase', 'ops'])
def test_purchase_fails_closed_before_click(bad):
    p,s,a = purchase_scene()
    if bad == 'duplicate': s['texts'].append(deepcopy(s['texts'][1]))
    if bad == 'missing': s['texts'].pop(2)
    if bad == 'moved': s['texts'][1]['x'] += 340
    if bad == 'offer': s['images'][-1]['path'] = PATHS[7]
    if bad == 'phase': p['phase'] = 0
    if bad == 'ops': a['operations'].pop()
    with pytest.raises(UIUnsupported):
        prepare(s,p,a,'SyntheticName')


@pytest.mark.parametrize('event', ['hit','miss'])
@pytest.mark.parametrize('delay', [0, 3, 5])
def test_chance_update_waits_then_clicks(tmp_path, settle_clock, event, delay):
    probe = FakeProbe(tmp_path)
    # Keep the first decision valid but add the public animation marker.
    probe.games[0]['events'].append(dict(action=event,playerId=1))
    probe.rs = results(probe.games)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    first_click = []
    send = probe.send
    def timed(c):
        if c['kind'] == 'clickPoint' and not first_click:
            first_click.append(settle_clock.now)
        return send(c)
    probe.send = timed
    report = run(probe,advisor,journal,name='SyntheticName',settle_timeout=1, chance_delay=delay)
    journal.close()
    assert report['status'] == 'complete'
    assert first_click == [delay]


def test_refill_missing_images_waits_without_clicking(tmp_path, settle_clock):
    probe = LagProbe(tmp_path, forever=True, mutation=lambda s: s['images'].pop())
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe,advisor,journal,name='SyntheticName',settle_timeout=1)
    journal.close()
    assert report['status'] == 'stopped'
    assert not any(c['kind'] == 'clickPoint' for c in probe.commands)


@pytest.mark.parametrize('reject_both', [False, True])
@pytest.mark.parametrize('ignored_update', [1, 2])
def test_bot_retries_confirmation_once_and_records_it(tmp_path, settle_clock, reject_both, ignored_update):
    class IgnoreConfirmation(FakeProbe):
        confirms = 0
        def send(self, c):
            confirmation = (c['kind'] == 'clickPoint' and c['y'] in (193,250)
                            and c['x'] in (350,690))
            if confirmation:
                self.confirms += 1
                ignored_before = ignored_update-1
                self.reject = self.index == ignored_before and (reject_both or self.confirms == ignored_update)
            return super().send(c)
    probe = IgnoreConfirmation(tmp_path)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe,advisor,journal,name='SyntheticName',timeout=7,settle_timeout=1)
    journal.close()
    records = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    retried = [r for r in records if r.get('confirmation_retries')]
    assert len(retried) == 1 and retried[0]['confirmation_retries'] == 1
    assert retried[0]['confirm_method'] == 'pane_retry'
    if ignored_update == 1:
        # Exchange card selection and allocation are each dispatched once.
        assert retried[0]['clicks'].count(point(230,533)) == 1
        assert retried[0]['clicks'].count(point(350,250)) == 2
    else:
        # The reported empty response has only two confirmation attempts.
        assert retried[0]['clicks'] == [point(690,193)]*2
    if reject_both:
        assert report['reason'] == 'acceptance_timeout' and probe.confirms == ignored_update+1
        assert retried[0]['accepted'] is False
    else:
        assert report['status'] == 'complete' and probe.confirms == 6
        assert retried[0]['accepted'] is True


def test_retry_requires_identical_confirmation_state():
    from gfrl.official_ui import same_confirmation
    p,s,a = purchase_scene()
    plan = prepare(s,p,a,'SyntheticName'); plan['confirmation_state'] = deepcopy(s)
    assert same_confirmation(s,p,a,'SyntheticName',plan)
    for change in ('gf','pane','selection','button'):
        altered = deepcopy(s)
        if change == 'gf': altered['body'] = altered['body'].replace('G.F.1','G.F.2')
        if change == 'pane': altered['texts'][0]['x'] += 340
        if change == 'selection': altered['images'].append(dict(path=PATHS[5],x=205,y=103,w=80,h=80))
        if change == 'button': altered['texts'][1]['y'] += 200
        assert not same_confirmation(altered,p,a,'SyntheticName',plan)


@pytest.mark.parametrize('kind', ['accept','decline'])
def test_purchase_click_is_accepted_by_received_update(tmp_path, kind):
    games = deepcopy([GAMES[0],GAMES[-1]])
    games[0]['attacks'] = [dict(playerId=1,targetPlayerId=2,itemModelIds=[5],buyingItemModelId=6)]
    class PurchaseProbe(FakeProbe):
        def state(self):
            if self.phase == 'battle' and self.index == 0:
                p,s,_ = purchase_scene(kind)
                # Use the received public resource/hand layout, plus the
                # independently measured buttons and offer geometry.
                s = visible(self.rs[0]['public']) | dict(texts=s['texts'])
                s['images'].append(dict(path=PATHS[6],x=545,y=203,w=80,h=80))
                return s
            return super().state()
        def send(self,c):
            if c == point(350,142.5 if kind == 'accept' else 342.5):
                self.commands.append(c); self.index = 1
                return dict(state=self.state())
            return super().send(c)
    probe = PurchaseProbe(tmp_path,games=games)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe,lambda r:dict(status='advice',action=action(r['public'],kind)),
                 journal,name='SyntheticName',timeout=.2,settle_timeout=.01)
    journal.close()
    assert report['status'] == 'complete'
    row = json.loads((directory/'moves.jsonl').read_text().splitlines()[0])
    assert row['accepted'] and row['received_update'] > row['state']['update']
    assert row['clicks'] == [point(350,142.5 if kind == 'accept' else 342.5)]


def test_chance_wait_refreshes_and_redecides_on_new_receive(tmp_path, settle_clock):
    probe = LagProbe(tmp_path, advance=True)
    probe.games[0]['events'].append(dict(action='hit',playerId=1))
    probe.rs = results(probe.games)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe,advisor,journal,name='SyntheticName',settle_timeout=2,dry_run=True)
    journal.close()
    rows = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    assert report['status'] == 'dry_run'
    assert rows[0]['status'] == 'superseded' and rows[0]['clicks'] == []
    assert rows[1]['state']['update'] == 2
    assert not any(c['kind'] == 'clickPoint' for c in probe.commands)


UNKNOWN_EVENTS = [dict(action='removeItems', items=None),
                  dict(action='dealDarkDamage', damage=True),
                  dict(action='hit', index=0)]


@pytest.mark.parametrize('event', UNKNOWN_EVENTS)
def test_lenient_event_advice_and_hidden_payload_invariance(tmp_path, event):
    import numpy as np
    from gfrl.model import OBS_FIELDS
    from gfrl.official_advisor import advise
    class FirstLegal:
        def __call__(self, env, rows):
            if int(env.stage[0]) == 2 and env.mask[0,2]:
                return np.array([2], np.int32)
            return env.mask.argmax(1).astype(np.int32)
    outputs = []
    for secret in (6, 296):
        game = deepcopy(GAMES[0])
        unknown = deepcopy(event)
        # Undisclosed foreign inventory can be arbitrarily embedded in an
        # unrecognized event. No part of it may affect projection or advice.
        unknown['items'] = [dict(id=999, modelId=secret, used=False)]
        game['events'].append(unknown)
        parser = parser_module()
        with pytest.raises(parser.ParseError):
            parser.parse_prefix(frame(game), 1)
        record = parser.parse_prefix(frame(game), 1, lenient_events=True)['games'][0]
        assert record['game']['events'][-1] == dict(action='unknownEvent')
        streams = OfficialStreams(OfficialSession(1), lenient_events=True)
        result = streams.poll([capture_file(tmp_path, [game], serial=secret)])
        assert result['status'] == 'decision' and not result['history_complete']
        assert streams.session.events == []
        arrays = {k: getattr(result['env'], k).copy() for k in OBS_FIELDS + ('mask',)}
        advice = advise(result, FirstLegal())
        assert advice['status'] == 'advice'
        assert dict(replaced_events=result['replaced_events']) in advice['warnings']
        outputs.append((result['public'], result['replaced_events'], arrays, advice))
    assert outputs[0][:2] == outputs[1][:2]
    for k in outputs[0][2]:
        np.testing.assert_array_equal(outputs[0][2][k], outputs[1][2][k])
    assert outputs[0][3] == outputs[1][3]


@pytest.mark.parametrize('strict', [False, True])
def test_bot_unknown_events_logged_or_strict_stop(tmp_path, capsys, strict):
    games = deepcopy(GAMES)
    games[0]['events'].extend(deepcopy(UNKNOWN_EVENTS))
    probe = FakeProbe(tmp_path, games=games)
    probe.rs = results()  # UI follows the authoritative panes, not unknown event payloads.
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', strict_events=strict)
    journal.close()
    rows = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    stderr = capsys.readouterr().err
    if strict:
        assert report['reason'] == 'unsupported_receive'
        assert not any(c['kind'] == 'clickPoint' for c in probe.commands)
        assert 'warning:replaced_events' not in stderr
    else:
        assert report['status'] == 'complete'
        assert len(rows[0]['replaced_events']) == 3
        assert [e['action'] for e in rows[0]['replaced_events']] == [e['action'] for e in UNKNOWN_EVENTS]
        assert len(stderr.splitlines()) == 1


@pytest.mark.parametrize('count', [5, 6])
@pytest.mark.parametrize('pane', [350, 690])
def test_large_defense_counts_images_without_row_assumptions(count, pane):
    from gfrl.official_ui import selected, same_confirmation
    p = deepcopy(results()[3]['public'])
    p['hand'] = [dict(id=i+100, face=130, used=False, disguised=False) for i in range(count)]
    s = visible(p)
    s['texts'][0]['x'] = pane-30
    s['images'] = [dict(path=PATHS[130], x=190+82*i, y=493, w=80, h=80) for i in range(count)]
    a = action(p, 'defend', [130]*count)
    commands = []
    class DefenseProbe:
        def send(self, c):
            commands.append(c)
            if c['kind'] == 'clickPoint' and c['y'] == 533:
                i = sum(selected(s, pane).values())
                # Shift horizontally, overlap rows, and put the fifth card
                # below the previous hardcoded y=403 ceiling.
                y = [103, 185, 267, 349, 431][i] if count == 5 else 103+60*i
                cx = pane-40 if i % 2 == 0 else pane+30
                s['images'].append(dict(path=PATHS[130], x=cx, y=y, w=80, h=80))
            return dict(state=deepcopy(s))
    plan = execute(DefenseProbe(), s, p, a, 'SyntheticName', settle_timeout=0)
    assert selected(s, pane) == {PATHS[130]: count}
    assert len(commands) == count+1
    confirm = commands[-1]
    assert pane-145 < confirm['x'] < pane+145 and 103 <= confirm['y'] < 480
    assert all(not (r['x'] <= confirm['x'] <= r['x']+r['w'] and
                    r['y'] <= confirm['y'] <= r['y']+r['h'])
               for r in s['images'])
    assert same_confirmation(s, p, a, 'SyntheticName', plan)


def test_confirmation_stops_when_pane_has_no_clear_candidate():
    from gfrl.official_ui import confirmation_point
    s = dict(images=[dict(path=PATHS[130], x=x, y=y, w=80, h=80)
                     for x in (205, 275, 345, 415) for y in (103, 180, 260, 340, 420)])
    with pytest.raises(UIUnsupported, match='^confirmation_point_unavailable$'):
        confirmation_point(s, 350, point(350, 193))


def test_selected_cards_use_goodfield_hand_limit():
    from gfrl import MatchConfig
    p = deepcopy(results()[3]['public'])
    count = MatchConfig().hand_limit
    p['hand'] = [dict(id=i+100, face=130, used=False, disguised=False) for i in range(count)]
    s = visible(p)
    s['images'] = [dict(path=PATHS[130], x=190+82*(i % 9), y=493+102*(i//9), w=80, h=80)
                   for i in range(count)]
    assert len(prepare(s, p, action(p, 'defend', [130]*count), 'SyntheticName')['cards']) == count
    p['hand'].append(dict(id=999, face=130, used=False, disguised=False))
    with pytest.raises(UIUnsupported, match='^selected_cards_over_hand_limit$'):
        prepare(s, p, action(p, 'defend', [130]*(count+1)), 'SyntheticName')


class PreviousRoomGameProbe(RoomProbe):
    def __init__(self, directory, *, own=False, finished=True, returns=True, receive=True):
        super().__init__(directory, start='bot')
        self.previous = deepcopy(self.games[-1 if finished else 0])
        if not own:
            self.previous['players'][0]['name'] = 'SyntheticStranger'
        self.finished = finished; self.returns = returns; self.receive = receive
        self.previous_reads = 0; self.previous_exits = 0
        self.entered_previous = False

    def state(self):
        if self.phase == 'previous':
            self.previous_reads += 1
            if not self.finished and self.returns and self.previous_reads >= 4:
                self.phase = 'room'
                return super().state()
            return dict(body='G.F.46\n勝利\n戦いを終わる' if self.finished else 'G.F.14',
                        images=[], inputs=[], texts=[])
        return super().state()

    def send(self, c):
        if c['kind'] == 'capture' and self.entered_previous:
            if self.phase == 'previous':
                self.commands.append(c)
                reply = dict(capture=dict(captures=[]))
            else:
                reply = super().send(c)
            self.serial += 1
            if self.receive:
                # Retain the spectator stream in every subsequent prefix set:
                # entering the new game must exclude it using the new baseline.
                reply['capture']['captures'].append(
                    capture_file(self.directory, [self.previous], seq=77, serial=self.serial))
            return reply
        if c.get('text') == '唱える':
            self.commands.append(c); self.phase = 'previous'; self.entered_previous = True
            return dict(state=self.state())
        if self.phase == 'previous' and c.get('text') == '戦いを終わる':
            self.commands.append(c); self.previous_exits += 1
            if self.returns: self.phase = 'room'
            return dict(state=self.state())
        return super().send(c)


@pytest.mark.parametrize('finished', [False, True])
def test_room_entry_leaves_foreign_game_and_can_start_own(tmp_path, room_clock, finished):
    probe = PreviousRoomGameProbe(tmp_path, finished=finished)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room', password='SyntheticPassword',
                 room_start='bot', dry_run=True, room_wait=2, settle_timeout=.01)
    journal.close()
    assert report['status'] == 'dry_run'
    assert probe.previous_exits == int(finished) and probe.joins == 1


@pytest.mark.parametrize('finished', [False, True])
def test_room_entry_stops_for_previous_own_game(tmp_path, room_clock, capsys, finished):
    probe = PreviousRoomGameProbe(tmp_path, own=True, finished=finished, returns=False)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode='room', password='SyntheticPassword',
                 room_wait=2, settle_timeout=.01)
    journal.close()
    assert report['reason'] == 'room_previous_game_in_progress'
    assert probe.previous_exits == probe.joins == 0
    saved = (directory/'moves.jsonl').read_text()+capsys.readouterr().err
    assert 'SyntheticName' not in saved and 'SyntheticOpponent' not in saved


@pytest.mark.parametrize('finished,receive', [(False, True), (True, True), (True, False)])
def test_room_entry_wait_is_bounded_and_exit_requires_received_roster(tmp_path, room_clock, finished, receive):
    from gfrl.official_bot import enter_room
    probe = PreviousRoomGameProbe(tmp_path, finished=finished, returns=False, receive=receive)
    with pytest.raises(UIUnsupported, match='^room_wait_timeout$'):
        enter_room(probe, 'SyntheticName', password='SyntheticPassword', timeout=.01, room_wait=1)
    assert room_clock[0] >= 1
    assert probe.previous_exits == int(finished and receive)
    assert probe.joins == 0


@pytest.mark.parametrize('mode', ['training', 'room'])
def test_birth_not_accepted_reports_private_reason(tmp_path, room_clock, capsys, mode):
    class RejectedLogin(RoomProbe):
        def send(self, c):
            if c.get('text') == '生誕する':
                self.commands.append(c)
                return dict(state=self.state())
            return super().send(c)
    probe = RejectedLogin(tmp_path)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advisor, journal, name='SyntheticName', mode=mode, password='SyntheticPassword',
                 timeout=1, settle_timeout=10)
    journal.close()
    assert report['reason'] == 'login_not_accepted'
    assert 1 <= room_clock[0] < 2
    stderr = capsys.readouterr().err
    assert stderr.splitlines().count('名前が使用中の可能性。別の名前で') == 1
    assert 'SyntheticName' not in stderr
    assert sum(c.get('text') == '生誕する' for c in probe.commands) == 1


@pytest.mark.parametrize('accept_allow', [True, False])
@pytest.mark.parametrize('retry_allowed', [True, False])
def test_empty_response_allow_button_once(tmp_path, settle_clock, monkeypatch, accept_allow, retry_allowed):
    if not retry_allowed:
        monkeypatch.setattr('gfrl.official_ui.same_confirmation', lambda *args: False)
    class AllowProbe(FakeProbe):
        def state(self):
            s = super().state()
            if self.phase == 'battle' and self.index == 1:
                s['texts'].append(dict(text='許す', x=664, y=406, w=52, h=33))
                if self.reject:
                    s['body'] += '\nincidental tooltip'
            return s
        def send(self, c):
            if self.phase == 'battle' and self.index == 1:
                if c == point(690,423):
                    self.commands.append(c)
                    if accept_allow:
                        self.index += 1; self.reject = False
                    return dict(state=self.state())
                if c == point(690,193):
                    self.reject = True
            return super().send(c)
    probe = AllowProbe(tmp_path)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe,advisor,journal,name='SyntheticName',timeout=7,settle_timeout=0)
    journal.close()
    rows = list(map(json.loads,(directory/'moves.jsonl').read_text().splitlines()))
    row = next(r for r in rows if r.get('state',{}).get('update') == 2)
    assert row['clicks'] == [point(690,193)]*(2 if retry_allowed else 1) + [point(690,423)]
    assert row.get('confirmation_retries',0) == int(retry_allowed)
    assert row['confirm_method'] == 'allow_button'
    assert row['accepted'] == accept_allow
    assert report['status'] == ('complete' if accept_allow else 'stopped')
    assert rows[0]['confirm_method'] == 'pane'


@pytest.mark.parametrize('bad', ['label', 'position', 'selected', 'main', 'changed_hand'])
def test_allow_button_rejects_unverified_or_nonempty_response(bad):
    from gfrl.official_ui import allow_button
    p = results()[1]['public']; s = visible(p); a = action(p,'defend')
    plan = prepare(s,p,a,'SyntheticName'); plan['confirmation_state'] = deepcopy(s)
    s['texts'].append(dict(text='許す', x=664, y=406, w=52, h=33))
    if bad == 'label': s['texts'][-1]['text'] = '攻撃'
    if bad == 'position': s['texts'][-1]['y'] = 300
    if bad == 'selected': a['cards'] = [dict(model=130)]
    if bad == 'main': p['phase'] = 0
    if bad == 'changed_hand': s['images'].pop(0)
    with pytest.raises(UIUnsupported): allow_button(s,p,a,'SyntheticName',plan)


def test_hand_region_excludes_guardian_status_and_fake_overlay():
    from gfrl.official_ui import hand_images
    p = results()[1]['public']; s = visible(p)
    wanted = hand_images(s)
    # Geometry from the stopped scene; all content here is synthetic/public.
    s['images'] += [dict(path='/images/guardians/small/mercury.webp',x=1200,y=264,w=50,h=40),
                    dict(path='/images/curses/small/dream.webp',x=1141,y=288,w=30,h=16),
                    dict(path='/images/items/fake.webp',x=436,y=493,w=80,h=80),
                    dict(path=PATHS[209],x=955,y=463,w=80,h=80),
                    dict(path='/images/guardians/test.webp',x=436,y=493,w=80,h=80)]
    assert hand_images(s) == wanted
    ready(s,p,'SyntheticName')


def test_unsupported_advice_records_reasons_and_action(tmp_path):
    def unsupported(r):
        return dict(status='unsupported', unsupported=['earth_moon_general_execution'],
                    action=action(r['public'],'exchange',[3]),fallback_samples=16,selection_exhausted=True)
    probe = FakeProbe(tmp_path)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe,unsupported,journal,name='SyntheticName',settle_timeout=0)
    journal.close()
    row = json.loads((directory/'moves.jsonl').read_text().splitlines()[-1])
    assert report['reason'] == 'unsupported_advice'
    assert row['detail'] == dict(unsupported=['earth_moon_general_execution'],
                                action=dict(kind='exchange',models=[3]))
    assert row['fallback_samples'] == 16
    assert row['selection_exhausted'] is True


def test_selected_response_never_uses_allow_button(tmp_path, settle_clock):
    class RejectSelected(FakeProbe):
        def state(self):
            s = super().state()
            if self.phase == 'battle':
                s['texts'].append(dict(text='許す',x=664,y=406,w=52,h=33))
            return s
        def send(self, c):
            if self.phase == 'battle' and self.index == 3:
                self.reject = True
            return super().send(c)
    probe = RejectSelected(tmp_path)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe,advisor,journal,name='SyntheticName',timeout=7,settle_timeout=0)
    journal.close()
    assert report['reason'] == 'acceptance_timeout'
    assert point(690,423) not in probe.commands
    row = json.loads((directory/'moves.jsonl').read_text().splitlines()[-1])
    assert row['state']['update'] == 4 and row['confirmation_retries'] == 1


def test_search_duration_does_not_consume_click_or_acceptance_timeout(tmp_path, settle_clock):
    probe = FakeProbe(tmp_path)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    def slow_advisor(result):
        settle_clock.now += 6  # Longer than both waits below, without wall-clock sleep.
        advice = advisor(result)
        advice.update(search_status='main_public_worlds', search_worlds=8,
            search_candidates=2, search_seconds=6., pi_search_top=[dict(path=[0], probability=1.)],
            pi_theta_top=[dict(path=[0], probability=.8)])
        return advice
    report = run(probe, slow_advisor, journal, name='SyntheticName', timeout=1, settle_timeout=1)
    journal.close()
    assert report['status'] == 'complete'
    rows = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    moves = [r for r in rows if 'advice' in r]
    assert moves and all(r['accepted'] for r in moves)
    for row in rows:
        assert all(k in row for k in ('search_status', 'search_worlds', 'search_candidates',
            'search_seconds', 'pi_search_top', 'pi_theta_top'))
    assert all(r['search_worlds'] == 8 and r['search_seconds'] == 6 for r in moves)


def test_receive_advancing_during_search_supersedes_without_click(tmp_path, settle_clock):
    probe = LagProbe(tmp_path, advance=True)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    def slow_advisor(result):
        settle_clock.now += 6
        return advisor(result)
    report = run(probe, slow_advisor, journal, name='SyntheticName', timeout=1,
                 settle_timeout=1, dry_run=True)
    journal.close()
    rows = [json.loads(line) for line in (directory/'moves.jsonl').read_text().splitlines()]
    assert report['status'] == 'dry_run'
    assert rows[0]['status'] == 'superseded' and rows[0]['clicks'] == []
    assert rows[1]['state']['update'] == 2
    assert not any(c['kind'] == 'clickPoint' for c in probe.commands)
