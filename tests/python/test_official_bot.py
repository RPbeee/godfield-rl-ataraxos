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
    if mismatch:
        assert report['reason'] == 'visible_target_mismatch'
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
    assert report['reason'] == 'unobserved_ui_kind'
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


@pytest.mark.parametrize('effect', ['confused', 'fog', 'hidden_status'])
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
