"""Offline regressions for the auto31/33 and roomA9/B9 visible layouts."""
from copy import deepcopy
import json
import stat

import pytest

from gfrl.official_bot import Journal, private_directory, run
from gfrl.official_ui import (
    PATHS, UIUnsupported, execute, point, ready, same_confirmation, verify_resources,
)
from test_official_bot import FakeProbe, action, advisor, results, settle_clock, visible
from test_official_live_fixes import fog_scene


@pytest.mark.parametrize('own_y', [131, 264])
def test_fog_status_band_is_opposite_own_name(own_y):
    p, s = fog_scene()
    s['texts'][-1]['y'] = own_y+5
    other_y = 395-own_y
    s['fogBands'][0]['y'] = other_y
    ready(s, p, 'SyntheticName')
    for failure in ('same_row', 'text', 'missing_own', 'foreign_own'):
        bad = deepcopy(s)
        if failure == 'same_row': bad['fogBands'][0]['y'] = own_y
        if failure == 'text': bad['texts'].append(dict(text='7', x=1100, y=other_y+5, w=10, h=20))
        if failure == 'missing_own': bad['texts'].pop()
        if failure == 'foreign_own': bad['texts'][-1]['text'] = 'Other'
        with pytest.raises(UIUnsupported): verify_resources(bad, p, 'SyntheticName')


@pytest.mark.parametrize('target', [1, 2])
@pytest.mark.parametrize('model', [198, 201])
@pytest.mark.parametrize('own_y', [131, 264])
@pytest.mark.parametrize('effect', ['success', 'unchanged', 'no_arrow', 'wrong_arrow', 'wrong_name'])
def test_target_switch_clicks_once_and_requires_result(target, model, own_y, effect, settle_clock):
    p = deepcopy(results()[2]['public'])
    p['hand'] = [dict(id=99, face=model, used=False, disguised=False)]
    s = visible(p)
    s['texts'] += [dict(text='SyntheticName', x=900, y=own_y+5, w=100, h=29),
                   dict(text='SyntheticOpponent', x=900, y=400-own_y, w=100, h=29)]
    a = action(p, 'play', [model], target=target)
    target_y = own_y if target == 1 else 395-own_y
    switch = point(950, target_y+19.5)

    class Probe:
        def __init__(self): self.commands = []; self.switched = False
        def send(self, command):
            self.commands.append(command)
            if command == point(886, 635):
                s['images'].append(dict(path=PATHS[model], x=205, y=103, w=80, h=80))
                if target == 1:
                    s['texts'].append(dict(text='SyntheticOpponent', x=660, y=60, w=60, h=20))
            elif command == switch:
                self.switched = True
                if effect != 'unchanged':
                    s['texts'] = [r for r in s['texts'] if not (55 <= r['y'] <= 95 and r['x'] >= 520)]
                    s['texts'].append(dict(text='SyntheticName' if target == 1 else 'SyntheticOpponent',
                                           x=660, y=60, w=60, h=20))
                    if effect == 'wrong_name': s['texts'][-1]['text'] = 'SyntheticName' if target == 2 else 'SyntheticOpponent'
                    s['targetArrow'] = {'no_arrow': None, 'wrong_arrow': 'left'}.get(effect, 'right')
            return dict(state=deepcopy(s))

    probe = Probe(); commands = []
    if effect == 'success':
        plan = execute(probe, deepcopy(s), p, a, 'SyntheticName', commands=commands, settle_timeout=1)
        assert commands == [point(886,635), switch, point(350,193)]
        assert plan['commands'] == commands
        assert same_confirmation(s, p, a, 'SyntheticName', plan)
        s['targetArrow'] = None
        assert not same_confirmation(s, p, a, 'SyntheticName', plan)
    else:
        with pytest.raises(UIUnsupported, match='^target_switch_failed$'):
            execute(probe, deepcopy(s), p, a, 'SyntheticName', commands=commands, settle_timeout=1)
        assert commands == [point(886,635), switch]
        assert settle_clock.now >= 1
    assert probe.commands.count(switch) == 1


@pytest.mark.parametrize('appears', [False, True])
def test_fog_response_wait_and_three_observations(tmp_path, settle_clock, appears):
    class Probe(FakeProbe):
        def __init__(self):
            super().__init__(tmp_path)
            self.screens = []; self.response_start = None
        def state(self):
            if self.phase == 'battle' and self.index == 1:
                if self.response_start is None: self.response_start = settle_clock.now
                p, s = fog_scene(True)
                if not appears or settle_clock.now-self.response_start < .5:
                    s['texts'] = [r for r in s['texts'] if r['y'] > 95]
                    s['texts'].append(dict(text='SyntheticOpponent', x=320,y=56,w=60,h=23))
                    s['images'] = [r for r in s['images'] if r['y'] >= 480]
                    s['images'].append(dict(path=PATHS[5],x=205,y=103,w=80,h=80))
                s['observed_at'] = settle_clock.now
                return s
            return super().state()
        def send(self, command):
            if command['kind'] == 'screenshot':
                self.commands.append(command); self.screens.append(settle_clock.now)
                path = tmp_path/'screenshot.png'; path.write_bytes(str(settle_clock.now).encode())
                return dict(screenshot=str(path))
            return super().send(command)
    probe = Probe()
    def advising(r):
        if r['public']['update'] == 2:
            fog, _ = fog_scene(True)
            r['public']['players'] = fog['players']
        return advisor(r)
    directory = private_directory(tmp_path/'log'); journal = Journal(directory)
    report = run(probe, advising, journal, name='SyntheticName', settle_timeout=1, dry_run=False)
    journal.close()
    if appears:
        # The now-observed response is allowed to proceed to confirmation.
        assert point(690,193) in probe.commands
        assert report['status'] == 'complete'
        assert not (directory/'stopped-state-2.json').exists()
    else:
        assert report['reason'] == 'fog_response_ui_unobserved'
        assert probe.screens[0]-probe.response_start >= 1
        assert probe.screens == [probe.screens[0]+i*5 for i in range(3)]
        first_stop = probe.commands.index(dict(kind='screenshot'))
        assert all(c['kind'] in ('state','screenshot') for c in probe.commands[first_stop:])
        row = json.loads((directory/'moves.jsonl').read_text().splitlines()[-1])
        assert row['detail'] == dict(phase='response', own_name_bands=0)
        assert row['clicks'] == []
        for i, suffix in enumerate(('', '-2', '-3')):
            state_file = directory/f'stopped-state{suffix}.json'
            screen_file = directory/f'stopped-screenshot{suffix}.png'
            assert json.loads(state_file.read_text())['observed_at'] == probe.screens[i]
            assert screen_file.read_bytes() == str(probe.screens[i]).encode()
            assert stat.S_IMODE(state_file.stat().st_mode) == 0o600
            assert stat.S_IMODE(screen_file.stat().st_mode) == 0o600


@pytest.mark.parametrize('own_y', [131, 264])
@pytest.mark.parametrize('preview', [False, True])
@pytest.mark.parametrize('arrow', ['right', 'left', None, 'rectangle'])
def test_probe_overlays_offline(own_y, preview, arrow):
    import base64
    from pathlib import Path
    import struct
    import subprocess
    import zlib

    def png(w, pixel):
        def chunk(kind, data):
            return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data))
        raw = b''.join(b'\0'+bytes(c for x in range(w) for c in pixel(x,y)) for y in range(40))
        data = (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB',w,40,8,2,0,0,0))
                +chunk(b'IDAT', zlib.compress(raw))+chunk(b'IEND', b''))
        return base64.b64encode(data).decode()
    def arrow_pixel(x, y):
        if arrow == 'left': x = 39-x
        ink = arrow == 'rectangle' or (arrow is not None and
              ((x < 20 and 12 <= y < 28) or (x >= 20 and abs(y-20) < 40-x)))
        return (255,102,153) if ink else (180,200,240)
    module = Path(__file__).resolve().parents[2]/'scripts/official_probe_pixels.mjs'
    payload = dict(own_y=own_y, purple=png(340, lambda x,y:(133,138,249)),
                   white=png(340, lambda x,y:(255,255,255)), arrow=png(40,arrow_pixel),
                   body='HP 40 MP 10 ¥ 20'+ (' HP 50 MP 20 ¥ 0 +10' if preview else ''))
    code = f'''
      import {{visibleOverlays}} from {json.dumps(module.as_uri())};
      import fs from 'node:fs';
      const p=JSON.parse(fs.readFileSync(0,'utf8')), clips=[];
      const page={{screenshot:async ({{clip,scale}})=>{{
        clips.push(clip); if(scale!=='css') throw new Error('scale');
        return Buffer.from(clip.width===40?p.arrow:clip.y===p.own_y?p.white:p.purple,'base64');
      }}}};
      const state=await visibleOverlays(page,p.body,[{{x:320,y:56}},{{x:660,y:56}}]);
      console.log(JSON.stringify({{state,clips}}));
    '''
    completed = subprocess.run(['node','--input-type=module','-e',code], input=json.dumps(payload),
                               text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    assert result['state'] == dict(fogBands=[dict(x=860,y=395-own_y,w=340,h=40)],
                                   targetArrow=arrow if arrow in ('left','right') else None)
    assert result['clips'] == [dict(x=860,y=y,width=340,height=40) for y in (131,264)] + [
        dict(x=500,y=48,width=40,height=40)]
