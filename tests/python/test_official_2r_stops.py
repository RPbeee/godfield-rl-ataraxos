"""Synthetic/offline regressions; no live receipts or screenshots are fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import subprocess
import zlib

import numpy as np
import pytest

from gfrl import _core
from gfrl.official import OfficialDecision, OfficialSession
from gfrl.official_advisor import advise_supported
from test_official import received_scene, align_hand, token_arrays
from test_official_bot import results
from test_official_live_fixes import PreferFirst

ROOT = Path(__file__).resolve().parents[2]


def test_translucent_fog_pixels_and_negative_overlays(tmp_path):
    def png(pixel):
        def chunk(k, d):
            return struct.pack('>I',len(d))+k+d+struct.pack('>I',zlib.crc32(k+d))
        rows = b''.join(b'\0'+b''.join(bytes(pixel(x,y)) for x in range(340)) for y in range(40))
        return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',340,40,8,2,0,0,0))+
                chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))
    # Measured roomB8/roomB11 envelope, with spatial and temporal variation.
    def fog(phase):
        return lambda x,y: (120+(x+y+phase)%19,130+((x+y+phase)%19)*12//18,246+(x+3*y+phase)%7)
    cases = [(fog(i), True) for i in (0,7)]
    cases += [(lambda x,y: (163,188,241), False),
              (lambda x,y: (20,20,20) if 30<x<45 and 10<y<30 else (255,255,255),False),
              (lambda x,y: (240,100,160) if 140<x<200 else (255,255,255),False),
              (lambda x,y: (120,146,250),False),
              (lambda x,y: (255,255,255) if (x,y)==(120,20) else fog(0)(x,y),False)]
    files=[]
    for i,(pixel,_) in enumerate(cases):
        f=tmp_path/f'{i}.png';f.write_bytes(png(pixel));files.append(str(f))
    code = f"import {{purpleFog}} from {json.dumps((ROOT/'scripts/official_probe_pixels.mjs').as_uri())};import fs from 'node:fs';console.log(JSON.stringify(process.argv.slice(1).map(f=>purpleFog(fs.readFileSync(f)))));"
    r=subprocess.run(['node','--input-type=module','-e',code,*files],text=True,capture_output=True,check=True)
    assert json.loads(r.stdout)==[expected for _,expected in cases]


def test_guardian_native_observations_and_masks():
    run=subprocess.run([str(ROOT/'build/gfrl_official_native_dump'),'--guardian-decisions'],
                       text=True,capture_output=True,check=True)
    records=[json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
    count=0
    for n in records:
        if n['viewer'] != n['view']['actor']: continue
        g=received_scene(n,1,[])
        pending=n['view']['pending']
        if pending:
            g['attacks'][0]['guardian']=pending['guardian']
            # An explicit received rate must preserve the native observation
            # and every response prefix, including chance guardian payloads.
            g['attacks'][0]['hitRate']=pending['hit_rate']
            if pending['models'][0] in (187,188,202):
                g['attacks'][0]['atk']=pending['amount']
            elif pending['models'][0]==58:
                g['attacks'][0]['atk']=pending['power']
        r=OfficialSession(n['viewer']).ingest(g)
        assert r['status']=='decision', r.get('unsupported')
        p=align_hand(r['public'],n)
        p['decisions']=n['policy_decisions']
        env=OfficialDecision(p,history_len=0)
        expected=token_arrays(n)
        for key,value in expected.items():
            np.testing.assert_allclose(getattr(env,key)[0,:55],value,atol=1e-7,err_msg=str(pending))
        for prefix in n['prefixes']:
            assert int(env.stage[0])==prefix['stage']
            np.testing.assert_array_equal(env.mask[0],prefix['mask'])
            env.step(np.array([prefix['choice']],np.int32))
        assert env.complete
        a=advise_supported(dict(env=OfficialDecision(p),history_complete=False),PreferFirst())
        assert a['status']=='advice'
        if any(x['status']['guardian'] in ('earth','moon') for x in p['players']):
            assert 'earth_moon_general_execution' in a['warnings']
        count+=1
    assert count>100


@pytest.mark.parametrize('guardian,models,expected',[
    ('earth',[30],True),('earth',[6,203],True),('moon',[218],True),
    ('earth',[4,6],False),('moon',[30],False),('moon',[218,232],False),
    ('earth',[207],False),('earth',[58],True)])
def test_typed_guardian_domain(guardian,models,expected):
    assert _core._official_guardian_payload_supported(guardian,models)==expected


def test_dream_discard_exhaustion_restores_only_observed_card_actions():
    p=deepcopy(results()[2]['public'])
    # A disguised weapon prevents prayer; the only discard is disguised too.
    p['hand']=[dict(id=1,face=6,used=False,disguised=True),
               dict(id=2,face=158,used=False,disguised=True)]
    a=advise_supported(dict(env=OfficialDecision(p),history_complete=False),PreferFirst())
    assert a['status']=='advice' and a['dream_last_resort']
    assert a['action']['kind']=='play'
    assert not a['unsupported']
    assert {i for e in a['mask_exclusions'] for i in e.get('item_ids', [])}=={1,2}
    assert 'fallback_samples' not in a


def test_dream_discard_can_be_excluded_with_safe_alternative():
    p=deepcopy(results()[2]['public'])
    p['hand']=[dict(id=1,face=158,used=False,disguised=True),
               dict(id=2,face=6,used=False,disguised=False)]
    class DiscardFirst(PreferFirst):
        def __call__(self,env,rows):
            if int(env.stage[0])==0 and env.mask[0,2]:return np.array([2],np.int32)
            return super().__call__(env,rows)
    a=advise_supported(dict(env=OfficialDecision(p),history_complete=False),DiscardFirst())
    assert a['status']=='advice' and a['action']['cards']==[dict(item_id=2,model=6)]
    assert a['mask_exclusions'][0]['item_ids']==[1]


def test_excluding_first_identical_copy_keeps_later_copy_selectable():
    p=deepcopy(results()[2]['public'])
    p['hand']=[dict(id=i,face=6,used=False,disguised=False) for i in (1,2)]
    env=OfficialDecision(p,excluded_ids=[1])
    a=advise_supported(dict(env=env,history_complete=False),PreferFirst())
    assert a['status']=='advice' and a['action']['cards']==[dict(item_id=2,model=6)]


@pytest.mark.parametrize('guardian',['earth','moon'])
def test_guardian_owner_does_not_block_empty_defense(guardian):
    p=deepcopy(results()[1]['public'])
    p['players'][1]['status']['guardian']=guardian
    class Pass(PreferFirst):
        def __call__(self,env,rows):
            if int(env.stage[0])==1 and env.mask[0,18]:return np.array([18],np.int32)
            return super().__call__(env,rows)
    a=advise_supported(dict(env=OfficialDecision(p),history_complete=False),Pass())
    assert a['status']=='advice' and a['action']['kind']=='defend' and a['action']['cards']==[]
    assert 'earth_moon_general_execution' in a['warnings']
    assert 'fallback_samples' not in a
