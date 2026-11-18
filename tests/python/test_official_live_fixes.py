"""Synthetic public regressions for the auto26–29 and room8 stop classes."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import subprocess
import zlib

import numpy as np
import pytest

from gfrl import _core
from gfrl.official import OfficialDecision, OfficialSession, project_game
from gfrl.official_advisor import advise_supported
from gfrl.official_ui import PATHS, UIUnsupported, prepare, ready, verify_target, verify_resources
from test_official_bot import GAMES, action, results, visible


def fog_scene(response=False):
    p = deepcopy(results()[1 if response else 2]['public'])
    own = next(x for x in p['players'] if x['id'] == p['viewer'])
    other = next(x for x in p['players'] if x['id'] != p['viewer'])
    s = visible(p)
    own['status']['curses'].append('fog')
    other.update(status=None, fogged=True)
    st = own['status']
    s['body'] = f"G.F.{p['gf']}\nHP {st['hp']} MP {st['mp']} ¥ {st['cp']}\n祈る"
    s['fogBands'] = [dict(x=860,y=131,w=340,h=40)]
    s['texts'].append(dict(text='SyntheticName',x=900,y=269,w=100,h=29))
    return p, s


def test_room8_opponent_only_fog_and_own_fog():
    p = deepcopy(results()[1]['public'])
    other = next(x for x in p['players'] if x['id'] != p['viewer'])
    other['status']['curses'].append('fog')
    s = visible(p)
    ready(s,p,'SyntheticName')
    prepare(s,p,action(p,'defend'),'SyntheticName')
    p,s = fog_scene()
    a = action(p,'play',[218],target=2)
    ready(s,p,'SyntheticName')
    prepare(s,p,a,'SyntheticName')
    verify_target(s,p,a,'SyntheticName',350)  # No foreign name/arrow.
    for bad in ('missing_band','visible_text','extra_resources'):
        t = deepcopy(s)
        if bad == 'missing_band': t['fogBands'] = []
        if bad == 'visible_text': t['texts'].append(dict(text='7',x=900,y=140,w=10,h=20))
        if bad == 'extra_resources': t['body'] += '\nHP 1 MP 2 ¥ 3'
        with pytest.raises(UIUnsupported): verify_resources(t,p)


def test_fog_response_requires_visible_attack():
    p,s = fog_scene(True)
    prepare(s,p,action(p,'defend'),'SyntheticName')
    s['images'] = [r for r in s['images'] if r['y'] >= 480]
    with pytest.raises(UIUnsupported,match='visible_attack_mismatch'):
        prepare(s,p,action(p,'defend'),'SyntheticName')


@pytest.mark.parametrize('target,expected',[(None,'none'),(1,'self'),(2,'opponent')])
def test_target_diagnostics_preserve_geometry_without_names(target,expected):
    p = results()[2]['public'];s=visible(p)
    s['texts'].append(dict(text='PrivateForeign',x=660,y=56,w=60,h=23))
    if target in (None, 2):
        verify_target(s,p,dict(target=target),'SyntheticName',350)
    else:
        with pytest.raises(UIUnsupported) as err:
            verify_target(s,p,dict(target=target),'SyntheticName',350)
        d=err.value.detail
        assert d['expected_target']==expected
        assert [b['owner'] for b in d['bands']] == ['self','opponent']
        assert 'PrivateForeign' not in json.dumps(d)


def finish(env, target=2):
    while not env.complete:
        stage=int(env.stage[0]);mask=env.mask[0]
        choice = target if stage==2 and mask[target] else int(np.flatnonzero(mask)[0])
        env.step(np.array([choice],np.int32))
    return env


def test_self_attack_is_unsupported_but_enemy_attack_is_supported():
    p=deepcopy(results()[2]['public'])
    p['hand']=[dict(id=101,face=6,used=False,disguised=False)]
    assert 'self_target_of_enemy_default_unobserved' in finish(OfficialDecision(p),1).action_support()
    assert finish(OfficialDecision(p),2).action_support()==[]


class PreferFirst:
    sample=False
    def __init__(self): self.modes=[];self.masks=[]
    def __call__(self,env,rows):
        self.modes.append(self.sample);self.masks.append((int(env.stage[0]),env.mask.copy()))
        stage=int(env.stage[0]);mask=env.mask[0]
        choice=2 if stage==2 and mask[2] else int(np.flatnonzero(mask)[0])
        return np.array([choice],np.int32)


@pytest.mark.parametrize('greedy',[True,False])
def test_dream_mask_removes_selected_card_preserves_slots_and_mode(greedy):
    p=deepcopy(results()[2]['public'])
    p['hand']=[dict(id=99,face=90,used=False,disguised=True),
               dict(id=100,face=6,used=False,disguised=False),
               dict(id=101,face=7,used=False,disguised=False)]
    actor=PreferFirst();actor.sample=not greedy
    a=advise_supported(dict(env=OfficialDecision(p),history_complete=False),actor, dream='last-resort')
    assert a['status']=='advice' and a['action']['cards']==[dict(item_id=100,model=6)]
    assert a['mask_exclusions']==[dict(reason='dream_actual_execution_not_verified_from_public_faces',item_ids=[99])]
    assert 'fallback_samples' not in a and actor.modes==[not greedy]*len(actor.modes)
    assert any(stage==1 and not mask[0,0] and mask[0,1] for stage,mask in actor.masks)
    assert p['hand'][0]['disguised']  # Original public board and observations retain all cards.


def base_game():
    g=deepcopy(GAMES[0]);g['events']=[dict(action='startGame')];g['attacks']=[]
    g['attackTurnPlayerId']=1;g['gf']=1;g['updateCount']=1
    return g


@pytest.mark.parametrize('kind,delta', [('boostCP',7),('takeCP',70),('boostCPOfEverybody',7)])
def test_cp_public_decision_and_history(kind,delta):
    g=base_game();s=OfficialSession(1);assert s.ingest(g)['history_complete']
    n=deepcopy(g);n['updateCount']=2;n['events']=[
        dict(action='useAttackItems',items=[dict(modelId=6)]),
        dict(action='setTargetPlayer',playerId=2),dict(action=kind,cp=delta)]
    if kind=='takeCP':
        n['players'][0]['cp']=90;n['players'][1]['cp']=0
    elif kind=='boostCP': n['players'][1]['cp']=27
    else:
        for p in n['players']:p['cp']=27
    r=s.ingest(n)
    assert r['history_complete'] and len(s.events)==1
    e=s.events[0]
    assert e['decision']['kind']==0 and e['decision']['target']==2
    assert e['after']['players']==r['public']['players']
    obs=_core._official_observation(r['public'],s.events,64)
    assert any(t==7 for t in obs['obs_type'][55:])  # Decision token.


def test_prayer_and_disease_use_one_public_decision():
    g=base_game();g['players'][0]['curses']=['cold'];s=OfficialSession(1);s.ingest(g)
    n=deepcopy(g);n['updateCount']=2;n['players'][0]['curses']=['fever'];n['players'][0]['hp']-=2
    n['events']=[dict(action='pray'),dict(action='upgradeDisease',playerId=1,curse='fever',natural=True),
                 dict(action='dealDamage',playerId=1,damage=2)]
    r=s.ingest(n)
    assert r['history_complete'] and len(s.events)==1
    assert s.events[0]['decision']['kind']==3 and s.events[0]['decision']['target'] is None
    assert s.events[0]['after']['players'][0]['status']['curses']==['fever']


def test_foreign_unused_overflow_never_projects_truth():
    g=base_game();g['events']=[dict(action='setTargetPlayer',playerId=2),dict(action='sell',
        item=dict(id=4,modelId=6,fakeModelId=0,used=False),
        overflowItem=dict(id=9,modelId=218,fakeModelId=0,used=False))]
    p=project_game(g,1)
    g['events'][1]['overflowItem']['modelId']=296
    assert project_game(g,1)==p
    assert p['events'][1]['overflow_unobserved']


def test_probe_checks_purple_pixels_offline(tmp_path):
    def png(color):
        def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
        return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',340,40,8,2,0,0,0))+
                chunk(b'IDAT',zlib.compress((b'\x00'+bytes(color)*340)*40))+chunk(b'IEND',b''))
    script=Path(__file__).resolve().parents[2]/'scripts/official_probe_pixels.mjs'
    for color,expected in [((133,138,249),'true'),((255,255,255),'false')]:
        f=tmp_path/'overlay.png';f.write_bytes(png(color))
        code=f"import {{purpleFog}} from {json.dumps(script.as_uri())};import fs from 'node:fs';console.log(purpleFog(fs.readFileSync(process.argv[1])));"
        r=subprocess.run(['node','--input-type=module','-e',code,str(f)],text=True,capture_output=True,check=True)
        assert r.stdout.strip()==expected


@pytest.mark.parametrize('scenario',['pray','cp','disease','sale'])
def test_frequent_events_match_native_public_history(scenario):
    root=Path(__file__).resolve().parents[2]
    run=subprocess.run([str(root/'build/gfrl_official_native_dump'),'--public-events',scenario],
                       text=True,capture_output=True,check=True)
    records=[json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
    for viewer in (0,7):
        rows=[r for r in records if r['viewer']==viewer]
        session=OfficialSession(viewer)
        for i,n in enumerate(rows):
            v=n['view'];b=v['board']
            players=[]
            for p in b['players']:
                st=p['status'];items=[dict(id=100+j,modelId=m,fakeModelId=0,used=True)
                                      for j,m in enumerate(p['used_models'])]
                if p['id']==viewer:
                    items=[dict(id=h['token'],modelId=h['face'],fakeModelId=0,used=h['used']) for h in b['hand']]
                players.append(dict(id=p['id'],hp=st['hp'],mp=st['mp'],cp=st['cp'],
                                    curses=st['curses'],guardian=st['guardian'],items=items))
            events=[dict(action='startGame')] if i==0 else []
            if i==1:
                if scenario=='sale': events=[dict(action='useAttackItems',items=[dict(modelId=4),dict(modelId=6)]),
                                             dict(action='setTargetPlayer',playerId=7)]
                elif scenario=='cp': events=[dict(action='useAttackItems',items=[dict(modelId=236)]),
                    dict(action='setTargetPlayer',playerId=0),dict(action='boostMP',playerId=0,mp=-5),
                    dict(action='boostCP',cp=10),dict(action='advanceGF',playerId=7)]
                else:
                    events=[dict(action='pray')]
                    if scenario=='disease':events += [dict(action='upgradeDisease',playerId=0,curse='fever',natural=True),
                                                      dict(action='dealDamage',playerId=0,damage=2)]
                    events.append(dict(action='advanceGF',playerId=7))
            if i==2:
                events=[dict(action='useDefenseItems',items=[]),dict(action='sell',
                    item=dict(id=20,modelId=6,fakeModelId=0,used=False),
                    overflowItem=dict(id=21,modelId=235,fakeModelId=0,used=True)),
                    dict(action='advanceGF',playerId=7)]
            pending=v['pending']
            attacks=[] if pending is None else [dict(playerId=pending['source'],targetPlayerId=pending['target'],
                                                     itemModelIds=pending['models'])]
            g=dict(updateCount=i+1,gf=b['gf'],attackTurnPlayerId=v['owner'],players=players,
                   attacks=attacks,events=events,isOver=False)
            result=session.ingest(g)
            assert result['history_complete'], (scenario,viewer,i,result)
            if i:
                assert session.previous['last_decision']['kind']==({'pray':3,'cp':0,'disease':3,'sale':5}[scenario] if i==1 else 1)
            obs=_core._official_observation(result['public'],session.events,64)
            native=n['tokens'][55:]
            for key,col in [('obs_type',0),('obs_aux_card',1),('obs_cards',2),('obs_cat',3),('obs_num',4)]:
                np.testing.assert_allclose(np.asarray(obs[key])[55:],np.asarray([t[col] for t in native]),atol=1e-7,
                                           err_msg=f'{scenario}/{viewer}/{i}/{key}')


def test_mask_pruning_does_not_create_prayer_or_change_observation():
    p=deepcopy(results()[2]['public'])
    p['hand']=[dict(id=99,face=90,used=False,disguised=True),
               dict(id=100,face=130,used=False,disguised=False)]
    regular=OfficialDecision(p)
    masked=OfficialDecision(p,excluded_ids=[99])
    # A weapon remains in the real hand: excluding its selection cannot permit prayer.
    assert not masked.mask[0,3]
    from gfrl.model import OBS_FIELDS
    for k in OBS_FIELDS:np.testing.assert_array_equal(getattr(regular,k),getattr(masked,k))
    assert finish(masked).semantic_action()['kind'] in ('discard','sacrifice')


def test_disguised_exclusions_accumulate_without_sampling():
    p=deepcopy(results()[2]['public'])
    p['hand']=[dict(id=99,face=90,used=False,disguised=True),
               dict(id=100,face=6,used=False,disguised=True),
               dict(id=101,face=7,used=False,disguised=False)]
    actor=PreferFirst()
    a=advise_supported(dict(env=OfficialDecision(p),history_complete=False),actor, dream='last-resort')
    assert a['status']=='advice' and a['action']['cards']==[dict(item_id=101,model=7)]
    assert [e['item_ids'] for e in a['mask_exclusions']]==[[99],[100]]
    assert not any(actor.modes)


def test_own_unused_eviction_does_not_remove_same_model_used_copy():
    g=base_game();g['attackTurnPlayerId']=2
    g['players'][0]['items']=[dict(id=1,modelId=235,fakeModelId=0,used=True),
                              dict(id=2,modelId=235,fakeModelId=0,used=False)]
    g['players'][1]['items']=[dict(id=3,modelId=4,fakeModelId=0,used=False),
                              dict(id=4,modelId=6,fakeModelId=0,used=False)]
    s=OfficialSession(1);s.ingest(g)
    n=deepcopy(g);n['updateCount']=2;n['attacks']=[dict(playerId=2,targetPlayerId=1,itemModelIds=[4,6])]
    n['events']=[dict(action='useAttackItems',items=[dict(modelId=4),dict(modelId=6)]),
                 dict(action='setTargetPlayer',playerId=1)]
    s.ingest(n)
    f=deepcopy(n);f['updateCount']=3;f['attacks']=[]
    f['players'][0]['cp']-=1;f['players'][1]['cp']+=1
    f['events']=[dict(action='useDefenseItems',items=[]),dict(action='sell',
        item=dict(id=5,modelId=6,fakeModelId=0,used=False),
        overflowItem=dict(id=2,modelId=235,fakeModelId=0,used=False))]
    r=s.ingest(f)
    assert r['history_complete']
    assert s.events[-1]['after']['players'][0]['used']==[235]
    assert r['public']['sale']['overflow_model']==235


def test_no_safe_mask_alternative_restores_dream_card_as_last_resort():
    p=deepcopy(results()[2]['public'])
    p['hand']=[dict(id=99,face=90,used=False,disguised=True)]
    actor=PreferFirst()
    a=advise_supported(dict(env=OfficialDecision(p),history_complete=False),actor,max_samples=2, dream='last-resort')
    assert a['status']=='advice' and a['dream_last_resort']
    assert 'selection_exhausted' not in a
    assert 'dream_actual_execution_not_verified_from_public_faces' in a['warnings']
    assert actor.sample is False and not any(actor.modes)
    assert a['mask_exclusions'][0]['item_ids']==[99]
    assert 'fallback_samples' not in a


def test_card_specific_danger_wide_gap_prunes_wide():
    p=deepcopy(results()[2]['public'])
    p['hand']=[dict(id=99,face=94,used=False,disguised=False),
               dict(id=100,face=232,used=False,disguised=False)]
    class PreferWide(PreferFirst):
        def __call__(self,env,rows):
            if int(env.stage[0])==1 and env.chosen_count[0]>0 and env.mask[0,1]:
                return np.array([1],np.int32)
            return super().__call__(env,rows)
    actor=PreferWide()
    a=advise_supported(dict(env=OfficialDecision(p),history_complete=False),actor)
    assert a['status']=='advice' and a['action']['cards']==[dict(model=94,item_id=99)]
    # The gfrl selection mask itself now forbids 94+232; no UI pruning is needed.
    assert 'mask_exclusions' not in a
    assert not any(actor.modes)
