"""Offline synthetic stop regressions; no captures, accounts or private hands."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from gfrl import _core
from gfrl.model import OBS_FIELDS
from gfrl.official import OfficialDecision, OfficialSession, OfficialStreams
from gfrl.official_advisor import advise, advise_supported
from gfrl.official_bot import accepted, ReceiveAdvanced, Journal, private_directory, run
from gfrl.official_ui import (PATHS, UIUnsupported, allocation_steps, execute, point,
                              prepare, ready, same_confirmation, submission_pending)
from test_official_bot import (action, advisor, capture_file, FakeProbe, GAMES,
                               results, settle_clock, visible)
from test_official_live_fixes import base_game, PreferFirst
from test_official_2t_stops import response_scene


def test_exchange_transfers_never_clip_and_reach_requested_allocation():
    rng = random.Random(13)
    pairs = [((92, 6, 21), (50, 60, 9)), ((74, 3, 34), (15, 96, 0)),
             ((0, 99, 99), (99, 99, 0)), ((99, 99, 0), (0, 99, 99))]
    for _ in range(1000):
        start = tuple(rng.randrange(100) for _ in range(3))
        total = sum(start)
        hp = rng.randint(max(0, total-198), min(99, total))
        mp = rng.randint(max(0, total-hp-99), min(99, total-hp))
        pairs.append((start, (hp, mp, total-hp-mp)))
    for start, end in pairs:
        current = dict(zip(('hp', 'mp', 'cp'), start))
        wanted = dict(zip(current, end))
        for command, expected in allocation_steps(current, wanted):
            delta = {123: 10, 183: 1, 303: -1, 363: -10}[command['y']]
            key = {620: 'mp', 760: 'cp'}[command['x']]
            current[key] += delta; current['hp'] -= delta
            assert all(0 <= v <= 99 for v in current.values())
            assert sum(current.values()) == sum(start) and current == expected
        assert current == wanted


@pytest.mark.parametrize('bad', [dict(hp=True, mp=10, cp=20), dict(hp=100, mp=0, cp=0)])
def test_invalid_exchange_start_cannot_authorize_clicks(bad):
    with pytest.raises(UIUnsupported, match='invalid_allocation'):
        allocation_steps(bad, dict(hp=40, mp=10, cp=20))


@pytest.mark.parametrize('advance', [False, True])
def test_response_waits_for_command_overlay_and_stable_scene(settle_clock, advance):
    p, s, a = response_scene()
    s['commandPanes'] = []
    with pytest.raises(UIUnsupported, match='command_control_unavailable'):
        prepare(s, p, a, 'SyntheticName')
    class Probe:
        reads = 0
        clicks = []
        def send(self, command):
            if command['kind'] == 'state':
                self.reads += 1
                if self.reads == 2:
                    s['commandPanes'] = [690]
                    s['texts'].append(dict(text='許す', x=664, y=406, w=52, h=33))
                elif self.reads == 3:
                    # A snapshot straddles the final layout. No mutation yet.
                    s['images'][0]['x'] += 2
            else:
                self.clicks.append(command)
                assert self.reads >= 4
                if command['y'] > 480:
                    s['images'].append(dict(path=PATHS[126], x=545, y=103, w=80, h=80))
            return dict(state=deepcopy(s))
    probe = Probe()
    def refresh():
        if advance and probe.reads >= 1:
            raise ReceiveAdvanced(dict(public=dict(update=p['update']+1)))
    # The caller's ready wait precedes execute's fresh-read stability check.
    from gfrl.official_ui import wait_visible
    if advance:
        with pytest.raises(ReceiveAdvanced):
            wait_visible(probe, s, lambda v: ready(v,p,'SyntheticName'), timeout=3, refresh=refresh)
        assert not probe.clicks
    else:
        state = wait_visible(probe, s, lambda v: ready(v,p,'SyntheticName'), timeout=3, refresh=refresh)
        plan = execute(probe, state, p, a, 'SyntheticName', settle_timeout=3, refresh=refresh)
        assert probe.clicks == [point(888, 635), point(690, 193)]
        assert not plan.get('selection_retries')


@pytest.mark.parametrize('use_used', [False, True])
@pytest.mark.parametrize('bad', [None, 'wrong_suffix', 'moving', 'extra_suffix'])
def test_multiple_distinct_used_miracles_validate_entire_suffix(use_used, bad):
    p = deepcopy(results()[1]['public'])
    p['hand'] = [dict(id=i+10, face=m, used=u, disguised=False)
                 for i, (m,u) in enumerate(((233,False), (214,True), (224,True), (233,True)))]
    s = visible(p)
    s['images'] = [r for r in s['images'] if r['y'] < 480]
    for m,x,y in ((233,190,493), (214,682,595), (224,764,595), (233,846,595)):
        s['images'].append(dict(path=PATHS[m], x=x, y=y, w=80, h=80))
    if bad == 'wrong_suffix': s['images'][-3]['path'] = PATHS[215]
    if bad == 'moving': s['images'][-2]['x'] -= .5
    if bad == 'extra_suffix': s['images'].append(dict(s['images'][-1]))
    a = action(p, 'defend', [233])
    if use_used:
        a['cards'][0]['item_id'] = a['operations'][1]['item_id'] = 13
    if bad:
        with pytest.raises(UIUnsupported, match='ambiguous_used_card_position|visible_hand_mismatch'):
            prepare(s,p,a,'SyntheticName')
    else:
        assert prepare(s,p,a,'SyntheticName')['cards'] == [point(886,635) if use_used else point(230,533)]


def fallback_game(case):
    game = base_game()
    game['attacks'] = [dict(playerId=2, targetPlayerId=1, itemModelIds=[185])]
    game['events'] = [dict(action='nextAttack', attack=deepcopy(game['attacks'][0]))]
    if case in ('dying', 'multiple'):
        game['dyingPlayerIds'] = [2]
        game['players'][1]['hp'] = 0
    if case == 'multiple':
        game['attacks'].append(dict(playerId=2, targetPlayerId=2, itemModelIds=[187]))
    return game


class PolicyOnly:
    config = SimpleNamespace(search_response=True)
    actor = PreferFirst()
    def __call__(self, *args):
        raise AssertionError('partial execution state must never reach search')


@pytest.mark.parametrize('case', ['dying', 'multiple'])
@pytest.mark.parametrize('model', [6, 185])
def test_death_metadata_allows_safe_head_response_and_policy_only(case, model, monkeypatch):
    game = fallback_game(case)
    game['attacks'][0]['itemModelIds'] = [model]
    result = OfficialSession(1).ingest(game)
    assert result['status'] == 'decision' and not result['unsupported']
    assert result['public']['players'][1]['dead'] is False
    assert 'dying_players_decision_unavailable' in result['public']['_policy_only']
    if case == 'multiple':
        assert 'multiple_pending_frames' in result['public']['_policy_only']
    with pytest.raises(ValueError, match='search_requires_public_launch_journal'):
        _core._official_search_root(result['public'], result['env'].events, 64)
    def forbidden(*args):
        raise AssertionError('search root must not be constructed')
    monkeypatch.setattr(_core, '_official_search_root', forbidden)
    a = advise_supported(result, PolicyOnly(), search=True)
    assert a['status'] == 'advice' and a['action']['kind'] == 'defend'
    assert a['search_status'] == ('forced_no_inference' if model == 185 else
                                  'response_execution_state_unavailable_policy_fallback')


@pytest.mark.parametrize('bad', ['dying_actor', 'zero_hp_actor', 'main', 'missing_target', 'dynamic_value'])
def test_partial_death_state_still_stops_when_safe_response_unavailable(bad):
    game = fallback_game('dying')
    if bad == 'dying_actor': game['dyingPlayerIds'] = [1, 2]
    if bad == 'zero_hp_actor': game['players'][0]['hp'] = 0
    if bad == 'main': game['attacks'] = []
    if bad == 'missing_target': game['attacks'][0].pop('targetPlayerId')
    if bad == 'dynamic_value': game['attacks'][0]['itemModelIds'] = [58]
    assert OfficialSession(1).ingest(game)['status'] == 'unsupported'


def redirected_games(kind, model):
    first = base_game()
    first['attacks'] = [dict(playerId=1, targetPlayerId=2, itemModelIds=[model])]
    first['events'] = [dict(action='useAttackItems', items=[dict(modelId=model)]),
                       dict(action='setTargetPlayer', playerId=2), dict(action='hit', playerId=2)]
    second = deepcopy(first)
    second.update(updateCount=2, events=[dict(action=kind, playerId=1)],
                  attacks=[dict(playerId=2, targetPlayerId=1, itemModelIds=[model], reversed=True)])
    return first, second


@pytest.mark.parametrize('kind', ['reflect', 'bounce'])
@pytest.mark.parametrize('model', [95, 106, 226, 229])
def test_redirect_chance_uses_policy_masks_and_explicit_unknown_activation(kind, model):
    session = OfficialSession(1)
    first, second = redirected_games(kind,model)
    assert session.ingest(first)['status'] == 'waiting'
    r = session.ingest(second)
    assert r['status'] == 'decision'
    p = r['public']
    assert p['pending']['forced_hit'] is None
    assert p['pending']['reflections' if kind=='reflect' else 'bounces'] == 1
    assert p['_policy_only'] == ['redirect_chance_activation_unavailable']
    assert advise_supported(r, PolicyOnly(), search=True)['status'] == 'advice'
    # Activation changes the model observation, not any legal response prefix.
    variants = []
    for forced in (None, False, True):
        other = deepcopy(p); other['pending']['forced_hit'] = forced
        env = OfficialDecision(other)
        masks = []
        while not env.complete:
            masks.append(env.mask.copy())
            env.step(np.array([np.flatnonzero(env.mask[0])[0]], np.int32))
        variants.append(masks)
    for masks in variants[1:]:
        for a,b in zip(variants[0], masks, strict=True): np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('case', ['dying', 'multiple', 'reflect', 'bounce'])
def test_fallback_streams_ignore_foreign_private_inventory_and_resume(tmp_path, case):
    sequence = list(redirected_games(case, 229)) if case in ('reflect','bounce') else [fallback_game(case)]
    outcomes = []
    for mutate in (False, True):
        games = deepcopy(sequence)
        if mutate:
            for g in games:
                g['players'][1]['items'] = [dict(id=800,modelId=209), dict(id=801,modelId=207,fakeModelId=6)]
        streams = OfficialStreams(OfficialSession(1))
        r = streams.poll([capture_file(tmp_path,games,serial=1+mutate)])
        assert r['status'] == 'decision' and not streams.failed
        obs = {k:getattr(r['env'],k).copy() for k in (*OBS_FIELDS,'mask')}
        a = advise(r,PreferFirst())['action']
        outcomes.append((r['public'],obs,a))
        nxt = deepcopy(games[-1]); nxt.update(updateCount=nxt['updateCount']+1,
            attacks=[],dyingPlayerIds=[],events=[],attackTurnPlayerId=1)
        nxt['players'][1]['hp'] = 10
        resumed = streams.poll([capture_file(tmp_path,games+[nxt],serial=3+mutate)])
        assert resumed['status'] == 'decision' and not resumed['public'].get('_policy_only')
    assert outcomes[0][0] == outcomes[1][0] and outcomes[0][2] == outcomes[1][2]
    for k in outcomes[0][1]: np.testing.assert_array_equal(outcomes[0][1][k],outcomes[1][1][k])


@pytest.mark.parametrize('delay', [1.5, None])
def test_submitted_command_gets_bounded_receive_only_grace(settle_clock, delay):
    p,s,a = response_scene()
    s['commandPanes'] = [690]
    plan = prepare(s,p,a,'SyntheticName')
    before = deepcopy(s)
    before['images'].append(dict(path=PATHS[126],x=545,y=103,w=80,h=80))
    plan['confirmation_state'] = before
    pending_state = deepcopy(s); pending_state['commandPanes'] = []
    assert submission_pending(pending_state,p,'SyntheticName',plan)
    assert not same_confirmation(pending_state,p,a,'SyntheticName',plan)
    class Probe:
        def send(self,c):
            assert c['kind'] == 'capture'  # Never reconstruct a submitted command.
            return dict(capture=dict(captures=[]))
    class Streams:
        def poll(self, captures):
            return dict(status='waiting',public=dict(update=p['update']+(delay is not None and settle_clock.now>=delay)))
    pending_checks = []
    def pending():
        pending_checks.append(settle_clock.now)
        return submission_pending(pending_state,p,'SyntheticName',plan)
    if delay is None:
        with pytest.raises(UIUnsupported,match='acceptance_timeout'):
            accepted(Probe(),Streams(),p['update'],timeout=1,pending=pending)
        assert settle_clock.now == pytest.approx(31)
    else:
        assert accepted(Probe(),Streams(),p['update'],timeout=1,pending=pending)['public']['update'] == p['update']+1
    assert len(pending_checks) == 1


@pytest.mark.parametrize('bad', ['no_capability','control_present','wrong_pane','selection_present','gf'])
def test_submission_grace_requires_observed_submit_transition(bad):
    p,s,a = response_scene(); s['commandPanes'] = [690]
    plan = prepare(s,p,a,'SyntheticName'); plan['confirmation_state'] = deepcopy(s)
    s['commandPanes'] = []
    if bad == 'no_capability': plan['confirmation_state'].pop('commandPanes')
    if bad == 'control_present': s['commandPanes'] = [690]
    if bad == 'wrong_pane': s['texts'][0]['x'] -= 340
    if bad == 'selection_present': s['images'].append(dict(path=PATHS[126],x=545,y=103,w=80,h=80))
    if bad == 'gf': s['body'] = s['body'].replace(f"G.F.{p['gf']}",'G.F.999')
    assert not submission_pending(s,p,'SyntheticName',plan)


def test_probe_command_controls_use_visible_geometry_style_and_hit_test_only():
    module = Path(__file__).resolve().parents[2]/'scripts/official_probe_controls.mjs'
    code = r'''
import {commandPanes} from MODULE;
let candidates=[], hit=null;
globalThis.getComputedStyle=e=>e.style;
globalThis.document={elementFromPoint:(x,y)=>{
  if (![210,550].includes(x) || y!==108) throw Error('hit-test must clear the rounded corner');
  return hit;
}};
const page={locator:s=>{if(s!=='div:visible')throw Error();return {evaluateAll:f=>f(candidates)}}};
const make=(x=195)=>({getBoundingClientRect:()=>({x,y:93,width:310,height:300}),
  style:{zIndex:'1',cursor:'pointer',pointerEvents:'auto',visibility:'visible',opacity:'1'}, contains:e=>e===hit});
const seen=[];
let e=make();candidates=[e];hit={};seen.push(await commandPanes(page));
e=make(535);candidates=[e];seen.push(await commandPanes(page));
for (const [key,value] of [['zIndex','auto'],['cursor',''],['pointerEvents','none'],['visibility','hidden'],['opacity','0']]) {
  e=make();e.style[key]=value;candidates=[e];seen.push(await commandPanes(page));
}
e=make(200);candidates=[e];seen.push(await commandPanes(page));
e=make();e.contains=()=>false;candidates=[e];seen.push(await commandPanes(page));
console.log(JSON.stringify(seen));
'''.replace('MODULE', json.dumps(module.as_uri()))
    r = subprocess.run(['node','--input-type=module','-e',code],check=True,text=True,capture_output=True)
    assert json.loads(r.stdout) == [[350],[690]]+[[]]*7


def test_bot_continues_through_dying_response(tmp_path, settle_clock):
    games = deepcopy(GAMES)
    games[1]['dyingPlayerIds'] = [2]
    games[1]['players'][1]['hp'] = 0
    probe = FakeProbe(tmp_path, games=games)
    journal = Journal(private_directory(tmp_path/'log'))
    report = run(probe, advisor, journal, name='SyntheticName', timeout=1, settle_timeout=1)
    journal.close()
    assert report['status'] == 'complete' and report['games'] == 1
    rows = [json.loads(line) for line in (journal.directory/'moves.jsonl').read_text().splitlines()]
    assert rows[1]['status'] == 'accepted' and rows[1]['state']['phase'] == 1


@pytest.mark.parametrize('settles', [False, True])
def test_bot_waits_for_used_suffix_before_any_card_click(tmp_path, settle_clock, settles):
    games = deepcopy(GAMES)
    games[2]['players'][0]['items'] += [dict(id=100001,modelId=214,used=True),
                                       dict(id=100002,modelId=218,used=True)]
    class Probe(FakeProbe):
        reads = 0
        def state(self):
            s = super().state()
            if self.phase == 'battle' and self.index == 2:
                self.reads += 1
                s['images'] = [r for r in s['images'] if not
                               (r['y']>=480 and r['path'] in (PATHS[214],PATHS[218]))]
                x = 764 if settles and self.reads >= 3 else 763.5
                s['images'] += [dict(path=PATHS[m],x=cx,y=595,w=80,h=80)
                                for m,cx in ((218,190),(214,x),(218,846))]
            return s
    probe = Probe(tmp_path, games=games)
    journal = Journal(private_directory(tmp_path/'log'))
    report = run(probe, advisor, journal, name='SyntheticName', timeout=1, settle_timeout=1)
    journal.close()
    assert probe.reads >= 3
    if settles:
        assert report['status'] == 'complete' and point(230,635) in probe.commands
    else:
        assert report['reason'] == 'ambiguous_used_card_position'
        assert point(230,635) not in probe.commands
        assert json.loads((journal.directory/'moves.jsonl').read_text().splitlines()[-1])['clicks'] == []


def test_bot_waits_for_delayed_receive_without_rebuilding_submitted_command(tmp_path, settle_clock):
    class Probe(FakeProbe):
        release_at = None
        confirms = 0
        def delayed(self):
            return self.release_at is not None and settle_clock.now < self.release_at
        def state(self):
            if self.delayed():
                s = visible(self.rs[self.index-1]['public'])
                s['commandPanes'] = []
                return s
            s = super().state()
            if self.phase == 'battle':
                s['commandPanes'] = [690 if self.index in (1,4,5) else 350]
            return s
        def send(self,c):
            if (self.phase == 'battle' and c['kind']=='clickPoint'
                    and c['x'] in (350,690) and c['y'] in (193,250)):
                self.confirms += 1
                if self.confirms == 1:
                    self.release_at = settle_clock.now + 1.5
            if c['kind']=='capture' and self.delayed():
                self.index -= 1
                try:
                    return super().send(c)
                finally:
                    self.index += 1
            return super().send(c)
    probe = Probe(tmp_path)
    journal = Journal(private_directory(tmp_path/'log'))
    report = run(probe, advisor, journal, name='SyntheticName', timeout=1, settle_timeout=1)
    journal.close()
    assert report['status'] == 'complete' and probe.confirms == 5
    rows = [json.loads(line) for line in (journal.directory/'moves.jsonl').read_text().splitlines()]
    assert rows[0]['acceptance_wait_extended'] and rows[0]['accepted']
    assert rows[0]['clicks'].count(point(230,533)) == 1
    assert rows[0]['clicks'].count(point(350,250)) == 1


@pytest.mark.parametrize('multi', [False, True])
def test_redirect_requires_history_policy_only_without_inventing_origin(multi):
    attacks = [dict(playerId=2, targetPlayerId=1, itemModelIds=[184], atk=22, reversed=True)]
    if multi:
        attacks.append(dict(playerId=1, targetPlayerId=2, itemModelIds=[6], atk=3))
    outcomes = []
    for opp_model in (30, 296):
        g = base_game()
        g['attacks'] = deepcopy(attacks)
        g['players'][1]['items'] = [dict(id=999, modelId=opp_model, used=False)]
        r = OfficialSession(1).ingest(g)
        assert r['status'] == 'decision'
        p = r['public']
        assert p['pending']['origin'] is None and p['pending']['forced_hit'] is None
        assert p['pending']['reflections'] == 0 and p['pending']['bounces'] == 0
        assert 'redirect_requires_history' in p['_policy_only']
        assert '_search_launch' not in p
        assert advise_supported(r, PolicyOnly(), search=True)['status'] == 'advice'
        with pytest.raises(ValueError, match='search_requires_public_launch_journal'):
            _core._official_search_root(p, [], 64, {})
        obs = {k: getattr(r['env'], k).copy() for k in (*OBS_FIELDS, 'mask')}
        outcomes.append((p, obs, advise(r, PreferFirst())['action']))
    assert outcomes[0][0] == outcomes[1][0] and outcomes[0][2] == outcomes[1][2]
    for k in outcomes[0][1]:
        np.testing.assert_array_equal(outcomes[0][1][k], outcomes[1][1][k])


@pytest.mark.parametrize('mode', ['foreign_actor', 'own_dying', 'own_hp0', 'other_unsupported'])
def test_redirect_requires_history_keeps_fail_closed_boundaries(mode):
    g = base_game()
    if mode == 'foreign_actor':
        g['attacks'] = [dict(playerId=1, targetPlayerId=2, itemModelIds=[184], atk=22, reversed=True)]
        r = OfficialSession(1).ingest(g)
        assert r['status'] == 'waiting' and 'redirect_requires_history' in r['unsupported']
        return
    g['attacks'] = [dict(playerId=2, targetPlayerId=1, itemModelIds=[184], atk=22, reversed=True)]
    if mode == 'own_dying':
        g['dyingPlayerIds'] = [1]
    elif mode == 'own_hp0':
        g['players'][0]['hp'] = 0
    elif mode == 'other_unsupported':
        g['attacks'] = [dict(playerId=2, targetPlayerId=1, itemModelIds=[58], reversed=True)]
    r = OfficialSession(1).ingest(g)
    assert r['status'] == 'unsupported' and 'redirect_requires_history' in r['unsupported']
