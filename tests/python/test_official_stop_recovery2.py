"""Second stop-recovery regressions. All games and UI scenes are synthetic."""
from copy import deepcopy
import json

import numpy as np
import pytest

from gfrl import _core
from gfrl.model import OBS_FIELDS
from gfrl.official import OfficialDecision, OfficialSession, OfficialStreams, parser_module
from gfrl.official_advisor import advise, advise_supported
from gfrl.official_bot import (Journal, accepted, blank_scene, enter_training, poll_state,
                              private_directory, run, training_loading)
from gfrl.official_ui import UIUnsupported, prepare
from test_official_bot import FakeProbe, advisor, capture_file, settle_clock, visible
from test_official_live_fixes import base_game, PreferFirst
from test_official_stop_recovery import PolicyOnly


def death_game():
    g = base_game()
    g['events'] = []
    g['attacks'] = [dict(playerId=2, targetPlayerId=1, itemModelIds=[112], atk=30, hitRate=75)]
    g['dyingPlayerIds'] = [2]
    g['players'][1]['hp'] = 0
    return g


@pytest.mark.parametrize('viewer', [1, 2])
def test_death_bow_public_power_rate_policy_and_ui(viewer):
    g = death_game()
    if viewer == 2:
        g['players'][0]['id'], g['players'][1]['id'] = 2, 1
        g['attackTurnPlayerId'] = 2
        g['attacks'][0].update(playerId=1, targetPlayerId=2)
        g['dyingPlayerIds'] = [1]
    r = OfficialSession(viewer).ingest(parser_module().sanitize_game(g, viewer))
    assert r['status'] == 'decision'
    p, env = r['public'], r['env']
    assert set(p['_policy_only']) == {'dynamic_hit_rate_unavailable',
        'death_attack_decision_unavailable', 'dying_players_decision_unavailable'}
    assert p['pending']['death'] is False  # No invented execution provenance.
    assert p['pending']['forced_hit'] is None
    names = env.obs_schema()['num']
    assert env.obs_num[0, 21, names['hit_rate']] == pytest.approx(.75)
    assert env.obs_num[0, 21, names['power_log1p']] == pytest.approx(np.log1p(30))
    with pytest.raises(ValueError, match='search_requires_public_launch_journal'):
        _core._official_search_root(p, env.events, 64)
    unmarked = deepcopy(p); unmarked.pop('_policy_only')
    with pytest.raises(ValueError, match='search_requires_public_launch_journal'):
        _core._official_search_root(unmarked, env.events, 64)
    a = advise_supported(r, PolicyOnly(), search=True)
    assert a['status'] == 'advice' and a['action']['kind'] == 'defend'
    s = visible(p)
    # Put the synthetic own name in the response pane, preserving attack face.
    s['texts'][0]['x'] = 660
    for row in s['images']:
        if row['y'] == 103:
            row['x'] = 205
    s['commandPanes'] = [690]
    prepare(s, p, a['action'], 'SyntheticName')


def response_paths(public):
    """Enumerate every selection prefix, including non-first card choices."""
    paths, pending = {}, [()]
    while pending:
        prefix = pending.pop()
        env = OfficialDecision(public)
        for c in prefix:
            env.step(np.array([c], np.int32))
        paths[prefix] = (env.complete, env.mask.tolist())
        if not env.complete:
            pending.extend(prefix + (int(c),) for c in np.flatnonzero(env.mask[0]))
    return paths


def test_response_legality_independent_of_rate_death_and_activation():
    g = death_game()
    # Armor, reflection and a miracle, with both affordable and costly choices.
    g['players'][0]['items'] = [dict(id=i+1, modelId=m) for i,m in enumerate((38, 190, 189, 233))]
    r = OfficialSession(1).ingest(g)
    baseline = response_paths(r['public'])
    assert len(baseline) > 3
    for rate, death, forced in ((25, False, False), (75, True, True), (100, False, None)):
        p = deepcopy(r['public'])
        p['pending'].update(hit_rate=rate, death=death, forced_hit=forced)
        assert response_paths(p) == baseline


@pytest.mark.parametrize('bad', ['power_absent', 'power_zero', 'own_dying', 'own_hp0',
                                  'own_dead', 'target_absent', 'unknown_key', 'unknown_event'])
def test_death_fallback_keeps_unsafe_states_closed(bad):
    g = death_game()
    if bad == 'power_absent': g['attacks'][0].pop('atk')
    if bad == 'power_zero': g['attacks'][0]['atk'] = 0
    if bad == 'own_dying': g['dyingPlayerIds'] = [1, 2]
    if bad == 'own_hp0': g['players'][0]['hp'] = 0
    if bad == 'own_dead': g['players'][0]['dead'] = True
    if bad == 'target_absent': g['attacks'][0].pop('targetPlayerId')
    if bad == 'unknown_key': g['players'][0]['futureStatus'] = 1
    if bad == 'unknown_event': g['events'] = [dict(action='unsupported')]
    parser = parser_module()
    if bad in ('unknown_key', 'unknown_event'):
        with pytest.raises(parser.ParseError): parser.sanitize_game(g, 1)
    else:
        assert OfficialSession(1).ingest(parser.sanitize_game(g, 1))['status'] == 'unsupported'


def test_death_streams_ignore_private_inventory_resume_and_clear_fallback(tmp_path):
    outcomes = []
    for mutate in (False, True):
        g = death_game()
        if mutate:
            g['players'][1]['items'] = [dict(id=800, modelId=209), dict(id=801, modelId=207, fakeModelId=6)]
        streams = OfficialStreams(OfficialSession(1))
        r = streams.poll([capture_file(tmp_path, [g], serial=1+mutate)])
        assert r['status'] == 'decision'
        outcomes.append((r['public'], {k:getattr(r['env'], k) for k in (*OBS_FIELDS, 'mask')},
                         advise(r, PreferFirst())['action']))
        nxt = deepcopy(g)
        nxt.update(updateCount=2, attacks=[], dyingPlayerIds=[], attackTurnPlayerId=1)
        nxt['players'][1]['hp'] = 10
        r = streams.poll([capture_file(tmp_path, [g, nxt], serial=3+mutate)])
        assert r['status'] == 'decision' and not r['public'].get('_policy_only')
    a,b = outcomes
    assert a[0] == b[0] and a[2] == b[2]
    for k in a[1]: np.testing.assert_array_equal(a[1][k], b[1][k])


@pytest.mark.parametrize('count', [0, 1, 3, 9007199254740991])
@pytest.mark.parametrize('player', [0, 1])
def test_confusion_count_is_validated_then_reduced_to_visible_boolean(tmp_path, count, player):
    g = base_game()
    g['players'][player]['confusionCount'] = count
    streams = OfficialStreams(OfficialSession(1))
    r = streams.poll([capture_file(tmp_path, [g])])
    assert r['status'] == ('waiting' if player == 0 and count else 'decision')
    assert r['public']['players'][player]['status']['confused'] == bool(count)
    assert 'confusionCount' not in json.dumps(r['public'])
    if count and player == 1:
        a = advise_supported(r, PolicyOnly(), search=True)
        assert a['search_status'] == 'confusion_execution_state_unavailable_policy_fallback'
        prepare(visible(r['public']), r['public'], a['action'], 'SyntheticName')


@pytest.mark.parametrize('value', [True, False, -1, 1.0, '1', None, [], {}, 9007199254740992])
@pytest.mark.parametrize('lenient', [False, True])
def test_invalid_confusion_is_not_swallowed_by_lenient_events(value, lenient):
    g = base_game(); g['players'][1]['confusionCount'] = value
    parser = parser_module()
    with pytest.raises(parser.ParseError) as e:
        parser.sanitize_game(g, 1, lenient_events=lenient)
    assert e.value.detail['path'] == 'game.players[].confusionCount'


@pytest.mark.parametrize('where', ['game', 'attack', 'event', 'item'])
def test_confusion_key_is_not_allowed_outside_player(where):
    g = base_game()
    obj = {'game': g, 'event': g['events'][0], 'item': g['players'][0]['items'][0]}
    g['attacks'] = [dict(playerId=2, targetPlayerId=1, itemModelIds=[6])]
    obj['attack'] = g['attacks'][0]
    obj[where]['confusionCount'] = 1
    parser = parser_module()
    with pytest.raises(parser.ParseError): parser.sanitize_game(g, 1)


def test_confusion_under_fog_is_hidden_dead_is_false_and_response_is_safe():
    g = death_game(); g['players'][0]['curses'] = ['fog']
    variants = []
    for count in (0,3):
        g['players'][1]['confusionCount'] = count
        r = OfficialSession(1).ingest(parser_module().sanitize_game(g,1))
        variants.append(r['public'])
    assert variants[0] == variants[1]
    g['players'][0]['curses'] = []
    g['players'][1]['dead'] = True
    assert not OfficialSession(1).ingest(g)['public']['players'][1]['status']['confused']
    g['players'][0]['confusionCount'] = 3
    r = OfficialSession(1).ingest(g)
    assert r['status'] == 'decision'
    a = advise_supported(r, PolicyOnly(), search=True)
    s = visible(r['public']); s['texts'][0]['x'] = 660
    for row in s['images']:
        if row['y'] == 103: row['x'] = 205
    prepare(s,r['public'],a['action'],'SyntheticName')


@pytest.mark.parametrize('recovers', [False, True])
def test_training_shell_gets_one_bounded_wait_without_extra_clicks(settle_clock, recovers):
    shell = dict(body='修行\n教典', inputs=[], images=[], texts=[], commandPanes=[])
    class Probe:
        commands = []
        def send(self,c):
            self.commands.append(c)
            if c.get('text') == '修行': self.entered = True
            if not getattr(self, 'entered', False):
                return dict(state=dict(body='コンピュータと対戦\n修行'))
            if recovers and settle_clock.now >= 2:
                return dict(state=dict(body='修行者 2 人\n終末の時\nなし\n戦いを始める'))
            return dict(state=deepcopy(shell))
    p = Probe()
    if recovers:
        assert '戦いを始める' in enter_training(p,'SyntheticName',settle_timeout=1)['body']
    else:
        with pytest.raises(UIUnsupported,match='visible_state_timeout'):
            enter_training(p,'SyntheticName',settle_timeout=1)
        assert settle_clock.now == pytest.approx(31)
    assert [c for c in p.commands if c['kind'] != 'state'] == [dict(kind='clickText',text='修行')]


def test_unrelated_screen_gets_no_loading_grace(settle_clock):
    class Probe:
        def send(self,c): return dict(state=dict(body='unknown', commandPanes=[]))
    with pytest.raises(UIUnsupported,match='visible_state_timeout'):
        poll_state(Probe(),lambda s:False,timeout=1,loading=training_loading)
    assert settle_clock.now == pytest.approx(1)


@pytest.mark.parametrize('recovers', [False, True])
def test_blank_after_exchange_confirmation_waits_without_resending(tmp_path, settle_clock, recovers):
    class BlankAfterSubmit(FakeProbe):
        submitted = None
        def send(self,c):
            if c['kind'] == 'clickPoint' and c['y'] == 250 and self.submitted is None:
                self.submitted = settle_clock.now
            if self.submitted is not None and (not recovers or settle_clock.now < self.submitted+2):
                if c['kind'] == 'capture': return dict(capture=dict(captures=[]))
                if c['kind'] == 'state':
                    return dict(state=dict(body='',texts=[],images=[],inputs=[],commandPanes=[]))
            return super().send(c)
        def state(self):
            s = super().state()
            if self.phase == 'battle':
                p = self.rs[self.index]['public']
                s['commandPanes'] = [690 if p['update'] in (2,5,6) else 350]
            return s
    p = BlankAfterSubmit(tmp_path)
    d = private_directory(tmp_path/'journal'); j = Journal(d)
    report = run(p,advisor,j,name='SyntheticName',timeout=1,settle_timeout=1)
    j.close()
    rows = [json.loads(l) for l in (d/'moves.jsonl').read_text().splitlines()]
    if recovers:
        assert report['status'] == 'complete'
        assert rows[0]['acceptance_wait_extended'] and rows[0]['acceptance_blank_wait']
    else:
        assert report['reason'] == 'acceptance_timeout'
        assert rows[0]['acceptance_wait_extended'] and rows[0]['acceptance_blank_wait']
        assert not rows[0]['accepted']
        assert settle_clock.now == pytest.approx(31)
    assert p.commands.count(dict(kind='clickPoint', x=350, y=250)) == 1


@pytest.mark.parametrize('key,value', [('body','G.F.1'), ('texts',[dict(text='x')]),
    ('images',[dict(path='/images/items/3.webp')]), ('inputs',[{}]), ('commandPanes',[350])])
def test_blank_grace_never_matches_a_partial_battle(key,value):
    s = dict(body='',texts=[],images=[],inputs=[],commandPanes=[])
    assert blank_scene(s)
    s[key] = value
    assert not blank_scene(s)


def test_receive_only_grace_disables_late_confirmation_retry(settle_clock):
    class Probe:
        def send(self, c):
            assert c['kind'] == 'capture'
            return dict(capture=dict(captures=[]))
    class Streams:
        def poll(self, captures):
            return dict(status='waiting', public=dict(update=int(settle_clock.now >= 9)))
    def forbidden_retry():
        raise AssertionError('receive-only grace must not retry confirmation')
    assert accepted(Probe(), Streams(), 0, timeout=1, pending=lambda: True,
                    retry=forbidden_retry)['public']['update'] == 1


def test_death_head_never_inherits_the_lethal_attack_source_or_power():
    g = base_game()
    session = OfficialSession(1)
    session.ingest(g)
    nxt = death_game()
    nxt.update(updateCount=2, events=[dict(action='useAttackItems', items=[dict(modelId=58)]),
        dict(action='setTargetPlayer', playerId=2), dict(action='unknownEvent')])
    r = session.ingest(nxt)
    assert r['status'] == 'decision'
    assert r['public']['pending']['source'] == r['public']['pending']['origin'] == 2
    assert r['public']['pending']['base_override'] == 30
    assert not r['history_complete']
    nxt['attacks'][0].pop('atk')
    assert OfficialSession(1).ingest(nxt)['status'] == 'unsupported'
