"""Offline tests for the read-only live spectator and stats dashboard."""
from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
import threading
import urllib.error
import urllib.request

import pytest

from gfrl.official import project_game
from gfrl.ranked_ledger import Ledger
from gfrl.ranked_trace import RankedTrace
from gfrl.ranked_viewer import ViewerStore, dispatch_request, make_handler
from test_official_bot import GAMES


def _seed_campaign(state_dir):
    led = Ledger(state_dir, budget=10, hard_limit=1000, clock=lambda: 1795300000)
    try:
        g = led.reserve(dict(model='m-xit1', ckpt_sha8='f2bd3ea3', code_rev='a41cecb8',
                             policy='greedy', search=False, dream='allow'))
        led.record(g, 'started', opp_kind='human', seat_id=1, end_gf=150,
                   started_at=1795300000, rating_screen=1771)
        led.record(g, 'finish', terminal='completed', outcome='win', first_mover='self',
                   decisions=1, updates=2, gf_final=2, ended_at=1795300060, duration_s=60,
                   rating_display_basis='offset_verified', rating_origin='derived_from_gain',
                   rating_self_before=271, rating_self_after=286, rating_self_first=271,
                   rating_self_last=271, rating_self_gain=15, rating_self_display_before=1771,
                   rating_self_display_after=1786, rating_self_display_before_src='wire',
                   rating_self_display_after_src='wire', rating_opp_before=250, rating_opp_after=235,
                   rating_opp_first=250, rating_opp_last=250, rating_opp_gain=-15,
                   rating_opp_display_before=1750, rating_opp_display_after=1735,
                   rating_opp_display_before_src='wire', rating_opp_display_after_src='wire',
                   rating_screen=1771, rating_screen_check='unavailable', rating_gain_cross='unknown')
    finally:
        led.close()
    tr = RankedTrace(state_dir, clock=lambda: 1795300010)
    try:
        raw = deepcopy(GAMES[0])
        raw['players'][0]['name'] = 'SECRET_PLAYER_A'
        raw['players'][1]['name'] = 'SECRET_PLAYER_B'
        pub = project_game(raw, 1)
        tr.record_update(1, pub)
        tr.record_decision(
            1, pub,
            dict(action=dict(kind='pray', cards=[], target=1, allocation=None), dream_played=False),
            decision_idx=1,
            policy_stats=dict(value=0.3, value_probs=[0.65, 0.35, 0.0], action_logp=-0.1,
                              stage_probs=[0.9], stage_entropies=[0.2]),
        )
    finally:
        tr.close()
    # Append an unterminated trailing line to verify safe mid-write reading.
    with open(state_dir / 'anon_turns.jsonl', 'ab') as f:
        f.write(b'{"ev":"update","partial":')


def test_viewer_store_and_dispatch_read_only(tmp_path):
    state_dir = tmp_path / 'ranked1000'
    _seed_campaign(state_dir)
    store = ViewerStore(state_dir, hard_limit=1000, prev_dir=None)

    code, ctype, body = dispatch_request(store, 'GET', '/')
    assert code == 200 and 'text/html' in ctype
    html = body.decode('utf-8')
    assert 'God Field RL' in html and '/api/live' in html

    code, _, body = dispatch_request(store, 'GET', '/api/catalog')
    assert code == 200 and json.loads(body)['3']['name'] == '両替'

    code, _, body = dispatch_request(store, 'GET', '/api/summary')
    assert code == 200
    summ = json.loads(body)
    assert summ['current']['counts']['wins'] == 1 and summ['current']['rating_self']['last_after'] == 1786
    assert summ['traced_games'] == [1]

    code, _, body = dispatch_request(store, 'GET', '/api/live')
    assert code == 200
    live = json.loads(body)
    assert live['game'] == 1 and len(live['rows']) == 2

    code, _, body = dispatch_request(store, 'GET', '/api/game/1')
    assert code == 200
    replay = json.loads(body)
    assert replay['game'] == 1 and len(replay['rows']) == 2

    for blob in (json.dumps(summ), json.dumps(live), json.dumps(replay)):
        assert 'SECRET_PLAYER_A' not in blob and 'SECRET_PLAYER_B' not in blob

    assert dispatch_request(store, 'GET', '/api/game/999')[0] == 404
    assert dispatch_request(store, 'GET', '/unknown')[0] == 404
    assert dispatch_request(store, 'POST', '/api/summary')[0] == 405
    assert dispatch_request(store, 'DELETE', '/api/live')[0] == 405


def test_viewer_http_server_loopback(tmp_path):
    state_dir = tmp_path / 'ranked1000'
    _seed_campaign(state_dir)
    store = ViewerStore(state_dir, hard_limit=1000, prev_dir=None)
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(store))
    except PermissionError:
        pytest.skip('loopback socket bind blocked by sandbox')
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{port}'
        try:
            html = urllib.request.urlopen(base + '/').read().decode('utf-8')
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, PermissionError):
                pytest.skip('loopback socket connect blocked by sandbox')
            raise
        assert 'God Field RL' in html
        live = json.loads(urllib.request.urlopen(base + '/api/live').read())
        assert live['game'] == 1 and len(live['rows']) == 2
        req = urllib.request.Request(base + '/api/summary', method='POST', data=b'{}')
        with pytest.raises(urllib.error.HTTPError) as e405:
            urllib.request.urlopen(req)
        assert e405.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
