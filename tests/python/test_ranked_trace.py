"""Offline tests for anonymous ranked traces (ranked_trace) and extended 1000-game campaigns."""
from copy import deepcopy
import json
import os
import stat

import numpy as np
import pytest
import torch

from gfrl.actor import Actor
from gfrl.model import Model, ModelConfig
from gfrl.official import OfficialSession, project_game
from gfrl.official_advisor import advise_supported
from gfrl.official_bot import run
from gfrl.official_ranked import NullJournal, RankedSession
from gfrl.ranked_campaign import report_from_path
from gfrl.ranked_drive import APPROVED, Driver, main as drive_main
from gfrl.ranked_ledger import HARD_LIMIT, MAX_CAMPAIGN_LIMIT, Ledger, LedgerRefused
from gfrl.ranked_trace import RankedTrace, TracingActor, build_decision_row, build_update_row, validate_trace_row
from test_official_bot import GAMES
from test_official_ranked import COND, RankedProbe, duel_games, session


def test_trace_rows_strip_identities_item_ids_and_foreign_private_cards():
    raw = deepcopy(GAMES[0])
    raw['players'][0]['name'] = 'SECRET_SELF_NAME'
    raw['players'][1]['name'] = 'SECRET_OPP_NAME'
    raw['players'][1]['items'] = [dict(id=998877, modelId=296, used=False),
                                  dict(id=998878, modelId=209, used=True)]
    public = project_game(raw, 1)
    row = build_update_row(7, public, t=1795200000)
    text = json.dumps(row)
    assert all(s not in text for s in ('SECRET_SELF_NAME', 'SECRET_OPP_NAME', '998877', '998878', '296'))
    assert row['ev'] == 'update' and row['game'] == 7 and row['owner_self'] is True
    assert row['opp']['used'] == [209]
    assert all(len(h) == 3 and h[1] in (0, 1) and h[2] in (0, 1) for h in row['hand'])
    # Mutating opponent's unused hand does not change the trace row at all.
    raw2 = deepcopy(raw)
    raw2['players'][1]['items'][0]['modelId'] = 30
    assert build_update_row(7, project_game(raw2, 1), t=1795200000) == row


def test_validate_trace_row_rejects_unknown_keys_and_free_text():
    raw = deepcopy(GAMES[0])
    good = build_update_row(1, project_game(raw, 1), t=1795200000)
    assert validate_trace_row(good) == good
    for mut in (
        lambda r: r.update(extra='nope'),
        lambda r: r.update(ev='other'),
        lambda r: r['self'].update(guardian='SecretGuardian'),
        lambda r: r['hand'].append([999, 0, 0]),
        lambda r: r['events'].append(dict(action='freeTextEvent')),
        lambda r: r.update(policy_only=['not_allowed']),
    ):
        bad = deepcopy(good)
        mut(bad)
        with pytest.raises(ValueError):
            validate_trace_row(bad)
    dec = build_decision_row(
        1, project_game(raw, 1),
        dict(action=dict(kind='pray', cards=[], target=1, allocation=None), dream_played=False),
        decision_idx=1,
        policy_stats=dict(value=0.25, value_probs=[0.6, 0.35, 0.05], action_logp=-0.2,
                          stage_probs=[0.8], stage_entropies=[0.5]),
        t=1795200001,
    )
    assert dec['value'] == 0.25 and dec['action']['target_self'] is True
    bad_dec = deepcopy(dec); bad_dec['value'] = float('nan')
    with pytest.raises(ValueError):
        validate_trace_row(bad_dec)


def test_tracing_actor_preserves_choices_and_records_stats():
    torch.manual_seed(0)
    model = Model(ModelConfig(d=32, depth=1, heads=2, ff=64)).eval()
    r1 = OfficialSession(1).ingest(deepcopy(GAMES[0]))
    r2 = OfficialSession(1).ingest(deepcopy(GAMES[0]))
    assert r1['status'] == 'decision'
    plain = Actor(model, r1['env'], 'cpu', sample=False)
    traced = TracingActor(Actor(model, r2['env'], 'cpu', sample=False))
    a1 = advise_supported(r1, plain, search=False)
    a2 = advise_supported(r2, traced, search=False)
    stats = traced.pop_stats()
    assert a1['action'] == a2['action']
    assert stats is not None and -1.0 <= stats['value'] <= 1.0
    assert len(stats['value_probs']) == 3 and abs(sum(stats['value_probs']) - 1.0) < 1e-3
    assert len(stats['stage_probs']) >= 1 and all(0.0 <= p <= 1.0 for p in stats['stage_probs'])


def test_ranked_run_writes_private_anonymous_trace(tmp_path, monkeypatch):
    monkeypatch.setattr('time.sleep', lambda *_: None)
    state_dir = tmp_path / 'ranked1000'
    now = 1795200000
    ledger = Ledger(state_dir, budget=10, hard_limit=1000, clock=lambda: now)
    trace = RankedTrace(state_dir, clock=lambda: now)
    s = session(ledger, trace=trace)
    probe = RankedProbe(tmp_path, games=duel_games())
    from gfrl.official import OfficialDecision
    from test_official_bot import advisor as script_advisor
    torch.manual_seed(0)
    model = Model(ModelConfig(d=32, depth=1, heads=2, ff=64)).eval()

    def advisor(result):
        # Compute real TracingActor stats on a fresh OfficialDecision while
        # returning the scripted probe action so RankedProbe runs to completion.
        env_copy = OfficialDecision(result['env'].public, result['env'].events, result['env'].history_len)
        tracer = TracingActor(Actor(model, env_copy, 'cpu', sample=False))
        advise_supported(dict(result, env=env_copy), tracer, search=False, dream='allow')
        s.set_policy_stats(tracer.pop_stats())
        return script_advisor(result)

    try:
        report = run(probe, advisor, NullJournal(), name=s.name, games=1, mode='ranked',
                     ranked=s, timeout=.01, settle_timeout=.01)
    finally:
        trace.close()
        ledger.close()
    assert report['status'] == 'complete', report
    trace_file = state_dir / 'anon_turns.jsonl'
    assert stat.S_IMODE(trace_file.stat().st_mode) == 0o600
    rows = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
    assert any(r['ev'] == 'update' for r in rows)
    assert any(r['ev'] == 'decision' for r in rows)
    assert all(validate_trace_row(r) == r for r in rows)
    raw_text = trace_file.read_text()
    assert s.name not in raw_text and 'Synthetic' not in raw_text


def test_extended_1000_limit_ledger_driver_and_report(tmp_path):
    state_dir = tmp_path / 'ranked1000'
    state_dir.mkdir(mode=0o700)
    ckpt = tmp_path / 'm-xit1.pt'
    ckpt.write_bytes(b'fake-checkpoint')
    pw = tmp_path / 'pw'
    pw.mkdir()
    now = [1795300000.0]
    launches = []

    def fake_launcher(argv, timeout):
        launches.append(argv)
        assert '--trace' in argv
        assert '--hard-limit' in argv and argv[argv.index('--hard-limit') + 1] == '15'
        budget = int(argv[argv.index('--budget') + 1])
        hard_limit = int(argv[argv.index('--hard-limit') + 1])
        led = Ledger(state_dir, budget, hard_limit=hard_limit, clock=lambda: int(now[0]))
        try:
            g = led.reserve(dict(model='m-xit1', ckpt_sha8=Driver(playwright=pw, checkpoint=ckpt).read.__name__[:8]
                                 if False else __import__('hashlib').sha256(ckpt.read_bytes()).hexdigest()[:8],
                                 code_rev='c871ba70', policy='greedy', search=False, dream='allow'))
            led.record(g, 'started', opp_kind='human', seat_id=1, end_gf=150,
                       started_at=int(now[0]), rating_screen=1771 + (g - 1) * 10)
            now[0] += 60
            led.record(g, 'finish', terminal='completed', outcome='win', first_mover='self',
                       decisions=5, updates=10, gf_final=5, ended_at=int(now[0]), duration_s=60,
                       rating_display_basis='offset_verified', rating_origin='derived_from_gain',
                       rating_self_before=271 + (g - 1) * 10, rating_self_after=281 + (g - 1) * 10,
                       rating_self_first=271 + (g - 1) * 10, rating_self_last=271 + (g - 1) * 10,
                       rating_self_gain=10, rating_self_display_before=1771 + (g - 1) * 10,
                       rating_self_display_after=1781 + (g - 1) * 10,
                       rating_self_display_before_src='wire', rating_self_display_after_src='wire',
                       rating_opp_before=200, rating_opp_after=190, rating_opp_first=200,
                       rating_opp_last=200, rating_opp_gain=-10,
                       rating_opp_display_before=1700, rating_opp_display_after=1690,
                       rating_opp_display_before_src='wire', rating_opp_display_after_src='wire',
                       rating_screen=1771 + (g - 1) * 10, rating_screen_check='unavailable',
                       rating_gain_cross='unknown')
        finally:
            led.close()
        return 0, json.dumps(dict(status='complete')), False

    lines = []
    rc = drive_main(
        ['--playwright', str(pw), '--checkpoint', str(ckpt), '--state-dir', str(state_dir),
         '--target', '15', '--gap', '60', '--trace'],
        launcher=fake_launcher, process_lister=lambda: [],
        clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s),
        emit=lambda line: lines.append(json.loads(line)),
    )
    assert rc == 0
    assert lines[-1]['status'] == 'done' and lines[-1]['counted'] == 15
    assert [int(a[a.index('--budget') + 1]) for a in launches] == [APPROVED] * APPROVED + [15] * 5
    report = report_from_path(state_dir / 'ledger.jsonl', hard_limit=15, expect_preliminary=False)
    assert report['counts']['counted'] == 15 and report['warnings'] == []
    assert report['rating']['mismatches'] == []

    # Cap boundary: MAX_CAMPAIGN_LIMIT accepted, MAX_CAMPAIGN_LIMIT + 1 refused.
    assert MAX_CAMPAIGN_LIMIT == 1000 and HARD_LIMIT == 100
    with pytest.raises(LedgerRefused, match='invalid_budget'):
        Ledger(tmp_path / 'over', budget=1, hard_limit=MAX_CAMPAIGN_LIMIT + 1)
