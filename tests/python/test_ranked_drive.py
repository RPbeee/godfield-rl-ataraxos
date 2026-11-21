"""Local synthetic tests for the resumable ranked campaign driver (no browser, no network, no site).

The "child" is a fake that writes the REAL ledger (same Ledger class, same budget and cap rules as
ranked_bot.py) in a temporary directory, so budgets, the preliminary exemption and the unresolved
rule are enforced by the production code, not by the test. Time is a fake clock.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from gfrl.ranked_drive import (APPROVED, FULL_BUDGET, MIN_GAP, STAGE1_BUDGET, Driver, clean, gate_id, gate_review,
                               ledger_lock_busy, main, other_ranked_processes, parse_child, validate_forwarded,
                               Finish)
from gfrl.ranked_campaign import build_report, read_events
from gfrl.ranked_ledger import HARD_LIMIT, PRELIMINARY_UNAPPLIED, Ledger, LedgerRefused

START = 1795180000


class Clock:
    def __init__(self, t=START):
        self.t, self.sleeps = t, []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        self.sleeps.append(seconds)


def finish_fields(clock, before, after, opp_before, opp_after, outcome, **patch):
    fields = dict(
        terminal='completed', outcome=outcome, first_mover='self', decisions=20, updates=40, gf_final=30,
        ended_at=int(clock()), duration_s=100, rating_display_basis='offset_verified', rating_gain_cross='unknown',
        rating_opp_after=opp_after - 1500, rating_opp_before=opp_before - 1500, rating_opp_first=opp_before - 1500,
        rating_opp_gain=opp_after - opp_before, rating_opp_last=opp_before - 1500,
        rating_opp_display_after=opp_after, rating_opp_display_after_src='wire',
        rating_opp_display_before=opp_before, rating_opp_display_before_src='wire',
        rating_origin='derived_from_gain', rating_screen=before, rating_screen_check='unavailable',
        rating_self_after=after - 1500, rating_self_before=before - 1500, rating_self_first=before - 1500,
        rating_self_gain=after - before, rating_self_last=before - 1500,
        rating_self_display_after=after, rating_self_display_after_src='wire',
        rating_self_display_before=before, rating_self_display_before_src='wire')
    fields.update(patch)
    return fields


class World:
    """A temporary campaign ledger (games 1-2 are the pinned preliminary ones) plus a fake child."""

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.dir = tmp_path / 'runs' / 'ranked100'
        self.dir.mkdir(parents=True)
        self.ckpt = tmp_path / 'm-xit1.pt'
        self.ckpt.write_bytes(b'synthetic checkpoint')
        self.sha8 = hashlib.sha256(b'synthetic checkpoint').hexdigest()[:8]
        self.cond = dict(model='m-xit1', ckpt_sha8=self.sha8, code_rev='39bfa212', policy='greedy', search=False,
                         dream='allow')
        events = [dict(e) for g in sorted(PRELIMINARY_UNAPPLIED) for e in PRELIMINARY_UNAPPLIED[g]]
        self.write_events(events)
        self.clock = Clock()
        self.me, self.opp = 1500, 1500
        self.launches, self.script, self.lines, self.counter = [], [], [], 0
        self.processes = []

    @property
    def ledger_path(self):
        return self.dir / 'ledger.jsonl'

    def write_events(self, events):
        self.ledger_path.write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in events))

    # ---- what a child would write ----
    def play_game(self, budget, win=True, accept=False, delta=None, **patch):
        delta = (16 if win else -16) if delta is None else delta
        ledger = Ledger(self.dir, budget, clock=self.clock)
        try:
            game = ledger.reserve(self.cond, accept_unresolved=accept)
            ledger.record(game, 'queued')
            ledger.record(game, 'queued', rating_screen=self.me)
            ledger.record(game, 'started', end_gf=150, opp_kind='human', rating_screen=self.me, seat_id=1,
                          started_at=int(self.clock()))
            self.clock.t += 100
            ledger.record(game, 'finish', **finish_fields(
                self.clock, self.me, self.me + delta, self.opp, self.opp - delta, 'win' if win else 'loss', **patch))
            self.me += delta
            self.opp -= delta
            return game
        finally:
            ledger.close()

    def fail_game(self, budget, reason='acceptance_timeout', started=True, accept=False):
        ledger = Ledger(self.dir, budget, clock=self.clock)
        try:
            game = ledger.reserve(self.cond, accept_unresolved=accept)
            ledger.record(game, 'queued')
            if started:
                ledger.record(game, 'started', end_gf=150, opp_kind='human', seat_id=1, started_at=int(self.clock()))
            self.clock.t += 50
            ledger.record(game, 'stop', terminal='stopped', outcome='unknown', stop_reason=reason,
                          decisions=3 if started else 0, updates=4 if started else 0, duration_s=50,
                          ended_at=int(self.clock()), first_mover=None, gf_final=None)
            return game
        finally:
            ledger.close()

    def seed(self, n):
        for _ in range(n):
            counted = self.counted()
            # The two preliminary games end 'stopped', so the first real game after them is reserved the
            # way the driver tells the child to: with --accept-unresolved (the production Ledger rule).
            self.play_game(STAGE1_BUDGET if counted < APPROVED else FULL_BUDGET, win=self.counter % 3 != 0,
                           accept=counted == 0)
            self.counter += 1

    def counted(self):
        events, _ = read_events(self.ledger_path)
        return len({e['game'] for e in events}) - 2

    # ---- fake launcher ----
    def launcher(self, argv, timeout):
        self.launches.append(dict(argv=argv, t=int(self.clock()), timeout=timeout))
        step = self.script.pop(0) if self.script else None
        kind, arg = (step if isinstance(step, tuple) else (step or 'auto', None))
        budget = int(argv[argv.index('--budget') + 1])
        accept = '--accept-unresolved' in argv
        if kind == 'pre':      # failed before reserving: ledger untouched
            return 1, json.dumps(dict(status='stopped', reason=arg, detail={})), False
        if kind == 'raw':
            return 1, arg, False
        if kind == 'error':
            return 2, 'error:' + arg, False
        try:
            if kind == 'fail':
                self.fail_game(budget, arg or 'acceptance_timeout', accept=accept)
                return 1, json.dumps(dict(status='stopped', reason=arg)), False
            if kind == 'unmatched':
                self.fail_game(budget, 'match_wait_timeout', started=False, accept=accept)
                return 1, json.dumps(dict(status='stopped', reason='match_wait_timeout')), False
            if kind == 'patch':
                self.play_game(budget, accept=accept, **arg)
            else:
                self.play_game(budget, win=self.counter % 3 != 0, accept=accept)
                self.counter += 1
        except LedgerRefused as error:
            return 2, 'error:' + error.code, False
        if kind == 'win_then_stop':
            self.driver_obj.stop_file.write_text('1\n')
        if kind == 'win_then_interrupt':
            self.driver_obj.interrupted = True
        return 0, json.dumps(dict(status='finished')), False

    def driver(self, **kw):
        options = dict(playwright=Path('/nonexistent/playwright-core'), checkpoint=self.ckpt, state_dir=self.dir,
                       launcher=self.launcher, process_lister=lambda: self.processes, clock=self.clock,
                       sleep=self.clock.sleep, emit=self.lines.append, supplement=None)
        options.update(kw)
        self.driver_obj = Driver(**options)
        return self.driver_obj

    def run(self, **kw):
        rc = self.driver(**kw).run()
        return rc, [json.loads(line) for line in self.lines]

    def end(self):
        return json.loads(self.lines[-1])


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def budgets(world):
    return [int(l['argv'][l['argv'].index('--budget') + 1]) for l in world.launches]


# ---- the whole campaign ------------------------------------------------------------------------

def test_full_campaign_budget_12_then_gate_then_102_with_gaps(world):
    rc, lines = world.run()
    end = lines[-1]
    assert rc == 0 and end['status'] == 'done' and end['code'] == 'target_reached' and end['counted'] == HARD_LIMIT
    assert len(world.launches) == HARD_LIMIT and end['played'] == HARD_LIMIT
    assert budgets(world) == [STAGE1_BUDGET] * APPROVED + [FULL_BUDGET] * (HARD_LIMIT - APPROVED)
    assert all('--reenter' in l['argv'] and '--resume' not in l['argv'] and '--allow-birth' not in l['argv']
               for l in world.launches)
    # Only the very first entry follows the two 'stopped' preliminary slots, so only it needs the flag.
    assert '--accept-unresolved' in world.launches[0]['argv']
    assert all('--accept-unresolved' not in l['argv'] for l in world.launches[1:])
    gates = [l for l in lines if l['ev'] == 'gate']
    assert len(gates) == 1 and gates[0]['passed'] is True and gates[0]['anomalies'] == []
    assert [l['ev'] for l in lines].count('game') == HARD_LIMIT and lines[0]['ev'] == 'start'
    # Every entry is at least the gap after the previous game's last ledger event.
    events, _ = read_events(world.ledger_path)
    last_t = None
    for event in events:
        if event['ev'] == 'reserve' and event['game'] > 3 and last_t is not None:
            assert event['t'] - last_t >= 25
        last_t = event['t']
    # The real ledger: never more than 100 counted, 102 numbers, preliminary ones still exempt.
    assert len({e['game'] for e in events}) == HARD_LIMIT + 2


def test_resumes_from_the_ledger_alone(world):
    world.seed(7)
    rc, lines = world.run(max_games=2)
    assert rc == 0 and lines[-1]['status'] == 'max_games' and lines[-1]['counted'] == 9
    assert lines[0]['counted'] == 7
    world.lines.clear()
    rc, lines = world.run()                       # a fresh driver, same ledger
    assert rc == 0 and lines[-1]['counted'] == HARD_LIMIT and budgets(world)[2:4] == [STAGE1_BUDGET, FULL_BUDGET]


def test_first_entry_waits_for_the_gap_after_the_last_ledger_event(world):
    world.seed(2)
    events, _ = read_events(world.ledger_path)
    last = max(e['t'] for e in events)
    world.clock.t = last + 5
    rc, _ = world.run(max_games=1, gap=30)
    assert rc == 0 and world.launches[0]['t'] == last + 30


# ---- the 10-game gate --------------------------------------------------------------------------

def test_gate_holds_at_ten_and_reports_the_anomaly(world):
    world.seed(7)
    world.script = ['auto', ('patch', dict(rating_opp_display_after=None, rating_opp_after=None)), 'auto', 'auto']
    rc, lines = world.run()
    end = lines[-1]
    assert rc == 3 and end['status'] == 'gate_hold' and end['counted'] == APPROVED
    assert len(world.launches) == 3                   # nothing past 10 is started
    assert 'g11_opp_after_missing' in end['anomalies'] and end['gate_id'] == gate_id(end['anomalies'])
    # Same ledger, no acknowledgement: still held, nothing launched.
    world.lines.clear()
    rc, lines = world.run(gate_ack='0' * 12)
    assert rc == 3 and len(world.launches) == 3
    # The exact gate id lets it proceed, now with the full budget.
    world.lines.clear()
    rc, lines = world.run(gate_ack=end['gate_id'], max_games=1)
    assert rc == 0 and lines[-1]['status'] == 'max_games' and budgets(world)[-1] == FULL_BUDGET
    assert any(l['ev'] == 'gate' and l['acknowledged'] is True for l in lines)


def test_gate_catches_a_win_that_lowered_the_rating(world):
    world.seed(3)
    world.play_game(STAGE1_BUDGET, win=True, delta=-16)       # game 6: "win", rating down
    world.seed(6)
    events, _ = read_events(world.ledger_path)
    found = gate_review(build_report(events))
    assert 'g6_win_but_self_delta_not_positive' in found and 'g6_outcome_vs_delta_self_delta' in found
    rc, lines = world.run()
    assert rc == 3 and lines[-1]['status'] == 'gate_hold' and world.launches == []


def test_light_check_stops_on_an_outcome_that_contradicts_the_rating(world):
    world.seed(2)
    world.script = [('patch', dict(win=False, delta=16))]
    rc, lines = world.run()
    assert rc == 1 and lines[-1]['code'] == 'outcome_contradicts_rating' and len(world.launches) == 1
    assert next(l for l in lines if l['ev'] == 'game')['anomalies'] == ['outcome_vs_delta_self_delta']


def test_gate_replays_the_real_ledger_with_legacy_correction(world):
    cond = world.cond
    legacy = [
        dict(v=1, ev='reserve', game=3, t=1795174045, budget=3, **cond),
        dict(v=1, ev='queued', game=3, t=1795174045),
        dict(v=1, ev='started', game=3, t=1795174045, end_gf=150, opp_kind='human', seat_id=1, started_at=1795174045),
        dict(v=1, ev='finish', game=3, t=1795174160, decisions=18, duration_s=115, ended_at=1795174160,
             first_mover='self', gf_final=23, outcome='win', rating_opp_after=16, rating_opp_before=32,
             rating_opp_first=32, rating_opp_gain=-16, rating_opp_last=32, rating_origin='derived_from_gain',
             rating_self_after=None, rating_self_before=None, rating_self_first=None, rating_self_gain=None,
             rating_self_last=None, terminal='completed', updates=37),
        dict(v=1, ev='reserve', game=4, t=1795174363, budget=10, **cond),
        dict(v=1, ev='queued', game=4, t=1795174363),
        dict(v=1, ev='started', game=4, t=1795174364, end_gf=150, opp_kind='human', seat_id=1, started_at=1795174364),
        dict(v=1, ev='finish', game=4, t=1795174448, decisions=14, duration_s=84, ended_at=1795174448,
             first_mover='self', gf_final=16, outcome='loss', rating_opp_after=154, rating_opp_before=139,
             rating_opp_first=139, rating_opp_gain=15, rating_opp_last=139, rating_origin='derived_from_gain',
             rating_self_after=1, rating_self_before=16, rating_self_first=16, rating_self_gain=-15,
             rating_self_last=16, terminal='completed', updates=29),
    ]
    prelim = [dict(e) for g in sorted(PRELIMINARY_UNAPPLIED) for e in PRELIMINARY_UNAPPLIED[g]]
    world.write_events(prelim + legacy)
    world.clock.t = 1795200000
    world.me, world.opp = 1501, 1639 + 15
    # Games 5 and 6 in the new format, then four more, all continuous on the pre-match screen.
    world.play_game(STAGE1_BUDGET, win=False, delta=-17)
    world.play_game(STAGE1_BUDGET, win=True)
    world.me, world.opp = 1500, 1454
    world.seed(6)
    supplement = world.root / 'sup.json'
    supplement.write_text(json.dumps(dict(correct_values=dict(
        self=dict(start_display=1500, end_display=1516), opponent=dict(start_display=1532, end_display=1516)))))
    from gfrl.ranked_campaign import read_supplement
    events, diag = read_events(world.ledger_path)
    with_supplement = gate_review(build_report(events, diag, read_supplement(supplement, 3)))
    without = gate_review(build_report(events, diag, None))
    assert world.counted() == APPROVED
    assert with_supplement == []
    assert 'g3_self_before_missing' in without
    # And through the driver: the legacy games 3-4 plus the supplement let the gate open.
    rc, lines = world.run(max_games=1, supplement=supplement)
    assert rc == 0 and lines[-1]['status'] == 'max_games' and budgets(world) == [FULL_BUDGET]
    world.lines.clear()
    world.launches.clear()
    rc, lines = world.run(max_games=1)               # no supplement: game 3's start is unknown -> held
    assert rc == 3 and 'g3_self_before_missing' in lines[-1]['anomalies']


# ---- retries before a reservation, never after ---------------------------------------------------

def test_transient_failures_before_reserve_are_retried_with_backoff(world):
    world.seed(2)
    world.script = [('pre', 'visible_state_timeout'), ('pre', 'login_not_accepted'), 'auto']
    rc, lines = world.run(max_games=1)
    assert rc == 0 and lines[-1]['status'] == 'max_games' and len(world.launches) == 3
    retries = [l for l in lines if l['ev'] == 'retry']
    assert [(r['code'], r['attempt'], r['wait_s']) for r in retries] == \
        [('visible_state_timeout', 1, 30), ('login_not_accepted', 2, 60)]
    assert world.launches[1]['t'] - world.launches[0]['t'] >= 30
    assert world.launches[2]['t'] - world.launches[1]['t'] >= 60
    assert world.counted() == 3                      # exactly one game was reserved


def test_retries_stop_after_three(world):
    world.seed(2)
    world.script = [('pre', 'visible_state_timeout')] * 4 + ['auto']
    rc, lines = world.run()
    end = lines[-1]
    assert rc == 1 and end['code'] == 'transient_retries_exhausted' and end['last_code'] == 'visible_state_timeout'
    assert len(world.launches) == 4 and end['retries'] == 3 and world.counted() == 2


@pytest.mark.parametrize('step,code', [(('pre', 'bot_prohibition_notice'), 'child_bot_prohibition_notice'),
                                       (('pre', 'ranked_game_in_progress_on_start'),
                                        'child_ranked_game_in_progress_on_start'),
                                       (('error', 'budget_exhausted'), 'child_budget_exhausted'),
                                       (('raw', 'https://secret.example/room?key=abc'), 'child_unparsed_output'),
                                       (('raw', ''), 'child_no_output')])
def test_other_failures_before_reserve_stop_at_once(world, step, code):
    world.seed(2)
    world.script = [step]
    rc, lines = world.run()
    assert rc == 1 and lines[-1]['code'] == code and len(world.launches) == 1
    assert 'secret' not in ''.join(world.lines)


def test_unknown_after_reserve_is_reported_and_never_retried_or_silently_skipped(world):
    world.seed(2)
    world.script = [('fail', 'acceptance_timeout')]
    rc, lines = world.run()
    end = lines[-1]
    assert rc == 1 and end['code'] == 'unresolved_after_reserve' and len(world.launches) == 1
    assert (end['game'], end['started'], end['stop_reason'], end['phase']) == (5, True, 'acceptance_timeout', 'stopped')
    assert end['counted'] == 3
    # Restart without acknowledgement: refuses before any launch.
    world.lines.clear()
    rc, lines = world.run()
    assert rc == 1 and lines[-1]['code'] == 'unresolved_game' and lines[-1]['games'] == [5] and len(world.launches) == 1
    # Acknowledged through game 5: continues, tells the child, and the slot stays counted.
    world.lines.clear()
    rc, lines = world.run(accept_unresolved_through=5, max_games=2)
    assert rc == 0 and '--accept-unresolved' in world.launches[1]['argv']
    assert '--accept-unresolved' not in world.launches[2]['argv']       # only while the last game is unresolved
    assert lines[-1]['counted'] == 5 and world.counted() == 5


def test_no_opponent_timeout_after_reserve_also_stops(world):
    world.seed(2)
    world.script = [('unmatched', None)]
    rc, lines = world.run()
    end = lines[-1]
    assert rc == 1 and end['code'] == 'unresolved_after_reserve' and end['started'] is False \
        and end['stop_reason'] == 'match_wait_timeout' and len(world.launches) == 1


def test_a_stale_open_game_is_never_taken_over_without_acknowledgement(world):
    world.seed(2)
    ledger = Ledger(world.dir, STAGE1_BUDGET, clock=world.clock)
    ledger.reserve(world.cond)
    ledger.record(5, 'queued')
    ledger.close()                                # the process "died": the game stays open
    rc, lines = world.run()
    assert rc == 1 and lines[-1]['code'] == 'unresolved_game' and world.launches == []


# ---- not interfering with a running game ---------------------------------------------------------

def test_refuses_while_another_ranked_process_runs(world):
    world.seed(2)
    world.processes = [(100, 1, 'sh scripts/ranked_loop.sh 7 12 /pw --reenter')]
    rc, lines = world.run()
    assert rc == 2 and lines[-1]['code'] == 'ranked_process_running' and world.launches == []
    world.processes = [(200, 1, '/x/.venv/bin/python scripts/ranked_bot.py --budget 12')]
    world.lines.clear()
    assert world.run()[0] == 2 and world.launches == []


def test_refuses_while_the_ledger_lock_is_held(world):
    world.seed(2)
    holder = Ledger(world.dir, STAGE1_BUDGET)     # what a running ranked_bot holds
    try:
        assert ledger_lock_busy(world.dir)
        rc, lines = world.run()
        assert rc == 2 and lines[-1]['code'] == 'ledger_busy' and world.launches == []
    finally:
        holder.close()
    assert not ledger_lock_busy(world.dir)


def test_refuses_a_second_driver(world):
    world.seed(2)
    fd = os.open(world.dir.with_name('ranked100.driver.lock'), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        rc, lines = world.run()
        assert rc == 2 and lines[-1]['code'] == 'driver_busy' and world.launches == []
    finally:
        os.close(fd)


def test_process_scan_matches_only_real_ranked_command_lines():
    rows = [(100, 1, 'sh scripts/ranked_loop.sh 7 12 /pw --reenter'),
            (200, 1, '/x/.venv/bin/python scripts/ranked_bot.py --budget 12'),
            (250, 1, '/x/.venv/bin/python -u /abs/scripts/ranked_bot.py --budget 12'),
            (300, 1, 'claude -p --model m 現行ranked_loopは ranked_bot.py ranked_drive.py を呼ぶ'),
            (350, 1, 'vim scripts/ranked_bot.py'),                 # an editor, not a run
            (360, 1, 'caffeinate -i /x/.venv/bin/python scripts/ranked_drive.py'),
            (399, 1, 'sh -c python scripts/ranked_drive.py --playwright x'),
            (400, 399, '/x/.venv/bin/python scripts/ranked_drive.py --playwright x'),
            (500, 1, '/x/.venv/bin/python scripts/ranked_drive.py --playwright y')]
    assert other_ranked_processes(rows, me=400) == [100, 200, 250, 360, 500]


def test_refuses_another_ledger_or_checkpoint(world, tmp_path):
    world.seed(2)
    world.ckpt.write_bytes(b'a different checkpoint')
    rc, lines = world.run()
    assert rc == 2 and lines[-1]['code'] == 'conditions_mismatch' and world.launches == []
    other = World(tmp_path / 'other')
    forged = [dict(e) for g in sorted(PRELIMINARY_UNAPPLIED) for e in PRELIMINARY_UNAPPLIED[g]]
    forged[0]['t'] += 1
    other.write_events(forged)
    rc, lines = other.run()
    assert rc == 2 and lines[-1]['code'] == 'ledger_not_the_campaign_ledger' and other.launches == []
    empty = World(tmp_path / 'empty')
    empty.write_events([])
    assert empty.run()[0] == 2 and empty.end()['code'] == 'ledger_not_the_campaign_ledger'


def test_a_damaged_ledger_stops(world):
    world.seed(2)
    with open(world.ledger_path, 'ab') as handle:
        handle.write(b'{"ev": "reserve"')            # half-written tail
    rc, lines = world.run()
    assert rc == 1 and lines[-1]['code'] == 'ledger_unterminated_tail' and world.launches == []
    world.write_events([])
    with open(world.ledger_path, 'ab') as handle:
        handle.write(b'not json\n')
    world.lines.clear()
    assert world.run()[0] == 1 and world.end()['code'] == 'ledger_invalid_lines'


# ---- stopping, arguments, output ---------------------------------------------------------------

def test_stop_file_ends_the_run_after_the_current_game(world):
    world.seed(2)
    world.script = ['win_then_stop']
    rc, lines = world.run()
    assert rc == 0 and lines[-1]['status'] == 'stop_requested' and len(world.launches) == 1
    world.lines.clear()
    rc, _ = world.run()                                # still present: nothing starts
    assert rc == 0 and world.end()['status'] == 'stop_requested' and len(world.launches) == 1
    world.driver_obj.stop_file.unlink()
    assert world.run(max_games=1)[0] == 0 and len(world.launches) == 2


def test_interrupt_ends_the_run_after_the_recorded_game(world):
    world.seed(2)
    world.script = ['win_then_interrupt']
    rc, lines = world.run()
    assert rc == 1 and lines[-1]['status'] == 'interrupted' and len(world.launches) == 1


def test_forwarded_arguments_are_an_allowlist():
    assert validate_forwarded(['--name', 'x', '--step=OK', '--threads', '4', '--strict-events']) == \
        ['--name', 'x', '--step=OK', '--threads', '4', '--strict-events']
    for bad in (['--budget', '3'], ['--state-dir', 'x'], ['--resume'], ['--accept-unresolved'], ['--allow-birth'],
                ['--checkpoint', 'x'], ['--dream', 'allow'], ['--discover'], ['--status'], ['--playwright', 'x'],
                ['positional']):
        with pytest.raises(Finish) as caught:
            validate_forwarded(bad)
        assert caught.value.code == 'forwarded_arg_not_allowed'
    with pytest.raises(Finish):
        validate_forwarded(['--name'])


def test_cli_validates_arguments_and_never_prints_the_name(world, capsys):
    world.seed(2)
    common = ['--playwright', '/nonexistent', '--checkpoint', str(world.ckpt)]
    assert main(common + ['--gap', str(MIN_GAP - 1)], state_dir=world.dir) == 2
    assert json.loads(capsys.readouterr().out)['code'] == 'invalid_arguments'
    assert main(common + ['--budget', '200'], state_dir=world.dir) == 2
    assert json.loads(capsys.readouterr().out)['code'] == 'forwarded_arg_not_allowed'
    secret = 'SECRET-ACCOUNT-NAME'
    rc = main(common + ['--max-games', '1', '--name', secret], state_dir=world.dir, launcher=world.launcher,
              process_lister=lambda: [], clock=world.clock, sleep=world.clock.sleep, emit=world.lines.append,
              supplement=None)
    assert rc == 0 and secret in world.launches[0]['argv'] and secret not in ''.join(world.lines)
    assert secret not in capsys.readouterr().out


def test_output_is_only_numbers_and_fixed_codes(world):
    world.seed(2)
    world.script = [('pre', 'visible_state_timeout'), 'auto', ('raw', 'ＢＯＴ 名前 http://x/y'), ]
    world.run(max_games=2)
    world.run(max_games=1)
    for line in world.lines:
        parsed = json.loads(line)
        assert clean(parsed) == parsed
        assert 'http' not in line and '名前' not in line
    with pytest.raises(ValueError):
        clean(dict(ev='x', value='Name With Spaces'))
    with pytest.raises(ValueError):
        clean(dict(ev='x', value=1.5))


def test_parse_child_codes():
    assert parse_child('{"reason": "visible_state_timeout", "status": "stopped"}\n') == 'visible_state_timeout'
    assert parse_child('error:budget_exhausted') == 'budget_exhausted'
    assert parse_child('error:Bad Code!') == 'other'
    assert parse_child('') == 'no_output' and parse_child('garbage') == 'unparsed_output'
    assert parse_child('{"reason": "Has Spaces"}') == 'other'


# ---- the real subprocess path --------------------------------------------------------------------

def test_subprocess_launcher_returns_stdout_only_and_enforces_the_timeout(world, tmp_path):
    driver = world.driver()
    code = 'import sys; print("{\\"reason\\": \\"visible_state_timeout\\"}"); print("STDERR-LEAK", file=sys.stderr)'
    rc, text, timed_out = driver.subprocess_launcher([sys.executable, '-c', code], 30)
    assert rc == 0 and not timed_out and 'visible_state_timeout' in text and 'STDERR-LEAK' not in text
    rc, text, timed_out = driver.subprocess_launcher([sys.executable, '-c', 'import time; time.sleep(60)'], 1)
    assert timed_out and rc != 0
    # The driver never holds the ledger lock itself: a child can take it while the driver is alive.
    world.seed(2)
    probe = 'import fcntl,os,sys; fd=os.open(sys.argv[1],os.O_RDWR); fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); print("ok")'
    rc, text, _ = driver.subprocess_launcher([sys.executable, '-c', probe, str(world.dir / 'ranked.lock')], 30)
    assert text.strip() == 'ok'
