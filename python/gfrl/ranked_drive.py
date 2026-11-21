"""Resumable driver for the capped official ranked (真剣タイマン) campaign.

It runs ``scripts/ranked_bot.py`` (the unchanged battle routine) ONE GAME PER CHILD PROCESS, never
two at once, on the one fixed ledger ``runs/ranked100``:

* stage 1: the already approved 10 counted games, child ``--budget 12`` (2 never-applied
  preliminary numbers + 10);
* gate: when 10 games count, the ledger is reviewed (both seats' display ratings before and after,
  win/loss against the rating change, no mismatch, one model/conditions). Any anomaly holds the
  campaign there and prints the reasons; nothing past 10 is started;
* stage 2: the same ledger with ``--budget 102`` until 100 games count (applied or undecided).

All state lives in the ledger, so a restart simply continues. Only numbers and fixed codes are
printed (``emit`` rejects anything else); the child's stdout is parsed for its reason code and
never echoed, its stderr is discarded. A reservation is never retried: only failures that left
the ledger untouched (nothing reserved) and carry a known transient code are retried, at most
three times with back-off. An unknown result after a reservation stops the campaign explicitly.
Everything is read through the read-only report functions; the driver never writes the ledger.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .ranked_campaign import (DEFAULT_SUPPLEMENT, DEFAULT_SUPPLEMENT_GAME, ROOT, build_report, read_events,
                              read_supplement)
from .ranked_ledger import HARD_LIMIT, MAX_CAMPAIGN_LIMIT, fold_games, unapplied_preliminary

STATE_DIR = ROOT / 'runs/ranked100'
APPROVED = 10                                # games the user approved before the review gate
EXEMPT_GAMES = frozenset((1, 2))             # the two verified never-applied preliminary numbers
STAGE1_BUDGET = APPROVED + len(EXEMPT_GAMES)  # 12
FULL_BUDGET = HARD_LIMIT + len(EXEMPT_GAMES)  # 102
MIN_GAP = 20                                 # seconds after a game before the next entry (required)
DEFAULT_GAP = 60                             # 100-game run: 25s hit visible_state_timeout 100% of the time
BACKOFF = (30, 60, 120)                      # seconds before retry 1, 2, 3 (so at most 3 retries)
CHILD_TIMEOUT = 7200
# Failures that happen BEFORE a game number is reserved (ledger untouched) and that the site's
# slowness can explain. Anything else (bot_prohibition_notice, ranked_game_in_progress_on_start, ...)
# stops at once.
TRANSIENT = frozenset(('visible_state_timeout', 'login_not_accepted', 'ranked_lobby_missing',
                       'ranked_button_not_unique'))
WATCHED_SCRIPTS = ('ranked_bot.py', 'ranked_loop.sh', 'ranked_drive.py')
RUNNERS = ('sh', 'bash', 'zsh', 'env', 'nohup', 'caffeinate', 'time', 'arch')
SNAKE = re.compile(r'[a-z][a-z0-9_]{0,63}')
LABEL = re.compile(r'[A-Za-z0-9_.-]{1,32}')
SAFE_OUT = re.compile(r'[a-z0-9][a-z0-9_.:-]{0,79}')
SAFE_KEY = re.compile(r'[a-z][a-z0-9_]{0,39}')
RATING_ITEMS = ('self_before', 'self_after', 'opp_before', 'opp_after')

# Flags forwarded to ranked_bot.py (allowlist). The driver owns budget, state dir, checkpoint,
# dream, playwright, resume/unresolved/birth handling and always adds --reenter.
FORWARD_VALUE = frozenset(('--name', '--device', '--threads', '--match-wait', '--turn-wait', '--rating-wait',
                           '--rating-screen-wait', '--timeout', '--settle-timeout', '--chance-delay', '--step'))
FORWARD_FLAG = frozenset(('--strict-events', '--watch-screen', '--trace'))


class Finish(Exception):
    """Ends the run with a status, a fixed code and numeric/enum details; ``rc`` is the exit code."""
    RC = dict(done=0, max_games=0, stop_requested=0, stopped=1, interrupted=1, refused=2, gate_hold=3)

    def __init__(self, status, code, **extra):
        super().__init__(code)
        self.status, self.code, self.extra = status, code, extra


def refuse(code, **extra):
    return Finish('refused', code, **extra)


def stop(code, **extra):
    return Finish('stopped', code, **extra)


# ---- output -----------------------------------------------------------------------------------

def clean(value):
    """Numbers, booleans, None and fixed lowercase codes only (lists/dicts of those)."""
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str and SAFE_OUT.fullmatch(value):
        return value
    if type(value) in (list, tuple):
        return [clean(v) for v in value]
    if type(value) is dict:
        return {key: clean(v) for key, v in value.items() if SAFE_KEY.fullmatch(key)}
    raise ValueError('output_value_rejected')


def print_line(line):
    print(line, flush=True)


# ---- pure ledger logic (also used by the tests) -------------------------------------------------

def settled(row):
    return row['phase'] == 'finished' and row.get('terminal') == 'completed' \
        and row.get('outcome') in ('win', 'loss', 'draw')


def unresolved_counted(events):
    """Counted games without a recorded result (open, stopped, unresolved, finished-unknown)."""
    exempt = unapplied_preliminary(events)
    return [row['game'] for row in fold_games(events) if row['game'] not in exempt and not settled(row)]


def reserve_conditions(events):
    """Conditions of the COUNTED games' reservations (the pinned preliminary numbers are excluded)."""
    exempt = unapplied_preliminary(events)
    return {tuple(e.get(k) for k in ('model', 'ckpt_sha8', 'policy', 'search', 'dream'))
            for e in events if e['ev'] == 'reserve' and e['game'] not in exempt}


def expected_conditions(checkpoint, dream='allow'):
    """What ranked_bot.py will record for this checkpoint (model label, sha8, greedy, no search)."""
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:8]
    label = Path(checkpoint).stem
    return (label if LABEL.fullmatch(label) else 'custom', digest, 'greedy', False, dream)


def gate_review(report):
    """Anomaly codes of the first APPROVED counted games; empty means the gate may open."""
    games = [g for g in report['games'] if g['counted']][:APPROVED]
    numbers = {g['game'] for g in games}
    found = []
    if len(games) < APPROVED:
        found.append('fewer_than_approved_games')
    for g in games:
        n = g['game']
        if g['klass'] != 'completed':
            found.append(f'g{n}_not_completed')
            continue
        outcome = g['outcome']
        if outcome not in ('win', 'loss'):
            found.append(f'g{n}_outcome_{outcome or "none"}')
        block = g.get('rating')
        if block is None:
            found.append(f'g{n}_rating_absent')
            continue
        for item in RATING_ITEMS:
            if block[item]['value'] is None:
                found.append(f'g{n}_{item}_missing')
        own, opp = block['self_delta'], block['opp_delta']
        if outcome == 'win':
            if own is not None and own <= 0:
                found.append(f'g{n}_win_but_self_delta_not_positive')
            if opp is not None and opp >= 0:
                found.append(f'g{n}_win_but_opp_delta_not_negative')
        elif outcome == 'loss':
            if own is not None and own >= 0:
                found.append(f'g{n}_loss_but_self_delta_not_negative')
            if opp is not None and opp <= 0:
                found.append(f'g{n}_loss_but_opp_delta_not_positive')
    for item in report['rating']['mismatches']:
        if item['game'] in numbers:
            found.append(f"g{item['game']}_{item['kind']}_{item['item']}")
    blocks = [g.get('rating') for g in games]
    if blocks and all(b is not None and b['self_delta'] is not None and b['self_before']['value'] is not None
                      and b['self_after']['value'] is not None for b in blocks):
        if sum(b['self_delta'] for b in blocks) != blocks[-1]['self_after']['value'] - blocks[0]['self_before']['value']:
            found.append('net_change_mismatch')
    if 'game_numbers_not_contiguous' in report['warnings']:
        found.append('game_numbers_not_contiguous')
    return sorted(set(found))


def gate_id(anomalies):
    return hashlib.sha256('\n'.join(anomalies).encode()).hexdigest()[:12]


def parse_child(text):
    """Reason code of a child that left no new ledger row: 'error:<code>' or the JSON 'reason'."""
    lines = [line for line in (text or '').strip().splitlines() if line.strip()]
    if not lines:
        return 'no_output'
    last = lines[-1].strip()
    if last.startswith('error:'):
        code = last[6:]
        return code if SNAKE.fullmatch(code) else 'other'
    try:
        reason = json.loads(last).get('reason')
    except (ValueError, AttributeError):
        return 'unparsed_output'
    return reason if type(reason) is str and SNAKE.fullmatch(reason) else 'other'


def validate_forwarded(tokens):
    """Only allowlisted ranked_bot.py flags pass; values (the name!) are never inspected or printed."""
    out, tokens, i = [], list(tokens), 0
    while i < len(tokens):
        token = tokens[i]
        flag, eq, value = token.partition('=')
        if flag in FORWARD_FLAG and not eq:
            out.append(token)
        elif flag in FORWARD_VALUE:
            if eq:
                out.append(token)
            elif i + 1 < len(tokens):
                out += [token, tokens[i + 1]]
                i += 1
            else:
                raise refuse('forwarded_arg_missing_value')
        else:
            raise refuse('forwarded_arg_not_allowed')
        i += 1
    return out


def list_processes():
    """(pid, ppid, command) of every process; fail closed if ``ps`` does not work."""
    try:
        text = subprocess.run(['ps', '-axo', 'pid=,ppid=,command='], capture_output=True, text=True,
                              timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        raise refuse('process_scan_failed') from None
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def other_ranked_processes(rows, me=None):
    """Pids of ranked scripts that are NOT this process or one of its ancestors.

    The script must be one of the first four words of the command line (interpreter, flags,
    script), so a prompt or note that merely mentions the name does not count.
    """
    me = os.getpid() if me is None else me
    parents = {pid: ppid for pid, ppid, _ in rows}
    mine, cursor = set(), me
    while cursor and cursor not in mine:
        mine.add(cursor)
        cursor = parents.get(cursor)
    return [pid for pid, _, command in rows if pid not in mine and is_ranked_command(command)]


def is_ranked_command(command):
    """An interpreter/wrapper (or the script itself) running one of the watched scripts."""
    words = command.split()[:4]
    if not words:
        return False
    head = os.path.basename(words[0])
    if not (head.lower().startswith('python') or head in RUNNERS or head in WATCHED_SCRIPTS):
        return False  # e.g. an editor or a pager that merely has the file open
    return any(os.path.basename(word) in WATCHED_SCRIPTS for word in words)


def ledger_lock_busy(state_dir):
    """True while a ranked_bot process holds ranked.lock (probe only; never creates the file)."""
    try:
        fd = os.open(Path(state_dir) / 'ranked.lock', os.O_RDWR | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)
    return False


# ---- the driver --------------------------------------------------------------------------------

class Driver:
    def __init__(self, *, playwright, checkpoint, state_dir=STATE_DIR, forwarded=(), gap=DEFAULT_GAP,
                 max_games=None, accept_unresolved_through=0, gate_ack=None, child_timeout=CHILD_TIMEOUT,
                 supplement=DEFAULT_SUPPLEMENT, supplement_game=DEFAULT_SUPPLEMENT_GAME, launcher=None,
                 process_lister=list_processes, clock=time.time, sleep=time.sleep, emit=print_line,
                 backoff=BACKOFF, target=HARD_LIMIT, fresh=False):
        self.playwright, self.checkpoint = playwright, checkpoint
        self.state_dir = Path(state_dir).absolute()
        self.ledger_path = self.state_dir / 'ledger.jsonl'
        self.stop_file = self.state_dir.with_name(self.state_dir.name + '.stop')
        self.driver_lock = self.state_dir.with_name(self.state_dir.name + '.driver.lock')
        self.forwarded = list(forwarded)
        self.gap, self.max_games = gap, max_games
        self.accept_through, self.gate_ack = accept_unresolved_through, gate_ack
        self.child_timeout = child_timeout
        self.fresh = bool(fresh)
        self.target = target
        self.exempt_games = frozenset() if self.fresh else EXEMPT_GAMES
        self.stage1_budget = min(APPROVED, target) + len(self.exempt_games)
        self.full_budget = target + len(self.exempt_games)
        self.supplement_path = None if self.fresh else supplement
        self.supplement_game = supplement_game
        self.launcher = launcher or self.subprocess_launcher
        self.process_lister, self.clock, self.sleep, self.emit_line = process_lister, clock, sleep, emit
        self.backoff = tuple(backoff)
        self.played = 0
        self.interrupted = False
        self.child = None
        self.last_exit = 0
        self.gate_reported = False

    # ---- output / signals ----
    def emit(self, **fields):
        try:
            self.emit_line(json.dumps(clean(fields), sort_keys=True, allow_nan=False))
        except (ValueError, TypeError):
            self.emit_line('{"code":"output_rejected","ev":"error"}')

    def install_signals(self):
        def handler(signum, frame):
            self.interrupted = True
            child = self.child
            if child is not None and child.poll() is None:
                child.terminate()  # ranked_bot turns SIGTERM into a recorded, conservative stop
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    # ---- reading ----
    def read(self):
        if self.fresh and not self.ledger_path.exists():
            events = []
            diagnostics = dict(bytes_read=0, lines=0, sha256=hashlib.sha256(b'').hexdigest(),
                               ignored_tail_bytes=0, invalid_lines=[])
        else:
            try:
                events, diagnostics = read_events(self.ledger_path, max_game=self.full_budget)
            except OSError:
                raise refuse('ledger_missing') from None
        if diagnostics['invalid_lines']:
            raise stop('ledger_invalid_lines')
        if diagnostics['ignored_tail_bytes']:
            raise stop('ledger_unterminated_tail')
        if not self.fresh and unapplied_preliminary(events) != EXEMPT_GAMES:
            raise refuse('ledger_not_the_campaign_ledger')
        supplement = None
        if self.supplement_path and Path(self.supplement_path).is_file():
            supplement = read_supplement(self.supplement_path, self.supplement_game)
        return events, build_report(events, diagnostics, supplement,
                                    hard_limit=self.target, expect_preliminary=not self.fresh)

    def counted(self, events):
        return len(fold_games(events)) - len(self.exempt_games)

    def check_conditions(self, events):
        try:
            expected = expected_conditions(self.checkpoint)
        except OSError:
            raise refuse('checkpoint_missing') from None
        if not reserve_conditions(events) <= {expected}:
            raise refuse('conditions_mismatch')

    # ---- safety checks ----
    def check_idle(self):
        if other_ranked_processes(self.process_lister()):
            raise refuse('ranked_process_running')
        if ledger_lock_busy(self.state_dir):
            raise refuse('ledger_busy')

    def check_requests(self):
        if self.interrupted:
            raise Finish('interrupted', 'interrupted')
        if self.stop_file.exists():
            raise Finish('stop_requested', 'stop_file_present')

    def wait_until(self, ready_at):
        while True:
            remaining = ready_at - self.clock()
            if remaining <= 0:
                return
            self.check_requests()
            self.sleep(min(remaining, 1.0))

    # ---- one child ----
    def argv(self, budget, accept_unresolved):
        argv = [sys.executable, str(ROOT / 'scripts/ranked_bot.py'), '--budget', str(budget),
                '--playwright', str(self.playwright), '--checkpoint', str(self.checkpoint),
                '--state-dir', str(self.state_dir), '--reenter']
        if self.target != HARD_LIMIT:
            argv += ['--hard-limit', str(self.target)]
        if accept_unresolved:
            argv.append('--accept-unresolved')
        return argv + self.forwarded

    def subprocess_launcher(self, argv, timeout):
        """Run the child; returns (rc, stdout text, timed_out). stderr is discarded, never read."""
        self.child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL)
        timed_out = False
        try:
            try:
                out, _ = self.child.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self.child.terminate()
                try:
                    out, _ = self.child.communicate(timeout=120)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    out, _ = self.child.communicate()
            return self.child.returncode, (out or b'')[-65536:].decode('utf-8', 'replace'), timed_out
        finally:
            self.child = None

    # ---- the campaign ----
    def run(self):
        try:
            return self.campaign()
        except Finish as end:
            return self.finish(end)

    def finish(self, end):
        try:
            events, report = self.read()
            counts = report['counts']
            numbers = dict(counted=counts['counted'], wins=counts['wins'], losses=counts['losses'],
                           draws=counts['draws'], remaining=report['cap']['remaining'])
        except Finish:
            numbers = {}
        self.emit(ev='end', status=end.status, code=end.code, played=self.played, **numbers, **end.extra)
        return Finish.RC[end.status]

    def campaign(self):
        self.check_requests()
        lock_fd = None
        try:
            if self.driver_lock.parent.is_dir():
                lock_fd = os.open(self.driver_lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    raise refuse('driver_busy') from None
            return self.loop()
        finally:
            if lock_fd is not None:
                os.close(lock_fd)

    def loop(self):
        first = True
        while True:
            events, report = self.read()
            self.check_conditions(events)
            self.check_idle()
            counted = self.counted(events)
            unresolved = unresolved_counted(events)
            if first:
                first = False
                self.emit(ev='start', counted=counted, remaining=max(0, self.target - counted),
                          unresolved=unresolved, accept_unresolved_through=self.accept_through,
                          code_revs=len({e.get('code_rev') for e in events if e['ev'] == 'reserve'}),
                          gap_s=self.gap, max_games=self.max_games)
            if any(game > self.accept_through for game in unresolved):
                raise stop('unresolved_game', games=unresolved)
            if counted >= self.target:
                raise Finish('done', 'target_reached')
            if counted >= APPROVED:
                self.open_gate(report)
            if self.max_games is not None and self.played >= self.max_games:
                raise Finish('max_games', 'max_games_reached')
            self.check_requests()
            last_event = max((e['t'] for e in events), default=0)
            self.wait_until(max(last_event, self.last_exit) + self.gap)
            self.check_requests()
            self.play_one(events, counted)

    def open_gate(self, report):
        anomalies = gate_review(report)
        identity = gate_id(anomalies)
        acked = bool(anomalies) and self.gate_ack == identity
        if not self.gate_reported:
            self.gate_reported = True
            self.emit(ev='gate', passed=not anomalies, acknowledged=acked, gate_id=identity,
                      anomalies=anomalies, games=APPROVED)
        if anomalies and not acked:
            raise Finish('gate_hold', 'gate_anomalies', gate_id=identity, anomalies=anomalies)

    def play_one(self, events, counted):
        budget = self.stage1_budget if counted < APPROVED else self.full_budget
        # ranked_bot refuses to reserve after a game without a recorded result unless told to; the
        # unresolved counted games were acknowledged by the checks in ``loop`` (never silently).
        prior_rows = fold_games(events)
        accept = bool(prior_rows) and prior_rows[-1].get('terminal') != 'completed'
        retries = 0
        while True:
            self.emit(ev='launch', game=len(fold_games(events)) + 1, budget=budget, attempt=retries)
            rc, text, timed_out = self.launcher(self.argv(budget, accept), self.child_timeout)
            self.last_exit = self.clock()
            after, report = self.read()
            if len(after) == len(events):  # the child left the ledger untouched: nothing was reserved
                if timed_out:
                    raise stop('child_timeout')
                code = parse_child(text)
                if code in TRANSIENT and retries < len(self.backoff) and not self.interrupted:
                    wait = self.backoff[retries]
                    retries += 1
                    self.emit(ev='retry', code=code, attempt=retries, wait_s=wait)
                    self.wait_until(self.clock() + wait)
                    self.check_requests()
                    events, report = self.read()  # re-check everything before the next try
                    self.check_idle()
                    continue
                if code in TRANSIENT:
                    raise stop('transient_retries_exhausted', last_code=code, retries=retries)
                if self.interrupted:
                    raise Finish('interrupted', 'interrupted')
                raise stop('child_' + code if SNAKE.fullmatch('child_' + code) else 'child_other', rc=rc)
            break
        rows = fold_games(after)
        if len(rows) != len(fold_games(events)) + 1 or len(after) < len(events):
            raise stop('ledger_changed_unexpectedly')
        row = rows[-1]
        if not settled(row):
            reason = row.get('stop_reason')
            raise stop('unresolved_after_reserve', game=row['game'], started=row.get('started_at') is not None,
                       phase=row['phase'], stop_reason=reason if type(reason) is str and SNAKE.fullmatch(reason)
                       else 'none', decisions=row.get('decisions') or 0, rc=rc, interrupted=self.interrupted)
        self.played += 1
        self.report_game(after, report, row)

    def report_game(self, events, report, row):
        entry = next(g for g in report['games'] if g['game'] == row['game'])
        block = entry.get('rating') or {}
        anomalies = sorted({f"{m['kind']}_{m['item']}" for m in report['rating']['mismatches']
                            if m['game'] == row['game']})
        value = lambda item: block[item]['value'] if item in block else None
        counts = report['counts']
        self.emit(ev='game', game=row['game'], counted=counts['counted'], outcome=row.get('outcome'),
                  opp_kind=row.get('opp_kind'), decisions=row.get('decisions'), updates=row.get('updates'),
                  duration_s=row.get('duration_s'), self_before=value('self_before'),
                  self_after=value('self_after'), opp_before=value('opp_before'), opp_after=value('opp_after'),
                  self_delta=block.get('self_delta'), opp_delta=block.get('opp_delta'), wins=counts['wins'],
                  losses=counts['losses'], draws=counts['draws'], anomalies=anomalies)
        if any(a.startswith('outcome_vs_delta') for a in anomalies):
            raise stop('outcome_contradicts_rating', game=row['game'])


# ---- CLI ---------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description='Resumable one-game-at-a-time driver of the capped ranked campaign (see scripts/ranked_drive.md). '
                    'Other ranked_bot.py flags (--name --device --threads --step ...) are forwarded.')
    parser.add_argument('--playwright', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/play/m-xit1.pt')
    parser.add_argument('--state-dir', type=Path, default=None,
                        help='Campaign directory (default runs/ranked100; non-default starts/continues a fresh campaign)')
    parser.add_argument('--target', type=int, default=HARD_LIMIT,
                        help=f'Counted-game target for this campaign (1..{MAX_CAMPAIGN_LIMIT}, default {HARD_LIMIT})')
    parser.add_argument('--fresh', action='store_true',
                        help='Treat ledger as a fresh campaign without preliminary-game exemptions')
    parser.add_argument('--gap', type=int, default=DEFAULT_GAP,
                        help=f'Seconds between the end of a game and the next entry (>= {MIN_GAP})')
    parser.add_argument('--max-games', type=int, default=None, help='Play at most this many games in this run')
    parser.add_argument('--accept-unresolved-through', type=int, default=0, metavar='N',
                        help='Acknowledge counted games without a result up to game number N (they stay counted)')
    parser.add_argument('--gate-ack', default=None, metavar='GATE_ID',
                        help='Continue past a held 10-game gate whose printed gate_id is exactly this')
    parser.add_argument('--child-timeout', type=int, default=CHILD_TIMEOUT)
    return parser


def main(argv=None, *, state_dir=STATE_DIR, **injected):
    args, rest = build_parser().parse_known_args(argv)
    try:
        if args.gap < MIN_GAP or (args.max_games is not None and args.max_games < 1) \
                or args.accept_unresolved_through < 0 or args.child_timeout < 1 \
                or not 1 <= args.target <= MAX_CAMPAIGN_LIMIT:
            raise refuse('invalid_arguments')
        forwarded = validate_forwarded(rest)
    except Finish as end:
        print_line(json.dumps(dict(ev='end', status=end.status, code=end.code), sort_keys=True))
        return Finish.RC[end.status]
    chosen_dir = args.state_dir if args.state_dir is not None else state_dir
    fresh = args.fresh or (args.state_dir is not None and args.state_dir.absolute() != STATE_DIR.absolute())
    driver = Driver(playwright=args.playwright, checkpoint=args.checkpoint, state_dir=chosen_dir,
                    forwarded=forwarded, gap=args.gap, max_games=args.max_games,
                    accept_unresolved_through=args.accept_unresolved_through, gate_ack=args.gate_ack,
                    child_timeout=args.child_timeout, target=args.target, fresh=fresh, **injected)
    if not injected:
        driver.install_signals()
    return driver.run()


if __name__ == '__main__':
    raise SystemExit(main())
