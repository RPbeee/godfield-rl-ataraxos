"""Append-only ledger for the capped official ranked (真剣タイマン) run.

Only numbers, booleans and fixed enums are ever written: no names, user IDs,
room keys, URLs, chat, raw receive data or free text. Every event goes through
``validate_event`` which rejects unknown keys and out-of-range values, so a bug
elsewhere cannot leak into the file. A game number (the research ID) is reserved
BEFORE matchmaking is applied for; a reservation is never released automatically,
so an undecided game keeps consuming its slot (conservative accounting).
Only one process may hold the ledger (flock), which also enforces concurrency 1.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import time

from .official_ui import UIUnsupported

HARD_LIMIT = 100  # Default per-campaign cap (initial 100-game run); kept for backwards compatibility.
MAX_CAMPAIGN_LIMIT = 1000  # Upper bound for explicitly approved extended campaigns (AGENTS.md §2).
SCHEMA = 1  # Unchanged: the exemption below writes nothing new, old ledgers stay valid as they are.

# Raw wire rating -> official display rating (official client: display = 1500 + raw; an absent
# key is 0). Lives here so read-only tools can use it without importing the browser/model stack.
WIRE_DISPLAY_OFFSET = 1500


def _pinned(game, budget, t_reserve, t_stop):
    """The exact ledger events of one preliminary game that never pressed 戦う (see below)."""
    return (
        dict(v=1, ev='reserve', game=game, t=t_reserve, budget=budget, model='m-xit1', ckpt_sha8='f2bd3ea3',
             code_rev='39bfa212', policy='greedy', search=False, dream='allow'),
        dict(v=1, ev='queued', game=game, t=t_reserve),
        dict(v=1, ev='stop', game=game, t=t_stop, ended_at=t_stop, terminal='stopped', outcome='unknown',
             stop_reason='match_wait_timeout', decisions=0, updates=0, duration_s=0, first_mover=None,
             gf_final=None),
    )


# The two research slots that were spent before any real match: the matchmaking entry was clicked
# but the application button 戦う was NOT pressed (user-confirmed on the real screen, 2026-11-20),
# no opponent, 0 decisions. They keep their game numbers 1 and 2 and still occupy two slots of the
# ledger, but do not count against HARD_LIMIT. The exemption is bound to these exact events
# (timestamps included): another ledger directory, another game number, a changed event or an
# additional event for the same game never qualifies, and nothing is ever written to mark it.
PRELIMINARY_UNAPPLIED = {
    1: _pinned(1, budget=1, t_reserve=1795173002, t_stop=1795173902),
    2: _pinned(2, budget=2, t_reserve=1795173913, t_stop=1795174033),
}
# Largest game number / budget a ledger can ever reach (only with both exemptions verified).
GAME_NUMBER_MAX = HARD_LIMIT + len(PRELIMINARY_UNAPPLIED)
MAX_GAME_NUMBER_CAP = MAX_CAMPAIGN_LIMIT + len(PRELIMINARY_UNAPPLIED)

OUTCOMES = ('win', 'loss', 'draw', 'unknown')
TERMINALS = ('completed', 'stopped', 'unresolved')
EVENTS = ('reserve', 'queued', 'started', 'resumed', 'finish', 'stop', 'recovered')
PHASES = dict(reserve='reserved', queued='queued', started='started', resumed='started',
              finish='finished', stop='stopped', recovered='unresolved')
OPEN_PHASES = ('reserved', 'queued', 'started')
FIRST_MOVERS = ('self', 'opponent')
OPPONENT_KINDS = ('human', 'bot', 'unknown')
RATING_ORIGINS = ('wire_consistent', 'derived_from_gain', 'wire_before_only', 'unknown')
# rating_screen_*: what the application/result SCREEN shows for our own account (the official
# display value). rating_*_first/last/gain/before/after: RAW wire values (display = 1500 + raw).
# rating_{self,opp}_display_before/after: the display scale with a *_src provenance, converted
# only when the offset is verified (basis 'offset_verified'); otherwise null and 'unconverted'.
RATING_SCREEN_CHECKS = ('consistent', 'inconsistent', 'delta_only', 'unavailable')
RATING_CROSS = ('zero_sum', 'not_zero_sum', 'unknown')
RATING_DISPLAY_BASES = ('unconverted', 'offset_verified')
# Provenance of a display rating (official client: display = 1500 + raw, an absent key = 0):
# 'wire' = integer raw + offset; 'wire_absent_zero' = absent key read as 0 (a gain only after the
# formal end); 'screen' = the official screen (our own seat only); 'unknown' = null.
RATING_DISPLAY_SOURCES = ('wire', 'wire_absent_zero', 'screen', 'unknown')
DREAM_MODES = ('allow', 'last-resort')
SNAKE = re.compile(r'[a-z][a-z0-9_]{0,63}')
HEX8 = re.compile(r'[0-9a-f]{8}')
LABEL = re.compile(r'[A-Za-z0-9_.-]{1,32}')


class LedgerRefused(UIUnsupported):
    """A budget / concurrency / safety rule refused the request (code only)."""
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _int(low, high):
    return lambda v: type(v) is int and low <= v <= high


def _opt(check):
    return lambda v: v is None or check(v)


def _enum(values):
    return lambda v: type(v) is str and v in values


_RATING = _opt(_int(-10 ** 6, 10 ** 6))
FIELDS = {
    # Common to every event.
    't': _int(0, 4102444800),
    # Conditions (reserve).
    'model': lambda v: type(v) is str and LABEL.fullmatch(v) is not None,
    'ckpt_sha8': lambda v: type(v) is str and HEX8.fullmatch(v) is not None,
    'code_rev': lambda v: type(v) is str and (HEX8.fullmatch(v) is not None or v == 'unknown'),
    'policy': _enum(('greedy',)),
    'search': lambda v: type(v) is bool,
    'dream': _enum(DREAM_MODES),
    'budget': _int(1, MAX_GAME_NUMBER_CAP),
    # Game facts.
    'seat_id': _int(0, 8),
    'first_mover': _opt(_enum(FIRST_MOVERS)),
    'opp_kind': _enum(OPPONENT_KINDS),
    'end_gf': _opt(_int(0, 10 ** 6)),
    'outcome': _enum(OUTCOMES),
    'terminal': _enum(TERMINALS),
    'stop_reason': _opt(lambda v: type(v) is str and SNAKE.fullmatch(v) is not None),
    'decisions': _int(0, 100000),
    'updates': _int(0, 10 ** 6),
    'gf_final': _opt(_int(0, 10 ** 6)),
    'started_at': _int(0, 4102444800),
    'ended_at': _int(0, 4102444800),
    'duration_s': _int(0, 10 ** 7),
    'resumes': _int(0, 1000),
    # Ratings: raw observations and the (grounded or null) before/after.
    'rating_self_first': _RATING, 'rating_self_last': _RATING, 'rating_self_gain': _RATING,
    'rating_opp_first': _RATING, 'rating_opp_last': _RATING, 'rating_opp_gain': _RATING,
    'rating_self_before': _RATING, 'rating_self_after': _RATING,
    'rating_opp_before': _RATING, 'rating_opp_after': _RATING,
    'rating_origin': _enum(RATING_ORIGINS),
    # The one number on the pre-match screen (照合材料; null when not unique).
    'rating_screen': _RATING,
    # The same screen after the game (display values, digits only): rating, signed gain, and the
    # fixed-enum cross-checks (screen delta vs screen gain; screen delta vs the opponent's raw gain).
    'rating_screen_after': _RATING, 'rating_screen_gain': _RATING,
    'rating_screen_check': _enum(RATING_SCREEN_CHECKS),
    'rating_gain_cross': _enum(RATING_CROSS),
    'rating_opp_display_before': _RATING, 'rating_opp_display_after': _RATING,
    'rating_display_basis': _enum(RATING_DISPLAY_BASES),
    'rating_self_display_before': _RATING, 'rating_self_display_after': _RATING,
    'rating_self_display_before_src': _enum(RATING_DISPLAY_SOURCES),
    'rating_self_display_after_src': _enum(RATING_DISPLAY_SOURCES),
    'rating_opp_display_before_src': _enum(RATING_DISPLAY_SOURCES),
    'rating_opp_display_after_src': _enum(RATING_DISPLAY_SOURCES),
}
EVENT_KEYS = frozenset(('v', 'ev', 'game')) | frozenset(FIELDS)


def validate_event(event, *, max_game=GAME_NUMBER_MAX):
    """Return a copy containing only allowlisted keys with valid values."""
    if not isinstance(event, dict) or set(event) - EVENT_KEYS:
        raise ValueError('ledger_unknown_key')
    if event.get('v') != SCHEMA or event.get('ev') not in EVENTS:
        raise ValueError('ledger_bad_event')
    if not _int(1, max_game)(event.get('game')):
        raise ValueError('ledger_bad_game')
    if 'budget' in event and not _int(1, max_game)(event['budget']):
        raise ValueError('ledger_bad_value')
    for key, value in event.items():
        if key in FIELDS and not FIELDS[key](value):
            raise ValueError('ledger_bad_value')  # key name intentionally omitted from messages
    return dict(event)


def unapplied_preliminary(events):
    """Game numbers whose ENTIRE event history equals the pinned never-applied preliminary game.

    ``events``: validated ledger events (``Ledger.events``). Fail-closed: any difference (a
    timestamp, an extra event such as a later ``started`` / ``recovered``, another game number)
    returns nothing for that game, which then simply counts against HARD_LIMIT.
    """
    return frozenset(game for game, pinned in PRELIMINARY_UNAPPLIED.items()
                     if [e for e in events if e.get('game') == game] == list(pinned))


def screen_check(before, after, gain):
    """Cross-check our own pre-match screen rating, post-game screen rating and shown gain."""
    if before is None or after is None:
        return 'unavailable'
    if gain is not None:
        return 'consistent' if after - before == gain else 'inconsistent'
    return 'delta_only' if after != before else 'unavailable'  # unchanged = possibly not yet updated


def gain_cross(before, after, gain, opp_wire_gain):
    """Does our screen delta equal minus the opponent's raw gain (rating points are zero-sum)?"""
    delta = gain
    if delta is None and before is not None and after is not None and after != before:
        delta = after - before
    if delta is None or opp_wire_gain is None:
        return 'unknown'
    return 'zero_sum' if delta == -opp_wire_gain else 'not_zero_sum'


def derive_ratings(first, last, gain, *, first_was_pregame):
    """Ground the before/after rating of one seat; null when it cannot be shown.

    first/last: raw ``rating`` values at the earliest/latest observation; gain: raw
    ``ratingGain`` at the latest observation. ``first_was_pregame`` is true only if
    the first observation was a live (not over) state without a gain yet.
    """
    if first is None or not first_was_pregame:
        return dict(before=None, after=None, origin='unknown')
    if last is not None and gain is not None and last != first and last - first == gain:
        return dict(before=first, after=last, origin='wire_consistent')
    if last == first and gain not in (None, 0):
        return dict(before=first, after=first + gain, origin='derived_from_gain')
    return dict(before=first, after=None, origin='wire_before_only')


def fold_games(events):
    """One merged row per game number (later events overwrite earlier keys), ordered by number."""
    rows = {}
    for event in events:
        row = rows.setdefault(event['game'], dict(game=event['game'], resumes=0))
        for key, value in event.items():
            if key not in ('v', 'ev', 't'):
                row[key] = value
        row['phase'] = PHASES[event['ev']]
        if event['ev'] == 'resumed':
            row['resumes'] = row.get('resumes', 0) + 1
    return [rows[k] for k in sorted(rows)]


class Ledger:
    def __init__(self, directory, budget=1, *, hard_limit=HARD_LIMIT, clock=time.time):
        # ``budget`` = total game NUMBERS (slots) this ledger may ever hold, as before. Above
        # ``hard_limit`` it is accepted only up to the verified preliminary exemptions (checked
        # after loading); the cap on counted games is enforced separately in ``reserve``.
        if type(hard_limit) is not int or not 1 <= hard_limit <= MAX_CAMPAIGN_LIMIT:
            raise LedgerRefused('invalid_budget')
        self.hard_limit = hard_limit
        self.max_game = hard_limit + len(PRELIMINARY_UNAPPLIED)
        if type(budget) is not int or not 1 <= budget <= self.max_game:
            raise LedgerRefused('invalid_budget')
        self.budget = budget
        self.clock = clock
        self.directory = Path(directory).absolute()
        if self.directory.is_symlink():
            raise LedgerRefused('ledger_dir_symlink')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.lock = os.open(self.directory / 'ranked.lock',
                            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock)
            raise LedgerRefused('ledger_busy') from None
        path = self.directory / 'ledger.jsonl'
        self.fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            self.events = self._load()
            if self.budget > self.hard_limit + len(self.exempt_games()):
                raise LedgerRefused('invalid_budget')
        except BaseException:
            self.close()
            raise

    def _load(self):
        events = []
        with os.fdopen(os.dup(self.fd), 'r') as source:
            source.seek(0)
            for line in source:
                if line.strip():
                    events.append(validate_event(json.loads(line), max_game=self.max_game))
        return events

    def close(self):
        for fd in (self.fd, self.lock):
            try:
                os.close(fd)  # Closing the lock descriptor releases the flock.
            except OSError:
                pass

    # ---- state ---------------------------------------------------------
    def games(self):
        return fold_games(self.events)

    def open_game(self):
        for row in self.games():
            if row['phase'] in OPEN_PHASES:
                return row
        return None

    def exempt_games(self):
        """Verified preliminary never-applied game numbers (do not count against HARD_LIMIT)."""
        return unapplied_preliminary(self.events)

    def summary(self):
        games = self.games()
        exempt = self.exempt_games()
        counted = len(games) - len(exempt)
        finished = [g for g in games if g['phase'] == 'finished']
        count = lambda outcome: sum(g.get('outcome') == outcome for g in finished)
        undecided = lambda g: g['phase'] in ('unresolved', 'stopped') and g.get('outcome') in (None, 'unknown')
        # consumed = every game number (slot) in the ledger; counted = those against hard_limit.
        return dict(budget=self.budget, hard_limit=self.hard_limit, consumed=len(games),
                    counted=counted, exempt_unapplied=sorted(exempt),
                    remaining=max(0, min(self.budget - len(games), self.hard_limit - counted)),
                    finished=len(finished), wins=count('win'), losses=count('loss'),
                    draws=count('draw'),
                    unresolved=sum(undecided(g) for g in games),
                    unresolved_counted=sum(undecided(g) and g['game'] not in exempt for g in games),
                    open=sum(g['phase'] in OPEN_PHASES for g in games))

    # ---- writing -------------------------------------------------------
    def _append(self, event):
        event = validate_event(dict(event, v=SCHEMA, t=int(self.clock())), max_game=self.max_game)
        data = (json.dumps(event, sort_keys=True, allow_nan=False) + '\n').encode()
        os.write(self.fd, data)
        os.fsync(self.fd)
        self.events.append(event)
        return event

    def recover(self, *, resume=False):
        """Handle an open game left by a dead process.

        Without ``resume`` it becomes ``unresolved`` and keeps consuming its slot.
        With ``resume`` the open game number is returned and reused (no new slot).
        """
        row = self.open_game()
        if row is None:
            if resume:
                raise LedgerRefused('no_open_game_to_resume')
            return None
        if resume:
            self._append(dict(ev='resumed', game=row['game']))
            return row['game']
        self._append(dict(ev='recovered', game=row['game'], terminal='unresolved',
                          outcome='unknown', stop_reason='process_restart',
                          ended_at=int(self.clock())))
        return None

    def reserve(self, conditions, *, accept_unresolved=False):
        """Reserve the next game number (fsynced) before any matchmaking action."""
        games = self.games()
        if self.open_game() is not None:
            raise LedgerRefused('another_game_open')
        # Two independent caps: game numbers (budget) and games that count (hard_limit; every
        # reserved game except the verified preliminary never-applied ones, undecided included).
        if len(games) >= self.budget or len(games) - len(self.exempt_games()) >= self.hard_limit:
            raise LedgerRefused('budget_exhausted')
        if games and games[-1].get('terminal') != 'completed' and not accept_unresolved:
            raise LedgerRefused('previous_game_unresolved')
        game = len(games) + 1
        self._append(dict(ev='reserve', game=game, budget=self.budget, **conditions))
        return game

    def record(self, game, ev, **fields):
        if ev == 'reserve':
            raise ValueError('use_reserve')
        return self._append(dict(ev=ev, game=game, **fields))
