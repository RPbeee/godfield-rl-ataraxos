"""Capped official ranked (真剣タイマン) run: greedy m-xit1, no search, one game per process.

This is the single, user-approved exception to "no public matches" (see AGENTS.md):
at most 100 games in total, one at a time, started by the parent. Everything that
leaves this module is a number, a boolean or a fixed code. Names, user IDs, room
keys, URLs, chat, screenshots, console output and raw receive data are never
written, printed or returned; the Playwright profile and the ledger live in a
git-ignored 0700 directory. The browser is driven through the normal UI probe only.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from .official import ROOT, OfficialSession, OfficialStreams, exception_detail
from .official_ui import UIUnsupported, require
from .ranked_ledger import (GAME_NUMBER_MAX, HARD_LIMIT, Ledger, LedgerRefused, OPEN_PHASES, SNAKE,
                            WIRE_DISPLAY_OFFSET, derive_ratings, gain_cross, screen_check)

RANKED_BUTTON = '2人個人戦'
# Buttons that may appear on a screen between the single RANKED_BUTTON click and the game
# (unobserved on the real site; chosen per run with --step, see scripts/ranked_bot.md).
# Only these, only exact text, only once per game, only in this order. Anything that could
# leave or cancel the queue (やめる, キャンセル, はい/いいえ の確認, 戦いをあきらめる) is excluded.
# '戦う' is the real application button of the screen the RANKED_BUTTON click opens
# (実機観測 11-20: 真剣タイマン / レーティング / 戦う / 教典); the CLI presses it by default.
FIGHT_BUTTON = '戦う'
END_BUTTON = '戦いを終わる'   # end screen's own exit button; the only click after the result is recorded
STEP_BUTTONS = (FIGHT_BUTTON, '戦いを始める', '開始', 'OK')
# Fixed vocabulary for the screen report: only these words are ever reported, never body text.
SCREEN_VOCAB = STEP_BUTTONS + (RANKED_BUTTON, '真剣タイマン', '隠れ乱闘', '修行', '生誕する', 'キャンセル',
                               'やめる', '戻る', 'はい', 'いいえ', '待機', '待って', 'マッチ', '対戦相手',
                               'レート', 'レーティング', 'ルール', '戦いをあきらめる', 'G.F.',
                               # Diagnostic additions for the first real-screen flow (11-20). Words only.
                               'ログイン', '本人確認', '登録', '認証', 'アカウント', 'ゲスト', 'メール',
                               'Google', 'Apple', '連携', '利用規約', '参加', '参戦',
                               '対戦相手を探す', '対戦する', 'マッチング', '中止', '中断',
                               '対戦', 'エラー', '失敗', '接続', '教典', '混雑', '条件',
                               '対戦相手を待', '暗証番号', '合言葉')
DEFAULT_STATE_DIR = ROOT / 'runs/ranked100'
MODEL_NAME = 'ぼぼぼーぼぼーぼぼ'

# A visible rule against bots stops the run before any matchmaking action. The
# matching text is never stored or printed: only the fixed reason code is.
_BOT = r'(?:BOT|Bot|bot|ＢＯＴ|ボット|ロボット|自動操作|自動化|マクロ|プログラム)'
_BAN = r'(?:禁止|禁じ|お断り|不可|認めて|使用しない|利用しない)'
_NOTICE = re.compile(_BOT + r'.{0,24}' + _BAN + '|' + _BAN + r'.{0,24}' + _BOT)
_RATING_LABEL = re.compile(r'(?:レーティング|レート|Rating|RATING)')
_RATING_NUMBER = re.compile(r'(?:レーティング|レート|Rating|RATING)\s*[:：]?\s*(-?\d{1,6})')


def bot_notice(body):
    return _NOTICE.search(re.sub(r'\s+', ' ', body)) is not None


def lobby_screen(state):
    return all(t in state['body'] for t in ('修行', '隠れ乱闘', '真剣タイマン'))


def screen_rating(state):
    """The one rating number shown on the application screen, else None (digits only).

    Candidates: a number right after the label in the body text, or a numeric text node
    right after a label text node. Disagreement or absence gives None. No other text
    (names included) is ever read into the result.
    """
    found = {int(m.group(1)) for m in _RATING_NUMBER.finditer(state.get('body', ''))}
    texts = [t.get('text') for t in state.get('texts', []) if isinstance(t, dict)]
    for label, value in zip(texts, texts[1:]):
        if isinstance(label, str) and isinstance(value, str) \
                and re.fullmatch(r'\s*(?:レーティング|レート|Rating|RATING)\s*[:：]?\s*', label) \
                and re.fullmatch(r'\s*-?\d{1,6}\s*', value):
            found.add(int(value))
    return found.pop() if len(found) == 1 else None


# Raw wire rating -> official display rating: display = WIRE_DISPLAY_OFFSET + raw. Basis: the
# official public client (parent's copy, sha256 ff77fdd0..fdfd2b): JS 4059-4062 treat a missing
# rating/ratingGain as 0, 4123-4140 bK.ax = rating and bK.ay = gain, 13165 / 8975 / 9094 show
# 1500 + the raw value. Game 3: opponent raw 32 / gain -16 = 1532 -> 1516, our own raw fields all
# absent = 1500 before, 1516 on the screen after. None would mean "not established" (never show
# a raw value as a display value). A missing gain is 0 only once the formal end was received.
# (The constant itself lives in ranked_ledger so read-only tools can share it.)
_LABEL =r'(?:レーティング|レート|Rating|RATING)'
_SIGNED = r'[+＋\-−]\d{1,4}'
_RESULT_NUMBERS = re.compile(_LABEL + r'\s*[:：]?\s*(-?\d{1,6})(?:\s*[（(]?\s*(' + _SIGNED + r')\s*[）)]?)?')


def wire_to_display(raw, offset):
    """A raw wire rating on the display scale, or None when no verified offset is given."""
    return raw + offset if type(raw) is int and type(offset) is int else None


def seat_display(seen, derived, offset):
    """One seat's display rating before/after with provenance, from the wire (never a guess).

    ``seen``: RatingTap.seat_ratings; ``derived``: derive_ratings of the same seat. Sources:
    'wire' (an integer raw value + offset), 'wire_absent_zero' (an absent key read as raw 0, as
    the official client does; the gain only after the formal end), 'unknown' (null).
    """
    out = dict(before=None, after=None, before_src='unknown', after_src='unknown')
    if type(offset) is not int:
        return out
    if derived['before'] is not None:
        out.update(before=wire_to_display(derived['before'], offset), before_src='wire')
    elif seen['rating_absent']:
        out.update(before=offset, before_src='wire_absent_zero')
    if derived['after'] is not None:
        out.update(after=wire_to_display(derived['after'], offset), after_src='wire')
    elif out['before'] is not None and seen['gain'] is None and seen['final_gain_absent']:
        # No gain was ever observed: "absent = 0" holds only once the formal end was received.
        out.update(after=out['before'], after_src='wire_absent_zero')
    return out


def screen_result(state):
    """Rating and signed gain shown on a screen, digits only: ``dict(rating=int|None, gain=int|None)``.

    The rating is `screen_rating`'s unique number. The gain is a signed number written right
    after it in the body ("レーティング 1516 (+16)"), or a signed numeric text node directly after
    the rating number node. Several different candidates (or none) give None. No other text is read.
    """
    gains = {int(m.group(2).replace('＋', '+').replace('−', '-'))
             for m in _RESULT_NUMBERS.finditer(state.get('body', '')) if m.group(2)}
    texts = [t.get('text') for t in state.get('texts', []) if isinstance(t, dict)]
    for number, signed in zip(texts[1:], texts[2:]):
        if isinstance(number, str) and isinstance(signed, str) and re.fullmatch(r'\s*-?\d{1,6}\s*', number) \
                and re.fullmatch(r'\s*' + _SIGNED + r'\s*', signed):
            gains.add(int(signed.strip().replace('＋', '+').replace('−', '-')))
    rating = screen_rating(state)
    return dict(rating=rating, gain=gains.pop() if rating is not None and len(gains) == 1 else None)


def exact_words(state):
    """SCREEN_VOCAB words that are a whole text node (not a substring of the body)."""
    texts = {t.get('text') for t in state.get('texts', []) if isinstance(t, dict)}
    return [w for w in SCREEN_VOCAB if w in texts]


def in_game_screen(state):
    return 'G.F.' in state['body']


def ranked_button_unique(state):
    return sum(t.get('text') == RANKED_BUTTON for t in state.get('texts', [])) == 1


def visible_words(state):
    """Which SCREEN_VOCAB words the current screen shows (a fixed-vocabulary projection)."""
    body = state.get('body', '')
    texts = {t.get('text') for t in state.get('texts', []) if isinstance(t, dict)}
    return [w for w in SCREEN_VOCAB if w in texts or w in body]  # SCREEN_VOCAB has no duplicates


def login_present(profile):
    """Whether the dedicated profile still holds a persisted Firebase Auth user.

    Only the presence of the record's key name is searched, in memory; the value
    (token, user ID) is never kept, printed or stored. A missing record (or one the
    LevelDB compression hides) errs on the safe side: no re-entry click.
    """
    try:
        for path in Path(profile).glob('Default/IndexedDB/*godfield*/*'):
            if path.is_file() and path.suffix in ('.ldb', '.log') \
                    and b'firebase:authUser' in path.read_bytes():
                return True
    except OSError:
        pass
    return False


def entry_action(*, birth_screen, allow_birth, reenter, marker, login):
    """What to do on the fixed 生誕 screen. Pure decision; non-action values are stop reasons.

    The site shows this screen on every fresh page load (also for a signed-in
    browser), so a second process on the same profile always meets it. Clicking it
    is a re-entry only if the persisted login is reused (same user, same rating);
    that must be confirmed on a throwaway copy of the profile before ``reenter`` is
    used on the dedicated one (scripts/ranked_bot.md).
    """
    if not birth_screen:
        return 'none'
    if marker and reenter:
        return 'reenter' if login else 'reenter_login_missing'
    if not allow_birth:
        return 'reenter_not_allowed' if marker else 'birth_not_allowed'
    return 'birth_screen_with_existing_profile' if marker else 'birth'


def safe_reason(code):
    return code if isinstance(code, str) and SNAKE.fullmatch(code) else 'other'


# Allowlisted fixed subcodes for ranked `unsupported_receive` stops.
# Values are static snake_case literals (<= 64 chars); free text, names, unknown keys
# and raw payload values can never cross this boundary.
RECEIVE_REASON_CODES = {
    # Parser errors (third_party/goodfield/scripts/parse_official_webchannel.py)
    'invalid_game_schema': 'ur_invalid_game_schema',
    'invalid_firestore_value': 'ur_invalid_firestore_value',
    'unsupported_firestore_value': 'ur_unsupported_firestore_value',
    'invalid_string': 'ur_invalid_string',
    'duplicate_json_key': 'ur_duplicate_json_key',
    'nonfinite_json_number': 'ur_nonfinite_json_number',
    'invalid_utf8': 'ur_invalid_utf8',
    'invalid_framing': 'ur_invalid_framing',
    'invalid_length_header': 'ur_invalid_length_header',
    'invalid_utf16_boundary': 'ur_invalid_utf16_boundary',
    'invalid_complete_json_frame': 'ur_invalid_complete_json_frame',
    'invalid_game_document': 'ur_invalid_game_document',
    # Boundary ValueError codes (official.VALUE_ERROR_CODES)
    'capture_failed': 'ur_capture_failed',
    'capture_integrity': 'ur_capture_integrity',
    'multiple_documents': 'ur_multiple_documents',
    'conflicting_update': 'ur_conflicting_update',
    'invalid_public_integer': 'ur_invalid_public_integer',
    'unsupported_roster': 'ur_unsupported_roster',
    'ambiguous_own_item_ids': 'ur_ambiguous_own_item_ids',
    # Session / stream unsupported codes (official.py)
    'team_game': 'ur_team_game',
    'hand_over_capacity': 'ur_hand_over_capacity',
    'sale_recipient_unavailable': 'ur_sale_recipient_unavailable',
    'unknown_event': 'ur_unknown_event',
    'ambiguous_pending_actor': 'ur_ambiguous_pending_actor',
    'dynamic_hit_rate_unavailable': 'ur_dynamic_hit_rate_unavailable',
    'death_attack_decision_unavailable': 'ur_death_attack_decision_unavailable',
    'redirect_requires_history': 'ur_redirect_requires_history',
    'dying_players_decision_unavailable': 'ur_dying_players_decision_unavailable',
    'confused_main': 'ur_confused_main',
    'invalid_or_unknown_public_state': 'ur_invalid_or_unknown_public_state',
    'out_of_order_or_conflicting_update': 'ur_out_of_order_or_conflicting_update',
    'new_game_requires_new_session': 'ur_new_game_requires_new_session',
    'dynamic_attack_value_unavailable': 'ur_dynamic_attack_value_unavailable',
    'earth_moon_payload': 'ur_earth_moon_payload',
    'redirect_chance_activation_unavailable': 'ur_redirect_chance_activation_unavailable',
    'unsupported_selection': 'ur_unsupported_selection',
    'stream_reader_stopped': 'ur_stream_reader_stopped',
    'invalid_or_unknown_stream': 'ur_invalid_or_unknown_stream',
}

SCHEMA_PATH_CODES = {
    'game': 'ur_schema_game',
    'game.players': 'ur_schema_players',
    'game.players[]': 'ur_schema_players',
    'game.players[].items': 'ur_schema_items',
    'game.players[].items[]': 'ur_schema_items',
    'game.players[].curses': 'ur_schema_curses',
    'game.players[].curses[]': 'ur_schema_curses',
    'game.events': 'ur_schema_events',
    'game.events[]': 'ur_schema_events',
    'game.attacks': 'ur_schema_attacks',
    'game.attacks[]': 'ur_schema_attacks',
    'game.attacks[].hitRate': 'ur_schema_hitrate',
    'game.attacks[].buyingItemModelId': 'ur_schema_buying',
    'game.dyingPlayerIds': 'ur_schema_dying',
}

EXC_TYPE_CODES = {
    'ValueError': 'ur_exc_valueerror',
    'KeyError': 'ur_exc_keyerror',
    'TypeError': 'ur_exc_typeerror',
    'IndexError': 'ur_exc_indexerror',
    'RuntimeError': 'ur_exc_runtimeerror',
}

CONFLICT_PATH_RULES = (
    (lambda p: p == 'is_over', 'ur_cu_is_over'),
    (lambda p: p == 'pending' or p.startswith('pending.'), 'ur_cu_pending'),
    (lambda p: p == 'purchase' or p.startswith('purchase.'), 'ur_cu_purchase'),
    (lambda p: p == 'hand' or p.startswith('hand['), 'ur_cu_hand'),
    (lambda p: p.startswith('players[') and '.dead' in p, 'ur_cu_players_dead'),
    (lambda p: p.startswith('players[') and '.status' in p, 'ur_cu_players_status'),
    (lambda p: p.startswith('players[') and '.used' in p, 'ur_cu_players_used'),
    (lambda p: p.startswith('players[') and '.fogged' in p, 'ur_cu_players_fogged'),
    (lambda p: p == 'players' or p.startswith('players['), 'ur_cu_players'),
    (lambda p: p == 'phase', 'ur_cu_phase'),
    (lambda p: p == 'actor', 'ur_cu_actor'),
    (lambda p: p == 'owner', 'ur_cu_owner'),
    (lambda p: p == 'gf', 'ur_cu_gf'),
    (lambda p: p == 'tiebreak_gf', 'ur_cu_tiebreak_gf'),
    (lambda p: p == 'unsupported' or p.startswith('unsupported[') or p == 'detail' or p.startswith('detail['),
     'ur_cu_unsupported'),
)


def _conflict_reason(detail):
    diffs = detail.get('differences') if isinstance(detail, dict) else None
    if not isinstance(diffs, list) or not diffs:
        return 'ur_conflicting_update'
    paths = [d.get('path') for d in diffs if isinstance(d, dict) and isinstance(d.get('path'), str)]
    if not paths:
        return 'ur_conflicting_update'
    if all(p == 'events' or p.startswith('events[') for p in paths):
        return 'ur_cu_events_only'
    for match, code in CONFLICT_PATH_RULES:
        if any(match(p) for p in paths):
            return code
    return 'ur_conflicting_update'


def _schema_path_reason(parser_detail):
    path = parser_detail.get('path') if isinstance(parser_detail, dict) else None
    if not isinstance(path, str):
        return 'ur_schema_other'
    if path in SCHEMA_PATH_CODES:
        return SCHEMA_PATH_CODES[path]
    if path.startswith('game.players[].items'):
        return 'ur_schema_items'
    if path.startswith('game.players[].curses'):
        return 'ur_schema_curses'
    if path.startswith('game.players[]'):
        return 'ur_schema_players'
    if path.startswith('game.events[]'):
        return 'ur_schema_events'
    if path.startswith('game.attacks[]'):
        return 'ur_schema_attacks'
    return 'ur_schema_other'


def ranked_receive_reason(detail):
    """Map an ``unsupported_receive`` detail object to a static allowlisted enum code."""
    if isinstance(detail, list):
        for item in detail:
            if not isinstance(item, dict):
                continue
            sub = item.get('code')
            if sub == 'multiple_pending_frames':
                continue
            if sub == 'unsupported_game_schema':
                return _schema_path_reason(item.get('parser'))
            if isinstance(sub, str) and sub in RECEIVE_REASON_CODES:
                return RECEIVE_REASON_CODES[sub]
        return 'unsupported_receive'
    if isinstance(detail, dict):
        sub = detail.get('code')
        if sub == 'unsupported_game_schema':
            return _schema_path_reason(detail.get('parser'))
        if sub == 'conflicting_update':
            return _conflict_reason(detail)
        if isinstance(sub, str) and sub in RECEIVE_REASON_CODES:
            return RECEIVE_REASON_CODES[sub]
        if sub is None and isinstance(detail.get('error'), str):
            return EXC_TYPE_CODES.get(detail['error'], 'ur_exc_other')
    return 'unsupported_receive'


def ranked_stop_code(error, default='bot_input_or_model_failure'):
    """Fixed stop code for ranked mode, preserving allowlisted receive subcodes."""
    if not isinstance(error, UIUnsupported):
        return default
    code = str(error)
    if code == 'unsupported_receive':
        return ranked_receive_reason(getattr(error, 'detail', None))
    return safe_reason(code)


class NullJournal:
    """Ranked runs keep no per-move log: the ledger is the only record."""
    directory = None

    def write(self, value):
        pass

    def close(self):
        pass


class RatingTap:
    """Collects integers per update from raw duel receives (never text)."""
    def __init__(self):
        self.updates = {}

    @staticmethod
    def extract(game):
        def number(value):
            return value if type(value) is int else None
        if not isinstance(game, dict) or type(game.get('updateCount')) is not int:
            return None
        def key_state(p, key):
            # 'int' / 'absent' (key not in a known player row) / 'invalid' (present, not an int).
            return 'absent' if key not in p else 'int' if number(p[key]) is not None else 'invalid'
        players = {}
        for p in game.get('players', []):
            if isinstance(p, dict) and type(p.get('id')) is int:
                players[p['id']] = dict(rating=number(p.get('rating')), gain=number(p.get('ratingGain')),
                                        rating_key=key_state(p, 'rating'), gain_key=key_state(p, 'ratingGain'),
                                        bot=p.get('isBot') is True)
        end = game.get('tiebreakGF')
        return dict(update=game['updateCount'], over=game.get('isOver') is True,
                    end_gf=end if type(end) is int else None, players=players)

    def add(self, observed):
        if observed is not None:
            self.updates[observed['update']] = observed

    def ordered(self):
        return [self.updates[k] for k in sorted(self.updates)]

    def seat_ratings(self, seat):
        rows = [o for o in self.ordered() if seat in o['players']]
        rated = [o for o in rows if o['players'][seat]['rating'] is not None]
        # Missing-as-0 (official client) only for a known seat whose rating key was absent in every
        # row, and for the gain only in a formal-end (isOver) row; an invalid type blocks it.
        keys = [o['players'][seat] for o in rows]
        absent = dict(rating_absent=bool(rows) and all(k['rating_key'] == 'absent' for k in keys),
                      final_gain_absent=any(o['over'] and o['players'][seat]['gain_key'] == 'absent' for o in rows)
                      and all(k['gain_key'] != 'invalid' for k in keys))
        if not rated:
            return dict(first=None, last=None, gain=None, pregame=False, **absent)
        first, last = rated[0], rated[-1]
        gain = next((o['players'][seat]['gain'] for o in reversed(rows)
                     if o['players'][seat]['gain'] is not None), None)
        pregame = (not first['over']) and first['players'][seat]['gain'] in (None, 0)
        return dict(first=first['players'][seat]['rating'], last=last['players'][seat]['rating'],
                    gain=gain, pregame=pregame, **absent)


class RankedSession:
    def __init__(self, ledger, conditions, *, name, match_wait=900, rating_wait=10, timeout=20,
                 settle_timeout=30, resume=False, allow_birth=False, accept_unresolved=False,
                 strict_events=False, marker=None, reenter=False, profile=None, steps=(),
                 screen_file=None, clock=time.time, rating_screen_wait=30, wire_offset=WIRE_DISPLAY_OFFSET, leave_wait=5):
        steps = tuple(steps)
        self.leave_wait = leave_wait
        self.rating_screen_wait = rating_screen_wait
        self.wire_offset = wire_offset
        self.fight_since = None       # monotonic time the application button was first usable
        self.fight_candidate = None   # last screen rating seen there
        self.fight_stable = 0         # consecutive identical polls of it
        self.rating_after = None      # result screen (display values)
        self.gain_after = None
        require(all(step in STEP_BUTTONS for step in steps) and len(set(steps)) == len(steps),
                'invalid_step')
        self.steps = steps
        self.screen_file = screen_file
        self.screen_seen = None
        self.screen_polls = 0
        self.steps_done = []
        self.screen = []
        self.lobby_words = frozenset()
        self.rating_screen = None
        self.ledger = ledger
        self.conditions = conditions
        self.name = name
        self.match_wait = match_wait
        self.rating_wait = rating_wait
        self.timeout = timeout
        self.settle_timeout = settle_timeout
        self.resume = resume
        self.allow_birth = allow_birth
        self.accept_unresolved = accept_unresolved
        self.strict_events = strict_events
        self.marker = marker
        self.reenter = reenter
        self.profile = profile
        self.clock = clock
        self.game = None
        self.viewer = None
        self.decisions = 0
        self.updates = 0
        self.last_update = None
        self.gf = None
        self.first_mover = None
        self.started_wall = None
        self.tap = RatingTap()
        self.finished = False
        self.resumed = False

    # ---- entry ---------------------------------------------------------
    def enter(self, probe):
        from .official_bot import capture, poll_state, wait_login
        state = poll_state(probe, lambda s: '生誕する' in s['body'] or lobby_screen(s)
                           or in_game_screen(s), self.settle_timeout)
        require(in_game_screen(state) or not bot_notice(state['body']), 'bot_prohibition_notice')
        if in_game_screen(state):
            # An account that is already in a game is never matched again. A
            # resume reuses the open game number (no second slot, no new click).
            require(self.resume and self.ledger.open_game() is not None,
                    'ranked_game_in_progress_on_start')
            self.game = self.ledger.recover(resume=True)
            self.resumed = True
            return []
        if '生誕する' in state['body']:
            state = self.pass_birth_screen(probe, state, wait_login)
        require(lobby_screen(state), 'ranked_lobby_missing')
        require(not bot_notice(state['body']), 'bot_prohibition_notice')
        require(ranked_button_unique(state), 'ranked_button_not_unique')
        if self.ledger.open_game() is not None:
            self.ledger.recover(resume=False)  # Dead process; the slot stays consumed.
        baseline = capture(probe)
        self.lobby_words = frozenset(exact_words(state))
        # Reserve (fsynced) BEFORE the single matchmaking click; never retried.
        self.game = self.ledger.reserve(self.conditions, accept_unresolved=self.accept_unresolved)
        probe.send(dict(kind='clickText', text=RANKED_BUTTON))
        self.ledger.record(self.game, 'queued')
        return baseline

    def advance(self, probe, state):
        """Click the next configured intermediate button of the ALREADY queued game.

        Never touches the ledger (no new slot, no second RANKED_BUTTON click). Conditions:
        not in the game, the next step's exact text is visible exactly once, and that text
        was NOT already on the lobby before the matchmaking click (so a new screen or dialog
        produced it, even if it overlays the lobby). The step is marked done BEFORE the click
        so it can never be repeated, even if the click fails.
        """
        if len(self.steps_done) >= len(self.steps) or in_game_screen(state):
            return
        want = self.steps[len(self.steps_done)]
        if want in self.lobby_words or lobby_screen(state):
            return
        if sum(t.get('text') == want for t in state.get('texts', []) if isinstance(t, dict)) != 1:
            return
        if want == FIGHT_BUTTON and '真剣タイマン' not in state['body']:
            return  # Only the ranked application screen has the application button.
        # The slot must already be reserved (fsynced) and open before anything is applied.
        row = next((g for g in self.ledger.games() if g['game'] == self.game), None)
        require(row is not None and row['phase'] in OPEN_PHASES, 'ranked_not_reserved')
        if want == FIGHT_BUTTON:
            # Our own starting rating is the number on this screen. It loads after the button
            # does, so wait until the same number was read twice in a row; never apply without
            # it. Giving up here is a stop BEFORE the application (the slot stays consumed).
            shown = screen_rating(state)
            now = time.monotonic()
            if self.fight_since is None:
                self.fight_since = now
            self.fight_stable = self.fight_stable + 1 if shown is not None and shown == self.fight_candidate \
                else (1 if shown is not None else 0)
            self.fight_candidate = shown
            if self.fight_stable < 2:
                require(now - self.fight_since <= self.rating_screen_wait, 'rating_screen_not_loaded')
                return
            self.rating_screen = shown
            # Persisted before the click (same phase, so no second slot).
            self.ledger.record(self.game, 'queued', rating_screen=self.rating_screen)
        self.steps_done.append(want)
        probe.send(dict(kind='clickText', text=want))

    def watch_screen(self, state, started):
        """Write the latest fixed-vocabulary projection of the waiting screen when it changes.

        Diagnostic only (``--watch-screen``). The file holds counts, booleans and words that
        are constants of SCREEN_VOCAB: never names, typed values, raw text, IDs or URLs.
        Nothing is clicked or decided from it.
        """
        self.screen_polls += 1
        if self.screen_file is None:
            return
        words = visible_words(state)
        if words == self.screen_seen:
            return
        self.screen_seen = words
        report = dict(stage='wait_start', words=words, word_count=len(words),
                      lobby=lobby_screen(state), in_game=in_game_screen(state),
                      ranked_button_unique=ranked_button_unique(state),
                      polls=self.screen_polls, waited_s=int(time.monotonic() - started),
                      steps_done=list(self.steps_done))
        try:
            tmp = Path(str(self.screen_file) + '.tmp')
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as handle:
                handle.write(json.dumps(report, sort_keys=True) + '\n')
            os.replace(tmp, self.screen_file)
        except OSError:
            pass

    def stop_detail(self):
        """Fixed-vocabulary words and the step clicks of a queued game; empty before queueing."""
        if self.game is None:
            return {}
        return dict(screen=list(self.screen), steps=list(self.steps_done))

    def birth_action(self, state):
        return entry_action(birth_screen='生誕する' in state['body'], allow_birth=self.allow_birth,
                            reenter=self.reenter, marker=self.marker is not None and self.marker.exists(),
                            login=self.profile is not None and login_present(self.profile))

    def pass_birth_screen(self, probe, state, wait_login):
        """Enter through 生誕: a first birth, or (``reenter``) the same account again."""
        action = self.birth_action(state)
        require(action in ('birth', 'reenter'), action)
        if action == 'birth':
            # Let a persisted login restore before concluding the profile is new.
            time.sleep(min(8, self.settle_timeout))
        probe.send(dict(kind='fillName', text=self.name))
        probe.send(dict(kind='clickText', text='生誕する'))
        if action == 'birth' and self.marker is not None:
            self.marker.write_text('1\n'); self.marker.chmod(0o600)
        # A rejected name leaves the name screen: login_not_accepted, nothing is reserved.
        return wait_login(probe, lobby_screen, max(self.timeout, self.settle_timeout))

    def discover(self, probe):
        """Lobby check only: never reserves, never clicks the ranked button."""
        from .official_bot import poll_state, wait_login
        state = poll_state(probe, lambda s: '生誕する' in s['body'] or lobby_screen(s)
                           or in_game_screen(s), self.settle_timeout)
        action = self.birth_action(state)
        if action in ('birth', 'reenter'):
            state = self.pass_birth_screen(probe, state, wait_login)
        body = state['body']
        candidate = _RATING_NUMBER.search(body)
        return dict(birth_screen='生誕する' in body, entry=action, screen=visible_words(state),
                    born_marker=self.marker is not None and self.marker.exists(),
                    profile_login_present=self.profile is not None and login_present(self.profile), lobby=lobby_screen(state),
                    in_game=in_game_screen(state), ranked_button_unique=ranked_button_unique(state),
                    bot_notice=bot_notice(body), rating_label_seen=_RATING_LABEL.search(body) is not None,
                    # Diagnostic only: the lobby label is not a verified source.
                    rating_candidate=int(candidate.group(1)) if candidate else None)

    # ---- game ----------------------------------------------------------
    def stream_parser(self, baseline, viewer):
        from .official_bot import named_room_parser
        parser, _ = named_room_parser(self.name, baseline, viewer, require_no_end=False, tap=self.tap)
        return parser

    def wait_start(self, probe, baseline):
        from .official_bot import capture, checked_capture, named_room_parser
        parser, boundaries = named_room_parser(self.name, baseline, require_no_end=False, tap=self.tap)
        started = time.monotonic()
        deadline = started + self.match_wait
        while time.monotonic() < deadline:
            state = probe.send(dict(kind='state'))['state']
            require(in_game_screen(state) or not bot_notice(state['body']), 'bot_prohibition_notice')
            viewers = set()
            for c in capture(probe):
                raw = checked_capture(c)
                try:
                    records = parser.parse_prefix(raw, boundary=boundaries.get(OfficialStreams.stream_key(c['file']), 0),
                                                  lenient_events=not self.strict_events)['games']
                except parser.ParseError as error:
                    raise UIUnsupported('unsupported_receive', exception_detail(error, parser.ParseError)) from None
                viewers.update(r['room_viewer'] for r in records if 'room_viewer' in r)
            require(len(viewers) <= 1, 'viewer_not_unique')
            if not viewers:
                self.screen = visible_words(state)
                self.watch_screen(state, started)
                self.advance(probe, state)
            if viewers:
                self.viewer = viewers.pop()
                self.started_wall = int(self.clock())
                observed = self.tap.ordered()
                opponent = [p for o in observed for seat, p in o['players'].items() if seat != self.viewer]
                kind = 'unknown' if not opponent else 'bot' if opponent[-1]['bot'] else 'human'
                end_gf = observed[-1]['end_gf'] if observed else None
                row = next((g for g in self.ledger.games() if g['game'] == self.game), {})
                if self.resumed and row.get('started_at') is not None:
                    self.started_wall = row['started_at']  # Keep the original start.
                else:
                    self.ledger.record(self.game, 'started', started_at=self.started_wall,
                                       seat_id=self.viewer, opp_kind=kind, end_gf=end_gf,
                                       **self.screen_fields())
                return self.viewer
            time.sleep(.5)
        raise UIUnsupported('match_wait_timeout')

    def observe(self, public):
        if public['update'] != self.last_update:
            self.last_update = public['update']
            self.updates += 1
        self.gf = public['gf']
        if self.first_mover is None and not self.resumed and public.get('owner') is not None:
            self.first_mover = 'self' if public['owner'] == self.viewer else 'opponent'

    def rating_fields(self):
        mine = self.tap.seat_ratings(self.viewer)
        other_seat = next((s for o in self.tap.ordered() for s in o['players'] if s != self.viewer), None)
        theirs = self.tap.seat_ratings(other_seat) if other_seat is not None else dict(
            first=None, last=None, gain=None, pregame=False, rating_absent=False, final_gain_absent=False)
        a = derive_ratings(mine['first'], mine['last'], mine['gain'], first_was_pregame=mine['pregame'])
        b = derive_ratings(theirs['first'], theirs['last'], theirs['gain'], first_was_pregame=theirs['pregame'])
        origin = a['origin'] if a['origin'] != 'unknown' else b['origin']
        # Raw wire values stay raw (rating_*_first/last/gain/before/after). Display values are
        # offset + raw, with an absent key read as 0 (official client), each with its source. For
        # our own seat the screen is the official display itself: the post-game screen wins over
        # the wire, the pre-match screen only fills a missing start.
        me, opp = (seat_display(mine, a, self.wire_offset), seat_display(theirs, b, self.wire_offset))
        if self.rating_after is not None:
            me.update(after=self.rating_after, after_src='screen')
        if me['before'] is None and self.rating_screen is not None:
            me.update(before=self.rating_screen, before_src='screen')
        return dict(rating_self_first=mine['first'], rating_self_last=mine['last'], rating_self_gain=mine['gain'],
                    rating_opp_first=theirs['first'], rating_opp_last=theirs['last'], rating_opp_gain=theirs['gain'],
                    rating_self_before=a['before'], rating_self_after=a['after'],
                    rating_opp_before=b['before'], rating_opp_after=b['after'], rating_origin=origin,
                    rating_self_display_before=me['before'], rating_self_display_after=me['after'],
                    rating_self_display_before_src=me['before_src'], rating_self_display_after_src=me['after_src'],
                    rating_opp_display_before=opp['before'], rating_opp_display_after=opp['after'],
                    rating_opp_display_before_src=opp['before_src'], rating_opp_display_after_src=opp['after_src'],
                    rating_display_basis='offset_verified' if self.wire_offset is not None else 'unconverted',
                    rating_gain_cross=gain_cross(self.rating_screen, self.rating_after, self.gain_after,
                                                 theirs['gain']))

    def screen_fields(self):
        return {} if self.rating_screen is None else dict(rating_screen=self.rating_screen)

    def result_fields(self):
        """Post-game screen values and their cross-check with the pre-match screen value."""
        fields = {} if self.rating_after is None else dict(rating_screen_after=self.rating_after,
                                                          rating_screen_gain=self.gain_after)
        fields['rating_screen_check'] = screen_check(self.rating_screen, self.rating_after, self.gain_after)
        return fields

    def read_result_screen(self, probe):
        """Take the latest display rating (and signed gain) from the screen after the game.

        Digits only, and only if the screen shows exactly one rating number; a screen that has
        not updated yet (same number, no gain) is kept as a weaker reading and may be replaced.
        """
        try:
            shown = screen_result(probe.send(dict(kind='state'))['state'])
        except Exception:
            return False
        if shown['rating'] is None:
            return False
        self.rating_after, self.gain_after = shown['rating'], shown['gain']
        return self.gain_after is not None or (self.rating_screen is not None
                                               and shown['rating'] != self.rating_screen)

    def timing(self):
        now = int(self.clock())
        start = self.started_wall if self.started_wall is not None else now
        return dict(ended_at=now, duration_s=max(0, now - start))

    def finish(self, probe, streams, outcome):
        from .official_bot import capture
        outcome = dict(wins='win', losses='loss', draws='draw').get(outcome, 'unknown')
        # The server may publish the rating change slightly after the last move.
        # The result screen (display rating and gain) is read in the same window.
        end = time.monotonic() + self.rating_wait
        while time.monotonic() < end:
            try:
                streams.poll(capture(probe))
            except Exception:
                break
            seats = self.tap.ordered()[-1]['players'] if self.tap.ordered() else {}
            screen_done = self.read_result_screen(probe)
            if seats and all(p['gain'] is not None for p in seats.values()) and screen_done:
                break
            time.sleep(.5)
        self.ledger.record(self.game, 'finish', terminal='completed', outcome=outcome,
                           first_mover=self.first_mover, decisions=self.decisions,
                           updates=self.updates, gf_final=self.gf, **self.timing(), **self.rating_fields(),
                           **self.screen_fields(), **self.result_fields())
        self.finished = True

    def leave_end_screen(self, probe):
        """After the result is fsynced in the ledger, press the end screen's own button once.

        Ranked only. Needs the official end (``finished`` is set only after the ledger's finish
        row was written and fsynced). Waits briefly for the exact word to show once, clicks it at
        most once, then checks that the game screen was left. Nothing here touches the ledger and
        any failure is swallowed: the recorded result and rating stay as they are. Returns a
        fixed enum.
        """
        if not self.finished or self.game is None:
            return 'not_finished'
        try:
            end = time.monotonic() + self.leave_wait
            while True:
                state = probe.send(dict(kind='state'))['state']
                words = state.get('body', '').count(END_BUTTON)
                nodes = sum(t.get('text') == END_BUTTON for t in state.get('texts', []) if isinstance(t, dict))
                if words == 1 and nodes <= 1:
                    break
                if time.monotonic() >= end:
                    return 'button_absent' if words == 0 else 'button_ambiguous'
                time.sleep(.2)
            probe.send(dict(kind='clickText', text=END_BUTTON))      # the only click, never repeated
            end = time.monotonic() + self.leave_wait
            while time.monotonic() < end:
                state = probe.send(dict(kind='state'))['state']
                if END_BUTTON not in state.get('body', '') and not in_game_screen(state):
                    return 'left'
                time.sleep(.2)
            return 'left_unconfirmed'
        except Exception:
            return 'leave_failed'

    def stop_code(self, error):
        return ranked_stop_code(error)

    def stopped(self, code):
        """Record an interrupted/undecided game conservatively; return a safe code."""
        reason = safe_reason(code)
        game = self.game
        if game is None or self.finished:
            return reason
        row = next((g for g in self.ledger.games() if g['game'] == game), None)
        if row is None or row['phase'] not in ('reserved', 'queued', 'started'):
            return reason
        fields = dict(terminal='stopped', outcome='unknown', stop_reason=reason,
                      decisions=self.decisions, updates=self.updates, gf_final=self.gf,
                      first_mover=self.first_mover, **self.timing())
        fields.update(self.screen_fields())
        if self.viewer is not None:
            fields.update(self.rating_fields())
            fields.update(self.result_fields())
        try:
            self.ledger.record(game, 'stop', **fields)
        except Exception:
            pass
        return reason


def checkpoint_conditions(checkpoint, dream='allow'):
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:8]
    label = Path(checkpoint).stem
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,32}', label):
        label = 'custom'
    try:
        rev = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short=8', 'HEAD'],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        rev = ''
    return dict(model=label, ckpt_sha8=digest, code_rev=rev if re.fullmatch(r'[0-9a-f]{8}', rev) else 'unknown',
                policy='greedy', search=False, dream=dream)


def ensure_ignored(path):
    """Refuse a state directory inside the repository that git would track."""
    path = Path(path).absolute()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    ignored = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', str(path / 'x')])
    require(ignored.returncode == 0, 'state_dir_not_git_ignored')


def build_parser():
    parser = argparse.ArgumentParser(description='Capped official ranked run (m-xit1 greedy, no search)')
    parser.add_argument('--checkpoint', default=str(ROOT / 'runs/play/m-xit1.pt'))
    parser.add_argument('--playwright', type=Path)
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument('--budget', type=int, default=1,
                        help='Total game numbers (slots) this ledger may ever hold, preliminary ones included '
                             f'(1, then 10, then up to {HARD_LIMIT}; up to {GAME_NUMBER_MAX} only while the two '
                             f'never-applied preliminary games stay verified in the ledger). At most {HARD_LIMIT} '
                             'games ever count, whatever the budget')
    parser.add_argument('--name', default=MODEL_NAME, help='Dedicated account name (never logged)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dream', choices=['allow', 'last-resort'], default='allow')
    parser.add_argument('--match-wait', type=float, default=900)
    parser.add_argument('--turn-wait', type=float, default=900)
    parser.add_argument('--rating-wait', type=float, default=10)
    parser.add_argument('--rating-screen-wait', type=float, default=30,
                        help='Seconds to wait for our own rating number on the screen before 戦う; '
                             'never applies without it (stop code rating_screen_not_loaded)')
    parser.add_argument('--leave-wait', type=float, default=5,
                        help='Seconds to wait for the end screen button 戦いを終わる (pressed once, after the '
                             'result is in the ledger; failure never changes the recorded result)')
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--settle-timeout', type=float, default=30)
    parser.add_argument('--chance-delay', type=float, default=3)
    parser.add_argument('--strict-events', action='store_true')
    parser.add_argument('--allow-birth', action='store_true',
                        help='Create the dedicated account once if the profile has none')
    parser.add_argument('--reenter', action='store_true',
                        help='Click 生誕 again on the profile that already holds the dedicated account '
                             '(every fresh page load shows it). Needs the persisted login; unverified on the '
                             'real site until a throwaway-copy check passed (scripts/ranked_bot.md)')
    parser.add_argument('--step', action='append', choices=STEP_BUTTONS, default=[],
                        help='Extra button to click once (戦う is always pressed first by the CLI), after the single '
                             f'{RANKED_BUTTON} click (repeatable). Needs the exact text visible exactly once and '
                             'the lobby left; never reserves or applies again (scripts/ranked_bot.md)')
    parser.add_argument('--watch-screen', action='store_true',
                        help='Diagnostic: while waiting for the match, write the latest fixed-vocabulary '
                             'screen words (numbers/booleans/constants only) to <state-dir>/screen-state.json '
                             'whenever they change. Clicks nothing; unknown screens are never dumped')
    parser.add_argument('--resume', action='store_true', help='Re-attach to the open game of a dead process')
    parser.add_argument('--accept-unresolved', action='store_true',
                        help='Reserve the next game although the previous one has no recorded result')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--status', action='store_true', help='Print ledger totals; no browser')
    mode.add_argument('--discover', action='store_true', help='Lobby check only; no reservation, no ranked click')
    return parser


def emit(line):
    print(line, flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.budget <= GAME_NUMBER_MAX or not 0 < args.timeout < float('inf') \
            or not 0 < args.match_wait < float('inf') or not 0 < args.turn_wait < float('inf') \
            or not 0 < args.rating_screen_wait < float('inf') or len(set(args.step)) != len(args.step):
        emit('error:invalid_arguments'); return 2
    state_dir = args.state_dir.absolute()
    try:
        ensure_ignored(state_dir)
        ledger = Ledger(state_dir, args.budget)
    except (LedgerRefused, UIUnsupported) as error:
        emit('error:' + safe_reason(getattr(error, 'code', None) or str(error))); return 2
    session = probe = None
    try:
        if args.status:
            emit(json.dumps(dict(status='ledger', **ledger.summary()), sort_keys=True)); return 0
        if not args.playwright:
            emit('error:playwright_required'); return 2
        if not args.discover:
            if not Path(args.checkpoint).is_file():
                emit('error:checkpoint_missing'); return 2
        def terminate(*_):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, terminate)
        conditions = None if args.discover else checkpoint_conditions(args.checkpoint, args.dream)
        profile = state_dir / 'profile'
        session = RankedSession(ledger, conditions, name=args.name, match_wait=args.match_wait,
                                rating_wait=args.rating_wait, timeout=args.timeout,
                                settle_timeout=args.settle_timeout, resume=args.resume,
                                allow_birth=args.allow_birth, accept_unresolved=args.accept_unresolved,
                                strict_events=args.strict_events, marker=state_dir / 'profile-born',
                                reenter=args.reenter, profile=profile,
                                rating_screen_wait=args.rating_screen_wait, leave_wait=args.leave_wait,
                                steps=[FIGHT_BUTTON] + [x for x in args.step if x != FIGHT_BUTTON],
                                screen_file=state_dir / 'screen-state.json' if args.watch_screen else None)
        # Raw receive bytes never touch the disk in a ranked run: the probe keeps them in
        # memory and hands them over the stdout pipe (GFRL_MEMORY_ONLY, Probe(memory=True)).
        # A `scratch/` left by an older version of this runner is deleted unread.
        shutil.rmtree(state_dir / 'scratch', ignore_errors=True)
        profile.mkdir(mode=0o700, exist_ok=True); profile.chmod(0o700)
        from .official_bot import Probe, run
        env = dict(os.environ, GFRL_PROFILE_DIR=str(profile), GFRL_MEMORY_ONLY='1')
        # No diagnostic channel of node/Playwright/Chrome may be switched on from outside.
        for key in ('NODE_DEBUG', 'NODE_OPTIONS', 'PWDEBUG', 'CHROME_LOG_FILE', 'NODE_REPORT_DIRECTORY'):
            env.pop(key, None)
        env['DEBUG'] = ''
        sink = open(os.devnull, 'w')
        with contextlib.redirect_stderr(sink):
            probe = Probe(['node', str(ROOT / 'scripts/official_probe.mjs'),
                           '--playwright', str(args.playwright)], env=env, memory=True)
            if args.discover:
                report = dict(status='discover', **session.discover(probe))
            else:
                import torch
                from .actor import Actor
                from .evaluation import load_policy_model, _policy_rng
                from .official_advisor import advise_supported
                torch.set_num_threads(args.threads)
                model = load_policy_model(args.checkpoint, args.device)
                agent = None
                def advisor(result):
                    nonlocal agent
                    if agent is None:
                        agent = Actor(model, result['env'], args.device, sample=False)
                    return advise_supported(result, agent, search=False, dream=args.dream)
                with _policy_rng(0, ()):
                    outcome = run(probe, advisor, NullJournal(), name=args.name, games=1, mode='ranked',
                                  ranked=session, timeout=args.timeout, settle_timeout=args.settle_timeout,
                                  turn_wait=args.turn_wait, strict_events=args.strict_events,
                                  chance_delay=args.chance_delay)
                report = dict(outcome, ledger=ledger.summary())
        emit(json.dumps(report, sort_keys=True))
        return 1 if report.get('status') == 'stopped' else 0
    except BaseException as error:  # Never print messages/tracebacks: only a code.
        code = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else (
            ranked_stop_code(error) if isinstance(error, (UIUnsupported, LedgerRefused)) else 'ranked_internal_error')
        reason = session.stopped(code) if session is not None else safe_reason(code)
        detail = session.stop_detail() if session is not None else {}
        emit(json.dumps(dict(status='stopped', reason=safe_reason(reason), detail=detail), sort_keys=True))
        return 1
    finally:
        if probe is not None:
            try:
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    probe.close()
            except BaseException:
                pass
        ledger.close()


if __name__ == '__main__':
    raise SystemExit(main())
