"""Offline tests for the capped ranked run. Synthetic names only; no browser, no network.

The rating fields (rating / ratingGain / alias / team on duel players) follow the
pinned client's player decoder as documented in third_party/goodfield/docs/
official-state-view.md; they are documentation-derived, not live-observed fixtures.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading

import pytest

from gfrl.official import OfficialSession, OfficialStreams, parser_module
from gfrl.official_bot import named_room_parser, run
from gfrl.official_ranked import (NullJournal, RankedSession, RatingTap, bot_notice, checkpoint_conditions,
                                  entry_action, lobby_screen, login_present, main, ranked_button_unique,
                                  safe_reason, screen_rating, screen_result, seat_display, wire_to_display,
                                  WIRE_DISPLAY_OFFSET)
from gfrl.official_ui import UIUnsupported
from gfrl.ranked_ledger import (HARD_LIMIT, Ledger, LedgerRefused, derive_ratings, gain_cross, screen_check,
                                validate_event)
from test_official_bot import FakeProbe, GAMES, advisor, capture_file, room_clock

ME, OPP = 'SyntheticRankedMe', 'SyntheticRankedOpp'
COND = dict(model='m-xit1', ckpt_sha8='0123abcd', code_rev='89abcdef', policy='greedy',
            search=False, dream='allow')


def duel_games(*, ratings=(1500, 1480), gain=(8, -8), team=0, end=None, over_gain=True):
    """Training fixture turned into a rated duel: human names, rating, alias, team."""
    games = deepcopy(GAMES)
    for g in games:
        for p in g['players']:
            p['isBot'] = False
            p['name'] = ME if p['id'] == 1 else OPP
            p['alias'] = 'SyntheticAlias' + str(p['id'])
            p['team'] = team
            p['rating'] = ratings[p['id'] - 1]
        if end is not None:
            g['tiebreakGF'] = end
    if over_gain:
        for p in games[-1]['players']:
            p['ratingGain'] = gain[p['id'] - 1]
            p['rating'] += gain[p['id'] - 1]
    return games


class RankedProbe(FakeProbe):
    """Lobby -> single ranked click -> synthetic duel; counts every click."""
    def __init__(self, directory, *, games=None, lobby_body=None, button_count=1, birth=False,
                 in_game_on_load=False, never_match=False, step=None, step_count=1, step_sticky=False,
                 step_in_lobby=False):
        super().__init__(directory, games=games or duel_games())
        self.name = ME
        self.phase = 'birth' if birth else ('battle' if in_game_on_load else 'lobby')
        self.lobby_body = lobby_body or '教典\n本人\n修行\nコンピュータと対戦\n隠れ乱闘\n友達と対戦\n真剣タイマン\n2人個人戦'
        self.button_count = button_count
        self.never_match = never_match
        # An intermediate screen after the ranked click, with one button ``step`` (shown
        # ``step_count`` times). ``step_sticky``: clicking it does not leave the screen.
        # ``step_in_lobby``: the same word is already on the lobby before any click.
        self.step = step
        self.step_count = step_count
        self.step_sticky = step_sticky
        self.step_in_lobby = step_in_lobby
        self.clicks = []
        self.polls_waiting = 0

    def state(self):
        if self.phase == 'lobby':
            texts = [dict(text='2人個人戦', x=600, y=500, w=80, h=20)] * self.button_count
            body = self.lobby_body
            if self.step_in_lobby:
                texts += [dict(text=self.step, x=100, y=100, w=80, h=20)]
                body += '\n' + self.step
            return dict(body=body, inputs=[], images=[], texts=texts)
        if self.phase == 'step':
            texts = [dict(text=self.step, x=600, y=500, w=80, h=20)] * self.step_count
            return dict(body='対戦の準備\n' + self.step, inputs=[], images=[], texts=texts)
        if self.phase == 'queue':
            self.polls_waiting += 1
            if self.polls_waiting > 3 and not self.never_match:
                self.phase = 'battle'; self.index = 0
                return self.state()
            return dict(body='対戦相手を待っています', inputs=[], images=[], texts=[])
        if self.phase == 'battle':
            s = super().state()
            s['body'] = s['body'].replace('修行\n', '', 1)
            return s
        return super().state()

    def send(self, c):
        # Every command is recorded exactly once: here when handled locally, otherwise by
        # FakeProbe.send (recording before falling through used to double-count fillName).
        k = c['kind']
        if k == 'clickText':
            self.clicks.append(c['text'])
            if c['text'] == '2人個人戦':
                self.commands.append(c)
                self.phase = 'step' if self.step else 'queue'; return dict(state=self.state())
            if c['text'] == '生誕する':
                self.commands.append(c)
                self.phase = 'lobby'; return dict(state=self.state())
            if self.phase == 'step' and c['text'] == self.step:
                self.commands.append(c)
                self.phase = 'step' if self.step_sticky else 'queue'; return dict(state=self.state())
        if k == 'capture' and self.phase in ('lobby', 'queue', 'step'):
            self.commands.append(c)
            return dict(capture=dict(captures=[]))
        if k == 'screenshot':
            raise AssertionError('ranked must never take screenshots')
        return super().send(c)


def session(ledger, **kw):
    options = dict(name=ME, match_wait=20, rating_wait=1, timeout=1, settle_timeout=1)
    options.update(kw)
    return RankedSession(ledger, COND, **options)


def play(tmp_path, ledger, probe=None, **kw):
    probe = probe or RankedProbe(tmp_path)
    s = session(ledger, **kw)
    report = run(probe, advisor, NullJournal(), name=ME, games=1, mode='ranked', ranked=s,
                 settle_timeout=.01, timeout=.01)
    return probe, s, report


@pytest.fixture
def ledger(tmp_path):
    value = Ledger(tmp_path / 'state', budget=3)
    yield value
    value.close()


# ---- ledger ----------------------------------------------------------------

def test_ledger_allowlist_rejects_free_text_and_unknown_keys():
    base = dict(v=1, ev='reserve', game=1, t=1)
    validate_event(dict(base, **COND))
    for bad in (dict(name='x'), dict(stop_reason='Has Spaces'), dict(stop_reason='x' * 65),
                dict(outcome='won'), dict(model='has space'), dict(code_rev='not-hex!'),
                dict(rating_self_first=True), dict(rating_self_first='1500'), dict(decisions=-1),
                dict(first_mover='Alice'), dict(opp_kind='SyntheticRankedOpp')):
        with pytest.raises(ValueError):
            validate_event(dict(base, **bad))


def test_ledger_files_are_private_and_numeric(tmp_path):
    ledger = Ledger(tmp_path / 's', 2)
    game = ledger.reserve(COND)
    ledger.record(game, 'finish', terminal='completed', outcome='win', decisions=3)
    text = (tmp_path / 's/ledger.jsonl').read_text()
    assert oct((tmp_path / 's/ledger.jsonl').stat().st_mode & 0o777) == '0o600'
    assert oct((tmp_path / 's').stat().st_mode & 0o777) == '0o700'
    ledger.close()
    assert json.loads(text.splitlines()[0])['ev'] == 'reserve'


def test_reserve_budget_hard_limit_and_open_game(ledger):
    assert ledger.reserve(COND) == 1
    with pytest.raises(LedgerRefused, match='another_game_open'):
        ledger.reserve(COND)
    ledger.record(1, 'finish', terminal='completed', outcome='loss')
    assert ledger.reserve(COND) == 2
    ledger.record(2, 'finish', terminal='completed', outcome='win')
    assert ledger.reserve(COND) == 3
    ledger.record(3, 'finish', terminal='completed', outcome='draw')
    with pytest.raises(LedgerRefused, match='budget_exhausted'):
        ledger.reserve(COND)
    with pytest.raises(LedgerRefused):
        Ledger(ledger.directory / 'other', HARD_LIMIT + 1)
    assert ledger.summary()['wins'] == 1 and ledger.summary()['losses'] == 1 and ledger.summary()['draws'] == 1


def test_hard_limit_100_cannot_be_exceeded(tmp_path):
    ledger = Ledger(tmp_path / 's', HARD_LIMIT)
    for n in range(1, HARD_LIMIT + 1):
        assert ledger.reserve(COND) == n
        ledger.record(n, 'finish', terminal='completed', outcome='win')
    with pytest.raises(LedgerRefused, match='budget_exhausted'):
        ledger.reserve(COND)
    ledger.close()
    # A larger budget in a later process cannot reopen slots beyond the cap.
    again = Ledger(tmp_path / 's', HARD_LIMIT)
    with pytest.raises(LedgerRefused, match='budget_exhausted'):
        again.reserve(COND)
    again.close()


def write_preliminary(directory, *, shift=0, extra=()):
    """A ledger directory holding the two pinned never-applied preliminary games (as the real one did)."""
    from gfrl.ranked_ledger import PRELIMINARY_UNAPPLIED
    directory.mkdir(parents=True, mode=0o700)
    events = [dict(e) for g in sorted(PRELIMINARY_UNAPPLIED) for e in PRELIMINARY_UNAPPLIED[g]]
    events[0]['t'] += shift
    lines = [json.dumps(e, sort_keys=True) for e in events + list(extra)]
    (directory / 'ledger.jsonl').write_text('\n'.join(lines) + '\n')
    return directory


def test_preliminary_unapplied_games_do_not_count_toward_the_100_cap(tmp_path):
    from gfrl.ranked_ledger import GAME_NUMBER_MAX
    assert GAME_NUMBER_MAX == HARD_LIMIT + 2
    ledger = Ledger(write_preliminary(tmp_path / 's'), GAME_NUMBER_MAX)
    summary = ledger.summary()
    assert (summary['consumed'], summary['counted'], summary['exempt_unapplied']) == (2, 0, [1, 2])
    assert summary['unresolved'] == 2 and summary['unresolved_counted'] == 0
    assert summary['remaining'] == HARD_LIMIT
    last = None
    for expected in range(3, GAME_NUMBER_MAX + 1):
        last = ledger.reserve(COND, accept_unresolved=True)
        assert last == expected
        ledger.record(last, 'finish', terminal='completed', outcome='win')
    assert last == 102 and ledger.summary()['counted'] == HARD_LIMIT and ledger.summary()['remaining'] == 0
    with pytest.raises(LedgerRefused, match='budget_exhausted'):
        ledger.reserve(COND)
    ledger.close()
    # Also after a restart, and an undecided game would have used a counted slot too.
    again = Ledger(tmp_path / 's', GAME_NUMBER_MAX)
    with pytest.raises(LedgerRefused, match='budget_exhausted'):
        again.reserve(COND)
    again.close()


def test_old_budget_still_means_total_game_numbers_with_the_exemption(tmp_path):
    """--budget 12 (the running loop) still ends at ledger game 12 = 10 counted games."""
    ledger = Ledger(write_preliminary(tmp_path / 's'), 12)
    assert ledger.summary()['remaining'] == 10
    for n in range(3, 13):
        assert ledger.reserve(COND, accept_unresolved=True) == n
        ledger.record(n, 'finish', terminal='completed', outcome='loss')
    with pytest.raises(LedgerRefused, match='budget_exhausted'):
        ledger.reserve(COND)
    assert ledger.summary()['counted'] == 10
    ledger.close()


def test_undecided_games_count_and_only_the_two_pinned_games_are_exempt(tmp_path):
    ledger = Ledger(write_preliminary(tmp_path / 's'), 5)
    ledger.reserve(COND, accept_unresolved=True); ledger.record(3, 'queued')
    ledger.recover()  # a dead process: unresolved, still a counted slot
    summary = ledger.summary()
    assert summary['counted'] == 1 and summary['unresolved_counted'] == 1 and summary['exempt_unapplied'] == [1, 2]
    ledger.close()


@pytest.mark.parametrize('how,exempt', [('timestamp', [2]), ('extra_event', [1]), ('empty_directory', [])])
def test_the_exemption_cannot_be_reused_or_forged(tmp_path, how, exempt):
    from gfrl.ranked_ledger import GAME_NUMBER_MAX
    if how == 'timestamp':
        state = write_preliminary(tmp_path / 's', shift=1)            # same shape, other ledger
    elif how == 'extra_event':                                        # game 2 later started a match
        state = write_preliminary(tmp_path / 's', extra=[dict(v=1, ev='started', game=2, t=1795174100,
                                                              started_at=1795174100, seat_id=1, opp_kind='human')])
    else:
        state = tmp_path / 's'
    # Only verified games raise the ceiling: both exemptions are never granted, nor a budget beyond them.
    with pytest.raises(LedgerRefused, match='invalid_budget'):
        Ledger(state, GAME_NUMBER_MAX)
    with pytest.raises(LedgerRefused, match='invalid_budget'):
        Ledger(state, GAME_NUMBER_MAX + 1)
    ledger = Ledger(state, HARD_LIMIT + len(exempt))   # the failed opens above released the lock
    assert ledger.summary()['exempt_unapplied'] == exempt
    ledger.close()


def test_game_numbers_above_the_cap_are_rejected_by_the_schema():
    base = dict(v=1, ev='queued', t=1)
    for game in (HARD_LIMIT + 1, HARD_LIMIT + 2):
        validate_event(dict(base, game=game))
    for bad in (HARD_LIMIT + 3, 0):
        with pytest.raises(ValueError):
            validate_event(dict(base, game=bad))


def test_only_one_process_holds_the_ledger(tmp_path):
    first = Ledger(tmp_path / 's', 1)
    with pytest.raises(LedgerRefused, match='ledger_busy'):
        Ledger(tmp_path / 's', 1)
    first.close()
    Ledger(tmp_path / 's', 1).close()


def test_restart_marks_open_game_unresolved_and_keeps_slot(tmp_path):
    first = Ledger(tmp_path / 's', 3)
    first.reserve(COND); first.record(1, 'queued'); first.record(1, 'started', started_at=10, seat_id=1, opp_kind='human')
    first.close()  # Process died with the game open.
    second = Ledger(tmp_path / 's', 3)
    assert second.open_game()['game'] == 1
    second.recover()
    assert second.open_game() is None
    summary = second.summary()
    assert summary['consumed'] == 1 and summary['unresolved'] == 1 and summary['finished'] == 0
    # Not counted twice, and the unresolved game blocks the next one by default.
    with pytest.raises(LedgerRefused, match='previous_game_unresolved'):
        second.reserve(COND)
    assert second.reserve(COND, accept_unresolved=True) == 2
    assert second.summary()['consumed'] == 2
    second.close()


def test_finished_game_is_never_recovered_or_double_counted(tmp_path):
    first = Ledger(tmp_path / 's', 3)
    first.reserve(COND); first.record(1, 'finish', terminal='completed', outcome='win')
    first.close()
    second = Ledger(tmp_path / 's', 3)
    assert second.recover() is None
    assert second.summary()['consumed'] == 1 and second.summary()['wins'] == 1
    second.close()


def test_resume_reuses_open_game_without_new_slot(tmp_path):
    first = Ledger(tmp_path / 's', 3)
    first.reserve(COND); first.record(1, 'started', started_at=5, seat_id=0, opp_kind='human')
    first.close()
    second = Ledger(tmp_path / 's', 3)
    assert second.recover(resume=True) == 1
    row = second.games()[0]
    assert row['resumes'] == 1 and row['phase'] == 'started'
    assert second.summary()['consumed'] == 1
    second.close()
    third = Ledger(tmp_path / 's', 3)
    third.recover(resume=False)  # Died again: unresolved, still one slot.
    assert third.summary()['consumed'] == 1
    with pytest.raises(LedgerRefused, match='no_open_game_to_resume'):
        third.recover(resume=True)
    third.close()


def test_derive_ratings_only_when_grounded():
    assert derive_ratings(1500, 1508, 8, first_was_pregame=True) == dict(before=1500, after=1508, origin='wire_consistent')
    assert derive_ratings(1500, 1500, -9, first_was_pregame=True) == dict(before=1500, after=1491, origin='derived_from_gain')
    assert derive_ratings(1500, 1500, None, first_was_pregame=True) == dict(before=1500, after=None, origin='wire_before_only')
    assert derive_ratings(1500, 1503, 8, first_was_pregame=True) == dict(before=1500, after=None, origin='wire_before_only')
    assert derive_ratings(1500, 1508, 8, first_was_pregame=False)['origin'] == 'unknown'
    assert derive_ratings(None, None, None, first_was_pregame=False) == dict(before=None, after=None, origin='unknown')


# ---- parser / tap ----------------------------------------------------------

def test_strict_parser_still_rejects_duel_fields_but_ranked_parser_accepts(tmp_path):
    games = duel_games()
    capture = capture_file(tmp_path, games)
    with pytest.raises(Exception):
        parser_module().parse_prefix(Path(capture['file']).read_bytes(), 1)
    tap = RatingTap()
    parser, boundaries = named_room_parser(ME, [], require_no_end=False, tap=tap)
    parsed = parser.parse_prefix(Path(capture['file']).read_bytes(), boundary=0)
    assert [r['room_viewer'] for r in parsed['games']] == [1] * len(games)
    for record in parsed['games']:
        for p in record['game']['players']:
            assert 'alias' not in p and 'rating' not in p and 'ratingGain' not in p and 'team' not in p
        assert 'Synthetic' not in json.dumps(record['game'])
    # Integer ratings in every row: neither "absent = 0" flag may fire (they are for absent keys only).
    flags = dict(rating_absent=False, final_gain_absent=False)
    assert tap.seat_ratings(1) == dict(first=1500, last=1508, gain=8, pregame=True, **flags)
    assert tap.seat_ratings(2) == dict(first=1480, last=1472, gain=-8, pregame=True, **flags)


def test_team_games_and_bad_rating_types_are_rejected(tmp_path):
    for games in (duel_games(team=1),):
        capture = capture_file(tmp_path, games)
        parser, _ = named_room_parser(ME, [], require_no_end=False, tap=RatingTap())
        with pytest.raises(Exception):
            parser.parse_prefix(Path(capture['file']).read_bytes(), boundary=0)
    games = duel_games()
    games[0]['players'][0]['rating'] = 'high'
    capture = capture_file(tmp_path, games, serial=2)
    parser, _ = named_room_parser(ME, [], require_no_end=False, tap=RatingTap())
    with pytest.raises(Exception):
        parser.parse_prefix(Path(capture['file']).read_bytes(), boundary=0)


def test_tap_ignores_frames_before_the_baseline(tmp_path):
    games = duel_games()
    old = capture_file(tmp_path, games[:2])
    tap = RatingTap()
    parser, _ = named_room_parser(ME, [old], require_no_end=False, tap=tap)
    boundary = old['bytes']
    parser.parse_prefix(Path(old['file']).read_bytes(), boundary=boundary)
    assert tap.updates == {}


# ---- full run --------------------------------------------------------------

def test_ranked_game_records_allowlisted_result_with_one_click(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger)
    assert report['status'] == 'complete' and report['games'] == 1
    assert probe.clicks.count('2人個人戦') == 1
    # After the fsynced finish row, the end screen's own button is pressed exactly once; nothing else.
    assert probe.clicks == ['2人個人戦', '戦いを終わる'] and not any(c['kind'] == 'screenshot' for c in probe.commands)
    assert report['end_screen'] == 'left' and probe.phase == 'settings'
    row = ledger.games()[0]
    assert row['phase'] == 'finished' and row['terminal'] == 'completed'
    assert row['outcome'] in ('win', 'loss', 'draw')
    assert row['seat_id'] == 1 and row['opp_kind'] == 'human' and row['first_mover'] in ('self', 'opponent')
    assert row['decisions'] >= 1 and row['updates'] >= 1 and row['end_gf'] is None
    assert row['model'] == 'm-xit1' and row['policy'] == 'greedy' and row['search'] is False
    assert (row['rating_self_before'], row['rating_self_after'], row['rating_self_gain']) == (1500, 1508, 8)
    assert (row['rating_opp_before'], row['rating_opp_after']) == (1480, 1472)
    assert row['rating_origin'] == 'wire_consistent'
    assert row['ended_at'] >= row['started_at'] and row['duration_s'] >= 0
    assert ledger.summary()['consumed'] == 1


def test_reservation_is_written_before_the_matchmaking_click(tmp_path, room_clock, ledger):
    seen = []

    class Watching(RankedProbe):
        def send(self, c):
            if c['kind'] == 'clickText' and c['text'] == '2人個人戦':
                seen.append((ledger.events[-1]['ev'], (ledger.directory / 'ledger.jsonl').read_text().count('reserve')))
            return super().send(c)
    play(tmp_path, ledger, Watching(tmp_path))
    assert seen == [('reserve', 1)]


def test_ledger_logs_and_report_contain_no_names_or_alias(tmp_path, room_clock, ledger, capsys):
    play(tmp_path, ledger)
    blob = (ledger.directory / 'ledger.jsonl').read_text()
    for secret in (ME, OPP, 'SyntheticAlias', '2人個人戦'):
        assert secret not in blob
    assert set(os.listdir(ledger.directory)) == {'ledger.jsonl', 'ranked.lock'}
    out = capsys.readouterr()
    assert ME not in out.out + out.err and OPP not in out.out + out.err


def test_no_rating_in_wire_records_nulls(tmp_path, room_clock, ledger):
    games = duel_games()
    for g in games:
        for p in g['players']:
            p.pop('rating', None); p.pop('ratingGain', None)
    play(tmp_path, ledger, RankedProbe(tmp_path, games=games))
    row = ledger.games()[0]
    assert row['rating_self_before'] is None and row['rating_self_after'] is None
    assert row['rating_origin'] == 'unknown' and row['outcome'] in ('win', 'loss', 'draw')
    # Absent keys are raw 0 = display 1500 (official client); the gain only after the formal end.
    for who in ('self', 'opp'):
        assert (row[f'rating_{who}_display_before'], row[f'rating_{who}_display_before_src']) == (1500, 'wire_absent_zero')
        assert (row[f'rating_{who}_display_after'], row[f'rating_{who}_display_after_src']) == (1500, 'wire_absent_zero')


def test_game_with_end_time_is_played_and_recorded(tmp_path, room_clock, ledger):
    play(tmp_path, ledger, RankedProbe(tmp_path, games=duel_games(end=100)))
    assert ledger.games()[0]['end_gf'] == 100 and ledger.games()[0]['phase'] == 'finished'


def test_bot_opponent_kind_recorded(tmp_path, room_clock, ledger):
    games = duel_games()
    for g in games:
        g['players'][1]['isBot'] = True
    play(tmp_path, ledger, RankedProbe(tmp_path, games=games))
    assert ledger.games()[0]['opp_kind'] == 'bot'


@pytest.mark.parametrize('kwargs,reason,consumed', [
    (dict(button_count=2), 'ranked_button_not_unique', 0),
    (dict(button_count=0), 'ranked_button_not_unique', 0),
    (dict(lobby_body='修行\n隠れ乱闘\n真剣タイマン\nBOTの使用は禁止です'), 'bot_prohibition_notice', 0),
    (dict(in_game_on_load=True), 'ranked_game_in_progress_on_start', 0),
    (dict(birth=True), 'birth_not_allowed', 0),
])
def test_stops_before_reservation_never_consume_or_click(tmp_path, room_clock, ledger, kwargs, reason, consumed):
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, **kwargs))
    assert report['status'] == 'stopped' and report['reason'] == reason and report['detail'] == {}
    assert '2人個人戦' not in probe.clicks and ledger.summary()['consumed'] == consumed


def test_match_timeout_consumes_the_slot_without_second_click(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, never_match=True), match_wait=3)
    assert report['reason'] == 'match_wait_timeout'
    assert probe.clicks.count('2人個人戦') == 1
    row = ledger.games()[0]
    assert row['phase'] == 'stopped' and row['outcome'] == 'unknown' and row['stop_reason'] == 'match_wait_timeout'
    assert ledger.summary()['consumed'] == 1 and ledger.summary()['unresolved'] == 1


def test_stop_during_game_is_recorded_not_dropped(tmp_path, room_clock, ledger):
    class Breaking(RankedProbe):
        def send(self, c):
            if c['kind'] == 'clickPoint' and self.index >= 1:
                raise RuntimeError('SyntheticRankedMe leaked in an exception message')
            return super().send(c)
    probe, s, report = play(tmp_path, ledger, Breaking(tmp_path))
    assert report['status'] == 'stopped' and report['reason'] == 'bot_input_or_model_failure'
    assert 'SyntheticRankedMe' not in json.dumps(report)
    row = ledger.games()[0]
    assert row['phase'] == 'stopped' and row['outcome'] == 'unknown' and row['terminal'] == 'stopped'
    assert row['seat_id'] == 1 and ledger.summary()['consumed'] == 1
    assert ME not in (ledger.directory / 'ledger.jsonl').read_text()


def test_second_run_after_stop_is_refused_unless_accepted(tmp_path, room_clock, ledger):
    play(tmp_path, ledger, RankedProbe(tmp_path, never_match=True), match_wait=3)
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path))
    assert report['reason'] == 'previous_game_unresolved' and '2人個人戦' not in probe.clicks
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path), accept_unresolved=True)
    assert report['status'] == 'complete' and ledger.summary()['consumed'] == 2


def test_budget_is_enforced_across_runs(tmp_path, room_clock):
    ledger = Ledger(tmp_path / 's', 1)
    play(tmp_path, ledger)
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path))
    assert report['reason'] == 'budget_exhausted' and '2人個人戦' not in probe.clicks
    assert ledger.summary()['consumed'] == 1
    ledger.close()


# ---- intermediate screen after the ranked click (--step) ----------------------------

def test_intermediate_screen_without_step_waits_and_reports_only_vocabulary(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, step='戦いを始める'), match_wait=3)
    assert report['reason'] == 'match_wait_timeout'
    assert probe.clicks == ['2人個人戦']                      # nothing else is ever pressed
    # The synthetic screen reads "対戦の準備 / 戦いを始める": the report is the SCREEN_VOCAB projection,
    # in vocabulary order, so the substring 対戦 (added to the vocabulary 11-20) is listed too.
    assert report['detail'] == dict(screen=['戦いを始める', '対戦'], steps=[])
    assert ledger.summary()['consumed'] == 1 and ledger.summary()['unresolved'] == 1


def test_watch_screen_writes_only_fixed_vocabulary_and_clicks_nothing(tmp_path, room_clock, ledger):
    import json
    from gfrl.official_ranked import SCREEN_VOCAB
    target = tmp_path / 'screen-state.json'
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, step='戦いを始める'), match_wait=3,
                            screen_file=target)
    assert probe.clicks == ['2人個人戦'] and report['detail']['steps'] == []   # no forced --step
    data = json.loads(target.read_text())
    assert data['stage'] == 'wait_start' and '戦いを始める' in data['words']
    assert set(data['words']) <= set(SCREEN_VOCAB) and data['word_count'] == len(data['words'])
    assert oct(target.stat().st_mode & 0o777) == '0o600' and not (tmp_path / 'screen-state.json.tmp').exists()
    assert set(data) == {'stage', 'words', 'word_count', 'lobby', 'in_game', 'ranked_button_unique',
                         'polls', 'waited_s', 'steps_done'}
    assert len(SCREEN_VOCAB) == len(set(SCREEN_VOCAB))


def test_step_clicks_the_intermediate_button_once_without_a_second_slot(tmp_path, room_clock, ledger):
    seen = []

    class Watching(RankedProbe):
        def send(self, c):
            if c['kind'] == 'clickText':
                seen.append((c['text'], (ledger.directory / 'ledger.jsonl').read_text().count('"ev": "reserve"')))
            return super().send(c)
    probe, s, report = play(tmp_path, ledger, Watching(tmp_path, step='戦いを始める'), steps=('戦いを始める',))
    assert report['status'] == 'complete'
    assert probe.clicks == ['2人個人戦', '戦いを始める', '戦いを終わる']   # the last is the post-finish leave
    assert seen == [('2人個人戦', 1), ('戦いを始める', 1), ('戦いを終わる', 1)]     # one reservation, before any click
    assert ledger.summary()['consumed'] == 1 and ledger.games()[0]['phase'] == 'finished'
    assert s.steps_done == ['戦いを始める']


def test_step_is_not_repeated_when_the_screen_stays(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, step='戦いを始める', step_sticky=True),
                            match_wait=3, steps=('戦いを始める',))
    assert report['reason'] == 'match_wait_timeout'
    assert probe.clicks == ['2人個人戦', '戦いを始める']
    assert report['detail']['steps'] == ['戦いを始める'] and ledger.summary()['consumed'] == 1


def test_step_needs_exactly_one_visible_button(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, step='戦いを始める', step_count=2),
                            match_wait=3, steps=('戦いを始める',))
    assert report['reason'] == 'match_wait_timeout' and probe.clicks == ['2人個人戦']


def test_step_already_on_the_lobby_is_never_pressed(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, step='戦いを始める', step_in_lobby=True),
                            match_wait=3, steps=('戦いを始める',))
    assert report['reason'] == 'match_wait_timeout' and probe.clicks == ['2人個人戦']
    assert report['detail']['steps'] == []


def test_steps_follow_the_given_order_and_only_the_allowlist(tmp_path, room_clock, ledger):
    for bad in (('戦いをあきらめる',), ('キャンセル',), ('2人個人戦',), ('OK', 'OK')):
        with pytest.raises(UIUnsupported, match='invalid_step'):
            session(ledger, steps=bad)
    # The second configured step is not pressed while the first is not done.
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, step='OK'), match_wait=3,
                            steps=('戦いを始める', 'OK'))
    assert report['reason'] == 'match_wait_timeout' and probe.clicks == ['2人個人戦']


def test_no_step_click_before_the_game_is_reserved(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, button_count=2, step='戦いを始める'),
                            steps=('戦いを始める',))
    assert report['reason'] == 'ranked_button_not_unique' and report['detail'] == {}
    assert probe.clicks == [] and ledger.summary()['consumed'] == 0


def test_step_cli_rejects_unknown_buttons(capsys):
    with pytest.raises(SystemExit):
        main(['--status', '--step', '戦いをあきらめる'])
    capsys.readouterr()


def fake_profile(tmp_path, *, login=True):
    """A profile directory shaped like Chromium's; the record is a key name only (no credentials)."""
    directory = tmp_path / 'profile/Default/IndexedDB/https_godfield.net_0.indexeddb.leveldb'
    directory.mkdir(parents=True)
    (directory / '000010.log').write_bytes(b'\x00\x01' + (b'firebase:authUser:k:[DEFAULT]' if login else b'other') + b'\x02')
    return tmp_path / 'profile'


def test_entry_action_table():
    def act(**kw):
        base = dict(birth_screen=True, allow_birth=False, reenter=False, marker=False, login=False)
        return entry_action(**dict(base, **kw))
    assert act(birth_screen=False, reenter=True, marker=True, login=True) == 'none'
    assert act() == 'birth_not_allowed'
    assert act(allow_birth=True) == 'birth'
    assert act(marker=True) == 'reenter_not_allowed'
    assert act(marker=True, allow_birth=True) == 'birth_screen_with_existing_profile'
    assert act(marker=True, reenter=True) == 'reenter_login_missing'
    assert act(marker=True, reenter=True, login=True) == 'reenter'
    # --reenter alone never births: no marker means this profile has no account to re-enter.
    assert act(reenter=True, login=True) == 'birth_not_allowed'
    assert act(reenter=True, login=True, allow_birth=True) == 'birth'


def test_login_present_reads_only_key_presence(tmp_path):
    assert login_present(fake_profile(tmp_path / 'a')) is True
    assert login_present(fake_profile(tmp_path / 'b', login=False)) is False
    assert login_present(tmp_path / 'missing') is False


def test_reenter_clicks_birth_again_for_the_known_profile(tmp_path, room_clock, ledger):
    marker = tmp_path / 'born'
    marker.write_text('1\n')
    profile = fake_profile(tmp_path)
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, birth=True), marker=marker,
                            reenter=True, profile=profile)
    assert report['status'] == 'complete'
    assert probe.clicks[:2] == ['生誕する', '2人個人戦']
    fills = [c for c in probe.commands if c['kind'] == 'fillName']
    assert len(fills) == 1 and fills[0]['text'] == ME
    assert marker.read_text() == '1\n'


@pytest.mark.parametrize('kwargs,reason', [
    (dict(reenter=False), 'reenter_not_allowed'),
    (dict(reenter=True, login=False), 'reenter_login_missing'),
])
def test_known_profile_without_reenter_or_login_never_clicks(tmp_path, room_clock, ledger, kwargs, reason):
    marker = tmp_path / 'born'
    marker.write_text('1\n')
    profile = fake_profile(tmp_path, login=kwargs.get('login', True))
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, birth=True), marker=marker,
                            reenter=kwargs['reenter'], profile=profile)
    assert report['status'] == 'stopped' and report['reason'] == reason
    assert probe.clicks == [] and ledger.summary()['consumed'] == 0


def test_reenter_rejected_name_stops_before_any_reservation(tmp_path, room_clock, ledger):
    class Rejecting(RankedProbe):
        def send(self, c):
            if c['kind'] == 'clickText' and c['text'] == '生誕する':
                self.clicks.append(c['text']); return dict(state=self.state())  # Stays on the name screen.
            return super().send(c)
    marker = tmp_path / 'born'
    marker.write_text('1\n')
    probe, s, report = play(tmp_path, ledger, Rejecting(tmp_path, birth=True), marker=marker,
                            reenter=True, profile=fake_profile(tmp_path))
    assert report['reason'] == 'login_not_accepted'
    assert probe.clicks == ['生誕する'] and ledger.summary()['consumed'] == 0


def test_birth_requires_flag_and_is_refused_for_known_profile(tmp_path, room_clock, ledger):
    marker = tmp_path / 'born'
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, birth=True), allow_birth=True, marker=marker)
    assert report['status'] == 'complete' and probe.clicks[0] == '生誕する' and marker.exists()
    ledger2 = Ledger(tmp_path / 's2', 1)
    probe, s, report = play(tmp_path, ledger2, RankedProbe(tmp_path, birth=True), allow_birth=True, marker=marker)
    assert report['reason'] == 'birth_screen_with_existing_profile' and ledger2.summary()['consumed'] == 0
    ledger2.close()


def test_resume_reattaches_without_click_or_new_slot(tmp_path, room_clock):
    first = Ledger(tmp_path / 's', 2)
    first.reserve(COND); first.record(1, 'queued')
    first.close()
    ledger = Ledger(tmp_path / 's', 2)
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, in_game_on_load=True), resume=True)
    assert report['status'] == 'complete' and '2人個人戦' not in probe.clicks
    row = ledger.games()[0]
    assert ledger.summary()['consumed'] == 1 and row['resumes'] == 1 and row['phase'] == 'finished'
    ledger.close()


def test_resume_without_open_game_refuses(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, in_game_on_load=True), resume=True)
    assert report['reason'] == 'ranked_game_in_progress_on_start' and ledger.summary()['consumed'] == 0


def test_ranked_requires_session_and_single_game(tmp_path, ledger):
    probe = RankedProbe(tmp_path)
    for kwargs in (dict(ranked=None), dict(ranked=session(ledger), games=2)):
        kwargs.setdefault('games', 1)
        kwargs.setdefault('ranked', None)
        report = run(probe, advisor, NullJournal(), name=ME, mode='ranked', **kwargs)
        assert report['status'] == 'stopped' and report['reason'] == 'ranked_session_required'
    assert '2人個人戦' not in probe.clicks


def test_existing_modes_unchanged_by_tap_free_parser(tmp_path):
    games = deepcopy(GAMES)
    for g in games:
        for p in g['players']:
            p['isBot'] = False
            p['name'] = 'SyntheticName' if p['id'] == 1 else 'SyntheticOpponent'
    capture = capture_file(tmp_path, games)
    parser, _ = named_room_parser('SyntheticName', [])
    parsed = parser.parse_prefix(Path(capture['file']).read_bytes(), boundary=0)
    assert parsed['games']


# ---- helpers ---------------------------------------------------------------

def test_bot_notice_detection_is_specific():
    assert bot_notice('ボットの使用は禁止です') and bot_notice('禁止事項: BOT行為') and bot_notice('マクロ  禁止')
    assert not bot_notice('修行\n隠れ乱闘\n真剣タイマン\n2人個人戦') and not bot_notice('禁止 カード')


def test_lobby_and_button_detection():
    state = dict(body='修行\n隠れ乱闘\n真剣タイマン\n2人個人戦', texts=[dict(text='2人個人戦')])
    assert lobby_screen(state) and ranked_button_unique(state)
    assert not ranked_button_unique(dict(state, texts=[]))


def test_safe_reason_is_an_enum_like_code():
    assert safe_reason('match_wait_timeout') == 'match_wait_timeout'
    for bad in ('Has Name', 'ui_kind_unobserved:discard', None, 'x' * 70, 'Capital'):
        assert safe_reason(bad) == 'other'


def test_checkpoint_conditions_are_enums_and_hash(tmp_path):
    path = tmp_path / 'm-test.pt'; path.write_bytes(b'abc')
    cond = checkpoint_conditions(path)
    validate_event(dict(v=1, ev='reserve', game=1, t=1, **cond))
    assert cond['model'] == 'm-test' and cond['ckpt_sha8'] == 'ba7816bf'
    odd = tmp_path / 'weird name!.pt'; odd.write_bytes(b'x')
    assert checkpoint_conditions(odd)['model'] == 'custom'


def test_status_cli_prints_only_numbers(tmp_path, capsys):
    state = tmp_path / 'st'
    ledger = Ledger(state, 5); ledger.reserve(COND); ledger.record(1, 'finish', terminal='completed', outcome='win')
    ledger.close()
    assert main(['--status', '--state-dir', str(state), '--budget', '5']) == 0
    row = json.loads(capsys.readouterr().out)
    assert row['consumed'] == 1 and row['wins'] == 1 and row['remaining'] == 4


def test_cli_refuses_when_another_process_holds_the_ledger(tmp_path, capsys):
    state = tmp_path / 'st'
    held = Ledger(state, 1)
    assert main(['--status', '--state-dir', str(state)]) == 2
    assert capsys.readouterr().out.strip() == 'error:ledger_busy'
    held.close()


def test_cli_refuses_state_dir_inside_repo_that_is_not_ignored(capsys):
    from gfrl.official import ROOT
    target = ROOT / 'tests/python/_ranked_not_ignored_state'
    try:
        assert main(['--status', '--state-dir', str(target)]) == 2
        assert capsys.readouterr().out.strip() == 'error:state_dir_not_git_ignored'
    finally:
        if target.exists():
            target.rmdir()


# ---- CLI wiring (browser, model and run stubbed) ----------------------------

class DummyProbe:
    created = []

    def __init__(self, command, timeout=30, env=None, memory=False):
        self.command, self.env, self.memory = command, env, memory
        DummyProbe.created.append(self)

    def close(self):
        pass


@pytest.fixture
def cli(tmp_path, monkeypatch):
    import gfrl.evaluation as evaluation
    import gfrl.official_bot as bot
    from gfrl.official import ROOT
    DummyProbe.created.clear()
    checkpoint = tmp_path / 'm-fake.pt'; checkpoint.write_bytes(b'x')
    monkeypatch.setattr(bot, 'Probe', DummyProbe)
    monkeypatch.setattr(evaluation, 'load_policy_model', lambda *a, **k: object())
    state = tmp_path / 'state'
    argv = ['--checkpoint', str(checkpoint), '--playwright', str(tmp_path / 'pw'),
            '--state-dir', str(state), '--budget', '2']
    return bot, state, argv, monkeypatch


def test_cli_runs_with_private_profile_and_keeps_raw_data_in_memory(cli, capsys):
    bot, state, argv, monkeypatch = cli
    seen = {}
    monkeypatch.setenv('NODE_OPTIONS', '--trace-events-enabled')
    monkeypatch.setenv('DEBUG', 'pw:*')

    def fake_run(probe, advisor, journal, *, ranked, **kw):
        seen.update(env=probe.env, command=probe.command, memory=probe.memory,
                    journal=type(journal).__name__, mode=kw['mode'], tree=sorted(p.name for p in state.rglob('*')))
        import sys
        print(SECRET, file=sys.stderr)
        ranked.game = ranked.ledger.reserve(ranked.conditions)
        ranked.finished = True
        ranked.ledger.record(ranked.game, 'finish', terminal='completed', outcome='win')
        return dict(games=1, wins=1, losses=0, draws=0, status='complete', detail={})
    SECRET = 'SyntheticRawReceiveText'
    monkeypatch.setattr(bot, 'run', fake_run)
    assert main(argv) == 0
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert report['status'] == 'complete' and report['ledger']['consumed'] == 1
    assert SECRET not in out.out + out.err
    assert seen['env']['GFRL_PROFILE_DIR'] == str(state / 'profile') and seen['mode'] == 'ranked'
    assert seen['journal'] == 'NullJournal'
    assert oct((state / 'profile').stat().st_mode & 0o777) == '0o700'
    # Memory-only probe: no output directory is passed or created, nothing but the
    # profile and the ledger files exist in the state directory.
    assert seen['memory'] is True and seen['env']['GFRL_MEMORY_ONLY'] == '1'
    assert '--out-dir' not in seen['command']
    assert 'scratch' not in seen['tree'] and not (state / 'scratch').exists()
    assert 'NODE_OPTIONS' not in seen['env'] and seen['env']['DEBUG'] == ''
    Ledger(state, 2).close()  # The lock was released.


def test_cli_deletes_stale_scratch_from_a_crashed_run(cli, capsys):
    bot, state, argv, monkeypatch = cli
    stale = state / 'scratch/probe-old'
    stale.mkdir(parents=True)
    (stale / 'prefix.bin').write_bytes(b'stale raw receive')
    monkeypatch.setattr(bot, 'run', lambda *a, ranked, **k: dict(games=0, wins=0, losses=0, draws=0, status='complete', detail={}))
    assert main(argv) == 0
    assert not stale.exists()


def test_cli_exception_prints_only_a_code(cli, capsys):
    bot, state, argv, monkeypatch = cli

    def boom(*a, **k):
        raise RuntimeError('SyntheticRankedMe SyntheticSecretRoom')
    monkeypatch.setattr(bot, 'run', boom)
    assert main(argv) == 1
    out = capsys.readouterr()
    assert json.loads(out.out) == dict(status='stopped', reason='ranked_internal_error', detail={})
    assert 'Synthetic' not in out.out + out.err


def test_cli_interrupt_marks_open_game_stopped(cli, capsys):
    bot, state, argv, monkeypatch = cli

    def interrupted(probe, advisor, journal, *, ranked, **kw):
        ranked.game = ranked.ledger.reserve(ranked.conditions)
        raise KeyboardInterrupt
    monkeypatch.setattr(bot, 'run', interrupted)
    assert main(argv) == 1
    assert json.loads(capsys.readouterr().out) == dict(
        status='stopped', reason='interrupted', detail=dict(screen=[], steps=[]))
    ledger = Ledger(state, 2)
    row = ledger.games()[0]
    assert row['phase'] == 'stopped' and row['stop_reason'] == 'interrupted' and row['outcome'] == 'unknown'
    ledger.close()


def test_cli_discover_never_reserves_or_clicks_ranked(cli, capsys, room_clock, tmp_path):
    bot, state, argv, monkeypatch = cli
    lobby = RankedProbe(tmp_path, lobby_body='修行\n隠れ乱闘\n真剣タイマン\n2人個人戦\nレート 1500')
    monkeypatch.setattr(bot, 'Probe', lambda *a, **k: lobby)
    assert main(argv + ['--discover']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == dict(status='discover', birth_screen=False, entry='none',
                          screen=['2人個人戦', '真剣タイマン', '隠れ乱闘', '修行', 'レート'], born_marker=False,
                          profile_login_present=False, lobby=True, in_game=False,
                          ranked_button_unique=True, bot_notice=False, rating_label_seen=True,
                          rating_candidate=1500)
    assert '2人個人戦' not in lobby.clicks
    ledger = Ledger(state, 2)
    assert ledger.summary()['consumed'] == 0
    ledger.close()


# ---- display rating on the screens vs raw wire values ---------------------------------------

def _texts(*items):
    return [dict(text=t, x=0, y=0, w=1, h=1) for t in items]


def test_screen_result_reads_digits_only_and_refuses_ambiguity():
    assert screen_result(dict(body='真剣タイマン\nレーティング 1516 (+16)', texts=[])) == dict(rating=1516, gain=16)
    assert screen_result(dict(body='レーティング: 1484（−16）', texts=[])) == dict(rating=1484, gain=-16)
    assert screen_result(dict(body='', texts=_texts('レーティング', '1516', '+16'))) == dict(rating=1516, gain=16)
    assert screen_result(dict(body='レーティング 1516', texts=[])) == dict(rating=1516, gain=None)
    # An unsigned number is never read as a gain; two different ratings or gains give None.
    assert screen_result(dict(body='レーティング 1516 16', texts=[])) == dict(rating=1516, gain=None)
    assert screen_result(dict(body='レーティング 1516\nレーティング 1500', texts=[]))['rating'] is None
    assert screen_result(dict(body='レーティング 1516 +16\nレーティング 1516 -16', texts=[]))['gain'] is None
    assert screen_result(dict(body='勝ち', texts=[])) == dict(rating=None, gain=None)
    assert screen_rating(dict(body='レーティング\n1516', texts=[])) == 1516


def test_wire_display_offset_is_the_official_1500_and_never_applied_to_a_non_integer():
    assert WIRE_DISPLAY_OFFSET == 1500   # official client: display = 1500 + raw (scripts/ranked_bot.md)
    assert wire_to_display(32, WIRE_DISPLAY_OFFSET) == 1532 and wire_to_display(None, 1500) is None
    assert wire_to_display(True, 1500) is None and wire_to_display(32, None) is None


def seat(*rows):
    """RatingTap seat summary for one seat: rows of (over, rating_key, gain_key, rating, gain)."""
    tap = RatingTap()
    for n, (over, rk, gk, rating, gain) in enumerate(rows):
        tap.add(dict(update=n, over=over, end_gf=None, players={
            1: dict(rating=rating, gain=gain, rating_key=rk, gain_key=gk, bot=False)}))
    return tap.seat_ratings(1)


def display(summary, offset=1500):
    return seat_display(summary, derive_ratings(summary['first'], summary['last'], summary['gain'],
                                                first_was_pregame=summary['pregame']), offset)


def test_tap_tells_an_absent_key_from_an_invalid_or_unobserved_one():
    row = RatingTap.extract(dict(updateCount=1, isOver=True, players=[
        dict(id=1), dict(id=2, rating=None, ratingGain='x'), dict(id=3, rating=7, ratingGain=-7)]))['players']
    assert (row[1]['rating_key'], row[1]['gain_key']) == ('absent', 'absent')
    assert (row[2]['rating_key'], row[2]['gain_key']) == ('invalid', 'invalid')
    assert (row[3]['rating_key'], row[3]['gain_key'], row[3]['rating'], row[3]['gain']) == ('int', 'int', 7, -7)
    assert RatingTap().seat_ratings(9)['rating_absent'] is False       # an unobserved seat is not "absent"


def test_display_rating_absent_means_zero_and_the_gain_only_after_the_formal_end():
    absent = ('absent', 'absent', None, None)
    live = display(seat((False, *absent), (False, *absent)))
    assert (live['before'], live['before_src'], live['after'], live['after_src']) == (1500, 'wire_absent_zero', None, 'unknown')
    ended = display(seat((False, *absent), (True, *absent)))
    assert (ended['after'], ended['after_src']) == (1500, 'wire_absent_zero')
    # Wrong-typed or mixed observations are never read as "absent = 0"; no offset gives nothing.
    assert display(seat((False, 'invalid', 'absent', None, None), (True, 'invalid', 'absent', None, None)))['before'] is None
    assert display(seat((False, 'absent', 'absent', None, None), (True, 'int', 'absent', 5, None)))['before'] is None
    assert display(seat((False, *absent), (True, 'absent', 'invalid', None, None)))['after'] is None
    assert display(seat((False, *absent), (True, *absent)), None) == dict(
        before=None, after=None, before_src='unknown', after_src='unknown')


def test_display_rating_from_integer_raw_values_adds_the_offset():
    won = display(seat((False, 'int', 'absent', 32, None), (True, 'int', 'int', 16, -16)))
    assert (won['before'], won['after'], won['before_src'], won['after_src']) == (1532, 1516, 'wire', 'wire')
    # A live game that ends without a gain on an integer rating: "absent = 0" -> unchanged.
    flat = display(seat((False, 'int', 'absent', 32, None), (True, 'int', 'absent', 32, None)))
    assert (flat['before'], flat['after'], flat['after_src']) == (1532, 1532, 'wire_absent_zero')


def test_screen_check_and_zero_sum_cross_check():
    assert screen_check(1500, 1516, 16) == 'consistent' and screen_check(1500, 1516, 15) == 'inconsistent'
    assert screen_check(1500, 1516, None) == 'delta_only'
    assert screen_check(1500, 1500, None) == 'unavailable' and screen_check(None, 1516, 16) == 'unavailable'
    assert gain_cross(1500, 1516, 16, -16) == 'zero_sum' and gain_cross(1500, 1516, None, -16) == 'zero_sum'
    assert gain_cross(1500, 1516, 16, -8) == 'not_zero_sum' and gain_cross(1500, 1516, 16, None) == 'unknown'
    assert gain_cross(1500, 1500, None, -16) == 'unknown'


def test_ledger_rejects_bad_display_rating_fields():
    base = dict(v=1, ev='finish', game=1, t=1)
    validate_event(dict(base, rating_screen_after=1516, rating_screen_gain=-16, rating_screen_check='consistent',
                        rating_gain_cross='zero_sum', rating_opp_display_before=None,
                        rating_display_basis='unconverted'))
    for bad in (dict(rating_screen_check='ok'), dict(rating_gain_cross=True), dict(rating_display_basis='wire'),
                dict(rating_screen_after='1516'), dict(rating_opp_display_after=1.5),
                dict(rating_self_display_after_src='guess'), dict(rating_opp_display_before_src=None),
                dict(rating_self_display_before=1500.0)):
        with pytest.raises(ValueError):
            validate_event(dict(base, **bad))


class FightProbe(RankedProbe):
    """The application screen: 真剣タイマン / レーティング <n> / 戦う, the number loading later;
    optionally a result screen with the display rating after the game."""
    def __init__(self, directory, *, loaded_after=1, shown=1500, result=None, flicker=False, **kw):
        super().__init__(directory, step='戦う', **kw)
        self.loaded_after, self.shown, self.result, self.flicker = loaded_after, shown, result, flicker
        self.fight_polls = 0

    def state(self):
        if self.phase == 'step':
            self.fight_polls += 1
            loaded = self.fight_polls > self.loaded_after
            number = str(self.shown + (self.fight_polls % 2 if self.flicker else 0)) if loaded else ''
            texts = _texts('戦う') + (_texts('レーティング', number) if loaded else [])
            return dict(body=f'真剣タイマン\nレーティング\n{number}\n戦う', inputs=[], images=[], texts=texts)
        state = super().state()
        if self.phase == 'battle' and self.result and '戦いを終わる' in state['body']:
            state['body'] += f'\nレーティング {self.result[0]} ({self.result[1]:+d})'
        return state


def test_fight_waits_for_the_screen_rating_and_records_the_start_rating(tmp_path, room_clock, ledger):
    probe = FightProbe(tmp_path, loaded_after=4)
    probe, s, report = play(tmp_path, ledger, probe, steps=('戦う',))
    assert report['status'] == 'complete' and probe.clicks == ['2人個人戦', '戦う', '戦いを終わる']
    assert probe.fight_polls >= 6            # not pressed on the unloaded polls nor on the first read
    row = ledger.games()[0]
    assert row['rating_screen'] == 1500 and ledger.summary()['consumed'] == 1
    assert row['rating_screen_check'] == 'unavailable' and 'rating_screen_after' not in row


def test_fight_is_never_pressed_when_the_screen_rating_does_not_load(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger, FightProbe(tmp_path, loaded_after=10 ** 6), steps=('戦う',),
                            rating_screen_wait=3)
    assert report['reason'] == 'rating_screen_not_loaded' and probe.clicks == ['2人個人戦']
    row = ledger.games()[0]
    assert row['phase'] == 'stopped' and row['stop_reason'] == 'rating_screen_not_loaded'
    assert 'rating_screen' not in row and ledger.summary()['consumed'] == 1


def test_fight_needs_the_same_rating_twice_in_a_row(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger, FightProbe(tmp_path, flicker=True), steps=('戦う',),
                            rating_screen_wait=3)
    assert report['reason'] == 'rating_screen_not_loaded' and probe.clicks == ['2人個人戦']


def test_result_screen_is_cross_checked_with_the_screen_before_and_the_wire(tmp_path, room_clock, ledger):
    probe, s, report = play(tmp_path, ledger, FightProbe(tmp_path, result=(1508, 8)), steps=('戦う',))
    row = ledger.games()[0]
    assert (row['rating_screen'], row['rating_screen_after'], row['rating_screen_gain']) == (1500, 1508, 8)
    assert row['rating_screen_check'] == 'consistent' and row['rating_gain_cross'] == 'zero_sum'
    assert row['rating_display_basis'] == 'offset_verified'
    assert (row['rating_opp_display_before'], row['rating_opp_display_after']) == (2980, 2972)  # synthetic raw 1480/1472
    assert (row['rating_self_display_before'], row['rating_self_display_before_src']) == (3000, 'wire')
    assert (row['rating_self_display_after'], row['rating_self_display_after_src']) == (1508, 'screen')


def test_first_real_game_shape_display_values_and_provenance(tmp_path, room_clock, ledger):
    """Game 3: our raw rating fields all missing; opponent raw 32 / gain -16; screen 1500 -> 1516."""
    games = duel_games(ratings=(1500, 32), over_gain=False)
    for g in games:
        for p in g['players']:
            if p['id'] == 1:
                p.pop('rating', None); p.pop('ratingGain', None)
    games[-1]['players'][1]['ratingGain'] = -16
    probe, s, report = play(tmp_path, ledger, FightProbe(tmp_path, games=games, result=(1516, 16)), steps=('戦う',))
    row = ledger.games()[0]
    assert report['status'] == 'complete'
    assert row['rating_self_first'] is None and row['rating_self_gain'] is None and row['rating_self_before'] is None
    assert (row['rating_opp_first'], row['rating_opp_gain'], row['rating_opp_before'], row['rating_opp_after']) \
        == (32, -16, 32, 16)                                       # raw wire values, as received
    assert (row['rating_opp_display_before'], row['rating_opp_display_after']) == (1532, 1516)
    assert (row['rating_opp_display_before_src'], row['rating_opp_display_after_src']) == ('wire', 'wire')
    assert row['rating_display_basis'] == 'offset_verified'
    # Ours: absent = raw 0 = 1500 before; after is the official screen (the wire's absent gain would say 1500).
    assert (row['rating_self_display_before'], row['rating_self_display_before_src']) == (1500, 'wire_absent_zero')
    assert (row['rating_self_display_after'], row['rating_self_display_after_src']) == (1516, 'screen')
    assert (row['rating_screen'], row['rating_screen_after'], row['rating_screen_gain']) == (1500, 1516, 16)
    assert row['rating_screen_check'] == 'consistent' and row['rating_gain_cross'] == 'zero_sum'


def test_without_an_offset_nothing_is_converted(tmp_path, room_clock, ledger):
    games = duel_games(ratings=(1500, 32), over_gain=False)
    games[-1]['players'][1]['ratingGain'] = -16
    probe, s, report = play(tmp_path, ledger, FightProbe(tmp_path, games=games), steps=('戦う',), wire_offset=None)
    row = ledger.games()[0]
    assert (row['rating_opp_before'], row['rating_opp_display_before'], row['rating_opp_display_after']) \
        == (32, None, None) and row['rating_display_basis'] == 'unconverted'
    assert row['rating_opp_display_before_src'] == 'unknown' and row['rating_self_display_before'] == 1500
    assert row['rating_self_display_before_src'] == 'screen'


def test_cli_rejects_a_bad_rating_screen_wait(capsys):
    assert main(['--status', '--rating-screen-wait', '0']) == 2
    assert capsys.readouterr().out.strip() == 'error:invalid_arguments'


# ---- leaving the end screen (after the result is recorded) -----------------

class EndProbe(RankedProbe):
    """mode: 'absent' (no button ever), 'twice' (button shown twice), 'stuck' (click does nothing),
    'raises' (state fails after the result), 'normal'."""
    def __init__(self, directory, mode='normal', **kw):
        super().__init__(directory, **kw)
        self.mode, self.ended = mode, False

    def state(self):
        s = super().state()
        if self.phase == 'battle' and '戦いを終わる' in s['body']:
            if self.mode == 'absent':
                s['body'] = s['body'].replace('戦いを終わる', '')
            elif self.mode == 'twice':
                s['body'] += '\n戦いを終わる'
            if self.mode == 'raises' and self.ended:
                raise RuntimeError('page closed')
            self.ended = True
        return s

    def send(self, c):
        if self.mode == 'stuck' and c['kind'] == 'clickText' and c['text'] == '戦いを終わる':
            self.clicks.append(c['text']); self.commands.append(c)
            return dict(state=self.state())
        return super().send(c)


@pytest.mark.parametrize('mode, expected, clicks', [
    ('absent', 'button_absent', 0), ('twice', 'button_ambiguous', 0),
    ('stuck', 'left_unconfirmed', 1)])
def test_end_screen_failure_never_changes_the_recorded_result(tmp_path, room_clock, ledger, mode, expected, clicks):
    probe, s, report = play(tmp_path, ledger, EndProbe(tmp_path, mode))
    assert report['status'] == 'complete' and report['end_screen'] == expected
    assert probe.clicks.count('戦いを終わる') == clicks      # at most once, never retried
    assert set(probe.clicks) <= {'2人個人戦', '戦いを終わる'}   # never 戦う / あきらめる etc.
    row = ledger.games()[0]
    assert row['phase'] == 'finished' and row['terminal'] == 'completed'
    assert row['outcome'] in ('win', 'loss', 'draw')


def test_leave_end_screen_exception_is_swallowed(tmp_path, room_clock, ledger):
    s = session(ledger)
    s.finished, s.game = True, 1
    class Boom:
        def send(self, c):
            raise RuntimeError('page closed')
    assert s.leave_end_screen(Boom()) == 'leave_failed'


def test_leave_end_screen_needs_a_recorded_finish(tmp_path, room_clock, ledger):
    s = session(ledger)
    probe = EndProbe(tmp_path)
    assert s.leave_end_screen(probe) == 'not_finished' and probe.clicks == []


# ---- allowlisted unsupported_receive subcodes ------------------------------

def test_ranked_receive_reason_allowlist_and_sanitization():
    from gfrl.official_ranked import (CONFLICT_PATH_RULES, EXC_TYPE_CODES, RECEIVE_REASON_CODES,
                                      SCHEMA_PATH_CODES, ranked_receive_reason, ranked_stop_code)
    from gfrl.ranked_ledger import SNAKE
    for value in (*RECEIVE_REASON_CODES.values(), *SCHEMA_PATH_CODES.values(), *EXC_TYPE_CODES.values(),
                  *(code for _, code in CONFLICT_PATH_RULES)):
        assert SNAKE.fullmatch(value), value
    # Session list details (skipping non-fatal multiple_pending_frames prefix).
    assert ranked_receive_reason([
        dict(code='multiple_pending_frames', count=2),
        dict(code='earth_moon_payload', phase=1),
    ]) == 'ur_earth_moon_payload'
    assert ranked_receive_reason([dict(code='confused_main', phase=0)]) == 'ur_confused_main'
    assert ranked_receive_reason([dict(code='unknown_event', action='secretName')]) == 'ur_unknown_event'
    # Parser schema details only emit static path buckets, never unknown key names or values.
    assert ranked_receive_reason(dict(
        error='ParseError', code='unsupported_game_schema',
        parser=dict(path='game.players[].secretField', unknown_keys=['secretField']),
    )) == 'ur_schema_players'
    assert ranked_receive_reason(dict(
        error='ParseError', code='unsupported_game_schema',
        parser=dict(path='game.events[]', action='mysteryAction'),
    )) == 'ur_schema_events'
    assert ranked_receive_reason(dict(
        error='ParseError', code='unsupported_game_schema',
        parser=dict(path='game.attacks[].hitRate'),
    )) == 'ur_schema_hitrate'
    assert ranked_receive_reason(dict(
        error='ParseError', code='unsupported_game_schema',
        parser=dict(path='totally.unknown.secret'),
    )) == 'ur_schema_other'
    # ValueError boundary codes and unexpected exception types.
    assert ranked_receive_reason(dict(error='ValueError', code='conflicting_update')) == 'ur_conflicting_update'
    assert ranked_receive_reason(dict(
        error='ValueError', code='conflicting_update', update=12,
        differences=[dict(path='phase', kind='changed'), dict(path='is_over', kind='changed')],
    )) == 'ur_cu_is_over'
    assert ranked_receive_reason(dict(
        error='ValueError', code='conflicting_update', update=12,
        differences=[dict(path='events[0].damage', kind='changed')],
    )) == 'ur_cu_events'
    assert ranked_receive_reason(dict(
        error='ValueError', code='conflicting_update', update=12,
        differences=[dict(path='players[1].status.hp', kind='changed')],
    )) == 'ur_cu_players_status'
    assert ranked_receive_reason(dict(
        error='ValueError', code='conflicting_update', update=12,
        differences=[dict(path='players[1].dead', kind='changed')],
    )) == 'ur_cu_players_dead'
    assert ranked_receive_reason(dict(
        error='ValueError', code='conflicting_update', update=12,
        differences=[dict(path='secretField', kind='changed')],
    )) == 'ur_conflicting_update'
    assert ranked_receive_reason(dict(error='KeyError', file='official.py', line=1)) == 'ur_exc_keyerror'
    assert ranked_receive_reason(dict(error='WeirdSecretError')) == 'ur_exc_other'
    # Unknown / free-text / adversarial payloads never leak.
    for bad in (
        None, '', 'PlayerName', dict(code='PlayerSecretName'),
        [dict(code='not_in_allowlist')],
    ):
        assert ranked_receive_reason(bad) == 'unsupported_receive'
    assert ranked_receive_reason(dict(code='unsupported_game_schema', parser='bad')) == 'ur_schema_other'
    # Non-receive UIUnsupported codes remain unchanged; non-UIUnsupported uses default.
    assert ranked_stop_code(UIUnsupported('visible_gf_changed')) == 'visible_gf_changed'
    assert ranked_stop_code(UIUnsupported('unsupported_receive', dict(
        error='ParseError', code='unsupported_game_schema', parser=dict(path='game.players[].x'),
    ))) == 'ur_schema_players'
    assert ranked_stop_code(RuntimeError('secret name in traceback')) == 'bot_input_or_model_failure'


def test_ranked_run_records_allowlisted_receive_subcode_in_ledger(tmp_path, room_clock, ledger):
    games = duel_games()
    games[-1]['players'][1]['unexpectedDuelField'] = 123
    probe, s, report = play(tmp_path, ledger, RankedProbe(tmp_path, games=games))
    assert report['status'] == 'stopped' and report['reason'] == 'ur_schema_players'
    assert 'unexpectedDuelField' not in json.dumps(report)
    row = ledger.games()[0]
    assert row['phase'] == 'stopped' and row['stop_reason'] == 'ur_schema_players'
    assert 'unexpectedDuelField' not in (ledger.directory / 'ledger.jsonl').read_text()
