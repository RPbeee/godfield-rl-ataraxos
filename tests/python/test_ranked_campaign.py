"""Offline tests for the read-only campaign report. The ledger lines are the real numeric ones
(games 1-5 of runs/ranked100 on 2026-11-20); nothing here names a person, room or opponent."""
import json
import os
from pathlib import Path

import pytest

from gfrl.ranked_campaign import (build_report, main, read_events, read_supplement, report_from_path, wilson)
from gfrl.ranked_ledger import PRELIMINARY_UNAPPLIED, HARD_LIMIT

COND = dict(ckpt_sha8='f2bd3ea3', code_rev='39bfa212', dream='allow', model='m-xit1', policy='greedy', search=False)


def preliminary():
    return [dict(e) for g in sorted(PRELIMINARY_UNAPPLIED) for e in PRELIMINARY_UNAPPLIED[g]]


def real_games(*, with_game5=True):
    """Games 3, 4 (old format: raw wire values only) and 5 (pre-match screen 1501, still running)."""
    e = [
        dict(v=1, ev='reserve', game=3, t=1795174045, budget=3, **COND),
        dict(v=1, ev='queued', game=3, t=1795174045),
        dict(v=1, ev='started', game=3, t=1795174045, end_gf=150, opp_kind='human', seat_id=1, started_at=1795174045),
        dict(v=1, ev='finish', game=3, t=1795174160, decisions=18, duration_s=115, ended_at=1795174160,
             first_mover='self', gf_final=23, outcome='win', rating_opp_after=16, rating_opp_before=32,
             rating_opp_first=32, rating_opp_gain=-16, rating_opp_last=32, rating_origin='derived_from_gain',
             rating_self_after=None, rating_self_before=None, rating_self_first=None, rating_self_gain=None,
             rating_self_last=None, terminal='completed', updates=37),
        dict(v=1, ev='reserve', game=4, t=1795174363, budget=10, **COND),
        dict(v=1, ev='queued', game=4, t=1795174363),
        dict(v=1, ev='started', game=4, t=1795174364, end_gf=150, opp_kind='human', seat_id=1, started_at=1795174364),
        dict(v=1, ev='finish', game=4, t=1795174448, decisions=14, duration_s=84, ended_at=1795174448,
             first_mover='self', gf_final=16, outcome='loss', rating_opp_after=154, rating_opp_before=139,
             rating_opp_first=139, rating_opp_gain=15, rating_opp_last=139, rating_origin='derived_from_gain',
             rating_self_after=1, rating_self_before=16, rating_self_first=16, rating_self_gain=-15,
             rating_self_last=16, terminal='completed', updates=29),
    ]
    if with_game5:
        e += [
            dict(v=1, ev='reserve', game=5, t=1795174784, budget=12, **COND),
            dict(v=1, ev='queued', game=5, t=1795174784),
            dict(v=1, ev='queued', game=5, t=1795174785, rating_screen=1501),
            dict(v=1, ev='started', game=5, t=1795174785, end_gf=150, opp_kind='human', rating_screen=1501,
                 seat_id=1, started_at=1795174785),
        ]
    return e


SUPPLEMENT = dict(correct_values=dict(self=dict(start_display=1500, end_display=1516, gain=16),
                                      opponent=dict(start_display=1532, end_display=1516, gain=-16)))


def write_ledger(directory, events, tail=b''):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'ledger.jsonl'
    path.write_bytes(''.join(json.dumps(e, sort_keys=True) + '\n' for e in events).encode() + tail)
    return path


def supplement_file(tmp_path, data=SUPPLEMENT):
    path = tmp_path / 'sup.json'
    path.write_text(json.dumps(data))
    return path


def test_wilson_matches_known_values():
    one_of_two = wilson(1, 2)
    assert one_of_two['rate'] == 0.5 and one_of_two['lo'] == pytest.approx(0.0945, abs=1e-4) \
        and one_of_two['hi'] == pytest.approx(0.9055, abs=1e-4)
    assert wilson(0, 0) == dict(k=0, n=0, rate=None, lo=None, hi=None)
    edge = wilson(10, 10)
    assert edge['hi'] == 1.0 and edge['lo'] == pytest.approx(0.7225, abs=1e-3)
    assert wilson(0, 10)['lo'] == 0.0


def test_preliminary_games_are_a_separate_tally_and_never_counted(tmp_path):
    path = write_ledger(tmp_path / 'l', preliminary() + real_games())
    report = report_from_path(path, supplement_file(tmp_path))
    assert report['preliminary_unapplied']['games'] == [1, 2] and report['preliminary_unapplied']['count'] == 2
    assert report['preliminary_unapplied']['decisions'] == 0
    assert report['preliminary_unapplied']['basis'] == 'pinned_ledger_events'
    counts = report['counts']
    assert (counts['counted'], counts['participated'], counts['completed']) == (3, 3, 2)
    assert (counts['wins'], counts['losses'], counts['draws'], counts['in_progress']) == (1, 1, 0, 1)
    assert report['cap']['counted'] == 3 and report['cap']['remaining'] == HARD_LIMIT - 3
    assert report['cap']['game_numbers'] == 5 and report['cap']['budgets_recorded'] == [1, 2, 3, 10, 12]
    assert [g['klass'] for g in report['games']] == ['preliminary_unapplied'] * 2 + ['completed'] * 2 + ['in_progress']
    assert 'game_in_progress' in report['warnings'] and 'preliminary_games_not_verified' not in report['warnings']
    assert report['by_opp_kind']['human']['participated'] == 3


def test_unverified_preliminary_games_count_like_any_other(tmp_path):
    forged = preliminary()
    forged[0]['t'] += 1
    report = report_from_path(write_ledger(tmp_path / 'l', forged + real_games(with_game5=False)))
    assert report['preliminary_unapplied']['games'] == [2]          # only the intact one
    assert report['counts']['counted'] == 3 and report['counts']['undecided_unmatched'] == 1   # games 1, 3, 4
    assert 'preliminary_games_not_verified' in report['warnings']


def test_win_rate_intervals_and_undecided_bounds(tmp_path):
    events = real_games(with_game5=False)
    events += [dict(v=1, ev='reserve', game=5, t=10, budget=12, **COND),
               dict(v=1, ev='started', game=5, t=11, started_at=11, seat_id=0, opp_kind='human'),
               dict(v=1, ev='stop', game=5, t=12, terminal='stopped', outcome='unknown', stop_reason='interrupted',
                    ended_at=12, decisions=3, updates=4, duration_s=1)]
    report = build_report(events)       # games 1-2 absent: nothing is exempt, nothing breaks
    rate = report['win_rate']
    assert rate['completed']['k'] == 1 and rate['completed']['n'] == 2
    assert rate['decided_only']['n'] == 2
    assert rate['unknown_as_loss']['n'] == 3 and rate['unknown_as_loss']['k'] == 1
    assert rate['unknown_as_win']['k'] == 2 and rate['unknown_as_win']['n'] == 3
    assert report['counts']['undecided_after_start'] == 1 and report['counts']['participated'] == 3
    assert report['by_opp_kind']['human'] == dict(participated=3, wins=1, losses=1, draws=0, undecided=1,
                                                  in_progress=0, win_rate_completed=wilson(1, 2))


def test_old_format_ratings_use_wire_plus_1500_the_supplement_and_the_next_screen(tmp_path):
    report = report_from_path(write_ledger(tmp_path / 'l', preliminary() + real_games()), supplement_file(tmp_path))
    g3, g4, g5 = (g['rating'] for g in report['games'][2:])
    assert g3['format'] == g4['format'] == 'legacy'
    # Game 3: our raw values are all missing -> the supplement; the opponent is raw + 1500.
    assert (g3['self_before']['value'], g3['self_before']['src']) == (1500, 'supplement')
    assert (g3['self_after']['value'], g3['self_after']['src']) == (1516, 'supplement')
    assert [c['src'] for c in g3['self_after']['candidates']] == ['supplement', 'next_wire_before']
    assert (g3['opp_before']['value'], g3['opp_before']['src'], g3['opp_after']['value']) == (1532, 'wire_offset', 1516)
    assert (g3['self_delta'], g3['opp_delta']) == (16, -16)
    # Game 4: the next game's official pre-match screen beats the wire value; both agree.
    assert (g4['self_before']['value'], g4['self_before']['src']) == (1516, 'wire_offset')
    assert (g4['self_after']['value'], g4['self_after']['src']) == (1501, 'next_prematch_screen')
    assert [c['src'] for c in g4['self_after']['candidates']] == ['next_prematch_screen', 'wire_offset']
    assert (g4['opp_before']['value'], g4['opp_after']['value'], g4['self_delta'], g4['opp_delta']) == (1639, 1654, -15, 15)
    # Game 5 is running: only the screen before it, which also chains from game 4.
    assert g5['self_before']['value'] == 1501 and g5['self_before']['src'] == 'prematch_screen'
    assert g5['self_after']['value'] is None and g5['self_delta'] is None
    assert report['rating']['mismatches'] == [] and 'rating_mismatch' not in report['warnings']
    assert report['rating']['self'] == dict(first_before=1500, last_after=1501, net_change=1, sum_of_known_deltas=1,
                                            games_with_unknown_after=1, min_after=1501, max_after=1516)


def test_without_the_supplement_game_3_self_ratings_stay_unknown(tmp_path):
    report = report_from_path(write_ledger(tmp_path / 'l', preliminary() + real_games()))
    g3 = report['games'][2]['rating']
    assert g3['self_before']['value'] is None and g3['self_before']['src'] == 'unknown'
    assert g3['self_after']['value'] == 1516 and g3['self_after']['src'] == 'next_wire_before'   # weakest source
    assert g3['self_delta'] is None


def test_disagreements_are_listed_explicitly(tmp_path):
    bad = dict(correct_values=dict(self=dict(start_display=1500, end_display=1520, gain=20),
                                   opponent=dict(start_display=1532, end_display=1516, gain=-16)))
    report = report_from_path(write_ledger(tmp_path / 'l', preliminary() + real_games()), supplement_file(tmp_path, bad))
    kinds = {(m['game'], m['item'], m['kind']) for m in report['rating']['mismatches']}
    assert (3, 'self_after', 'value_disagreement') in kinds       # supplement 1520 vs game 4's wire 1516
    assert (3, 'deltas', 'not_zero_sum') in kinds                  # +20 vs the opponent's -16
    one = next(m for m in report['rating']['mismatches'] if m['item'] == 'self_after')
    assert one['chosen_src'] == 'supplement' and {c['value'] for c in one['candidates']} == {1520, 1516}
    assert 'rating_mismatch' in report['warnings']


def test_screen_before_and_after_and_v2_display_fields_are_used(tmp_path):
    events = [dict(v=1, ev='reserve', game=1, t=1, budget=1, **COND),
              dict(v=1, ev='started', game=1, t=2, started_at=2, seat_id=1, opp_kind='human', rating_screen=1500),
              dict(v=1, ev='finish', game=1, t=3, terminal='completed', outcome='win', rating_screen=1500,
                   rating_screen_after=1516, rating_screen_gain=16, rating_screen_check='consistent',
                   rating_gain_cross='zero_sum', rating_display_basis='offset_verified',
                   rating_self_display_before=1500, rating_self_display_before_src='wire_absent_zero',
                   rating_self_display_after=1516, rating_self_display_after_src='screen',
                   rating_opp_display_before=1532, rating_opp_display_before_src='wire',
                   rating_opp_display_after=1516, rating_opp_display_after_src='wire',
                   rating_opp_before=32, rating_opp_after=16, rating_self_gain=None, ended_at=3)]
    block = build_report(events)['games'][0]['rating']
    assert block['format'] == 'display_v2'
    assert (block['self_before']['src'], block['self_after']['src']) == ('prematch_screen', 'screen_after')
    assert [c['src'] for c in block['self_after']['candidates']] == ['screen_after', 'ledger_display']
    assert (block['opp_before']['value'], block['opp_before']['src']) == (1532, 'ledger_display')
    assert block['self_before']['agree'] and block['opp_after']['agree'] and block['self_delta'] == 16


def test_reading_never_modifies_or_adds_files_and_ignores_a_partial_tail(tmp_path):
    path = write_ledger(tmp_path / 'l', preliminary() + real_games(), tail=b'{"ev": "finish", "game": 5, "t": 17')
    before = (path.read_bytes(), os.stat(path).st_mtime_ns, sorted(os.listdir(path.parent)))
    report = report_from_path(path)
    assert (path.read_bytes(), os.stat(path).st_mtime_ns, sorted(os.listdir(path.parent))) == before
    assert report['source']['ignored_tail_bytes'] > 0 and 'unterminated_tail_ignored' in report['warnings']
    assert report['counts']['in_progress'] == 1                 # the half-written line was not used
    events, info = read_events(path)
    assert len(events) == 18 and info['invalid_lines'] == []
    assert info['bytes_read'] + info['ignored_tail_bytes'] == len(before[0])


def test_invalid_lines_are_counted_by_number_and_never_echoed(tmp_path):
    path = write_ledger(tmp_path / 'l', preliminary())
    with open(path, 'ab') as handle:
        handle.write(b'not json SyntheticSecretName\n{"v": 1, "ev": "reserve", "game": 3, "name": "x"}\n')
    report = report_from_path(path)
    assert report['source']['invalid_lines'] == [7, 8] and 'invalid_lines' in report['warnings']
    assert 'Synthetic' not in json.dumps(report)
    assert report['preliminary_unapplied']['games'] == [1, 2]


def test_supplement_must_be_numeric_and_is_only_used_for_its_game(tmp_path):
    assert read_supplement(supplement_file(tmp_path, dict(correct_values=dict(self=dict(start_display='1500', end_display=1),
                                                                              opponent=dict(start_display=1, end_display=2)))), 3) is None
    assert read_supplement(tmp_path / 'missing.json', 3) is None
    other = report_from_path(write_ledger(tmp_path / 'l', preliminary() + real_games()),
                             supplement_file(tmp_path), supplement_game=4)
    assert other['games'][2]['rating']['self_before']['value'] is None     # supplement only applies to game 4 now


def test_cli_prints_one_json_object_and_refuses_to_write_next_to_the_ledger(tmp_path, capsys):
    path = write_ledger(tmp_path / 'l', preliminary() + real_games())
    (tmp_path / 'out').mkdir()
    assert main(['--ledger', str(path), '--supplement', str(supplement_file(tmp_path)),
                 '--out', str(tmp_path / 'out/report.json')]) == 0
    text = capsys.readouterr().out
    report = json.loads(text)
    assert report['report_schema'] == 1 and (tmp_path / 'out/report.json').read_text() == text
    assert oct((tmp_path / 'out/report.json').stat().st_mode & 0o777) == '0o600'
    assert main(['--ledger', str(path), '--no-supplement', '--out', str(path.parent / 'report.json')]) == 2
    assert capsys.readouterr().out.strip() == 'error:out_not_writable'
    assert sorted(os.listdir(path.parent)) == ['ledger.jsonl']
    assert main(['--ledger', str(tmp_path / 'nope.jsonl')]) == 2
    assert capsys.readouterr().out.strip() == 'error:ledger_missing'


def test_report_is_anonymous_numbers_and_enums_only(tmp_path):
    report = report_from_path(write_ledger(tmp_path / 'l', preliminary() + real_games()), supplement_file(tmp_path))
    blob = json.dumps(report)
    for forbidden in (str(tmp_path), 'http', 'name', 'room'):
        assert forbidden not in blob
