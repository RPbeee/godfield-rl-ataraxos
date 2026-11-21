"""Read-only, anonymous campaign report for the capped official ranked run.

It reads ``ledger.jsonl`` only: no lock, no create, no write next to it, so it can run while the
ranked loop is appending. The report is one JSON object made of numbers, booleans and fixed enums
(the ledger itself holds nothing else): no names, user IDs, room keys, URLs, paths or raw receive
data. See scripts/ranked_campaign.md for the field list and the rating-source priority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .ranked_ledger import (GAME_NUMBER_MAX, HARD_LIMIT, OPEN_PHASES, WIRE_DISPLAY_OFFSET, fold_games,
                            unapplied_preliminary, validate_event)

REPORT_SCHEMA = 1
Z95 = 1.959963984540054
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / 'runs/ranked100/ledger.jsonl'
DEFAULT_SUPPLEMENT = ROOT / 'measurements/ranked_game3_20261120/game3_rating.json'
DEFAULT_SUPPLEMENT_GAME = 3

# Order in which candidate display ratings are preferred (first available wins; every candidate is
# kept in the report and any disagreement between them is listed under rating.mismatches).
SELF_BEFORE_PRIORITY = ('prematch_screen', 'ledger_display', 'supplement', 'wire_offset', 'prev_game_after')
SELF_AFTER_PRIORITY = ('screen_after', 'next_prematch_screen', 'ledger_display', 'supplement', 'wire_offset',
                       'next_wire_before')
OPP_PRIORITY = ('ledger_display', 'wire_offset', 'supplement')


def is_int(value):
    return type(value) is int


def wilson(k, n, z=Z95):
    """Wilson score interval of k successes in n trials (None fields when n == 0)."""
    if n <= 0:
        return dict(k=k, n=n, rate=None, lo=None, hi=None)
    p, z2 = k / n, z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return dict(k=k, n=n, rate=round(p, 6), lo=round(max(0.0, centre - half), 6),
                hi=round(min(1.0, centre + half), 6))


# ---- reading ---------------------------------------------------------------------------------

def read_events(path):
    """Complete, valid ledger events plus read diagnostics. Never writes, never locks.

    Only bytes up to the last newline are used (the loop may be mid-append); an unterminated tail
    is ignored and counted. Unparseable / invalid lines are counted by line number, never echoed.
    """
    with open(path, 'rb') as handle:
        data = handle.read()
    head, _, tail = data.rpartition(b'\n')
    consumed = data[:len(head) + 1] if len(data) > len(tail) else b''
    events, issues = [], []
    for number, line in enumerate(consumed.split(b'\n'), 1):
        if not line.strip():
            continue
        try:
            events.append(validate_event(json.loads(line)))
        except (ValueError, RecursionError):
            issues.append(number)
    return events, dict(bytes_read=len(consumed), lines=consumed.count(b'\n'),
                        sha256=hashlib.sha256(consumed).hexdigest(), ignored_tail_bytes=len(tail),
                        invalid_lines=issues)


def read_supplement(path, game):
    """Hand-written display-rating supplement for one game (numbers only), or None when unusable."""
    try:
        values = json.loads(Path(path).read_text())['correct_values']
        out = dict(game=game,
                   self_before=values['self']['start_display'], self_after=values['self']['end_display'],
                   opp_before=values['opponent']['start_display'], opp_after=values['opponent']['end_display'])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return out if all(is_int(v) and -10 ** 6 <= v <= 10 ** 6 for k, v in out.items() if k != 'game') else None


# ---- ratings ---------------------------------------------------------------------------------

def wire_display(raw):
    return raw + WIRE_DISPLAY_OFFSET if is_int(raw) else None


def resolve(candidates, priority):
    """Pick by priority; keep all candidates and say whether the available ones agree."""
    ranked = sorted((c for c in candidates if is_int(c['value'])), key=lambda c: priority.index(c['src']))
    values = {c['value'] for c in ranked}
    return dict(value=ranked[0]['value'] if ranked else None, src=ranked[0]['src'] if ranked else 'unknown',
                candidates=ranked, agree=len(values) <= 1)


def cand(src, value, **extra):
    return dict(src=src, value=value, **extra)


def rating_block(row, nxt, supplement, prev_resolved):
    """Display before/after of both seats for one participated game, from every grounded source."""
    sup = supplement if supplement and supplement['game'] == row['game'] else None
    consecutive_next = nxt if nxt is not None and nxt['game'] == row['game'] + 1 else None
    legacy = 'rating_display_basis' not in row
    s_basis = lambda key: row.get(key)
    self_before = [
        cand('prematch_screen', row.get('rating_screen')),
        cand('ledger_display', row.get('rating_self_display_before'), basis=s_basis('rating_self_display_before_src')),
        cand('supplement', sup['self_before'] if sup else None),
        cand('wire_offset', wire_display(row.get('rating_self_before'))),
        cand('prev_game_after', prev_resolved),
    ]
    self_after = [
        cand('screen_after', row.get('rating_screen_after')),
        cand('next_prematch_screen', consecutive_next.get('rating_screen') if consecutive_next else None),
        cand('ledger_display', row.get('rating_self_display_after'), basis=s_basis('rating_self_display_after_src')),
        cand('supplement', sup['self_after'] if sup else None),
        cand('wire_offset', wire_display(row.get('rating_self_after'))),
        cand('next_wire_before', wire_display(consecutive_next.get('rating_self_before')) if consecutive_next else None),
    ]
    opp_before = [
        cand('ledger_display', row.get('rating_opp_display_before'), basis=s_basis('rating_opp_display_before_src')),
        cand('wire_offset', wire_display(row.get('rating_opp_before'))),
        cand('supplement', sup['opp_before'] if sup else None),
    ]
    opp_after = [
        cand('ledger_display', row.get('rating_opp_display_after'), basis=s_basis('rating_opp_display_after_src')),
        cand('wire_offset', wire_display(row.get('rating_opp_after'))),
        cand('supplement', sup['opp_after'] if sup else None),
    ]
    block = dict(format='legacy' if legacy else 'display_v2',
                 self_before=resolve(self_before, SELF_BEFORE_PRIORITY), self_after=resolve(self_after, SELF_AFTER_PRIORITY),
                 opp_before=resolve(opp_before, OPP_PRIORITY), opp_after=resolve(opp_after, OPP_PRIORITY))
    delta = lambda a, b: b['value'] - a['value'] if a['value'] is not None and b['value'] is not None else None
    block['self_delta'] = delta(block['self_before'], block['self_after'])
    block['opp_delta'] = delta(block['opp_before'], block['opp_after'])
    return block


def rating_mismatches(row, block):
    """Explicit list of every inconsistency inside one game's rating evidence."""
    found = []
    for item in ('self_before', 'self_after', 'opp_before', 'opp_after'):
        if not block[item]['agree']:
            found.append(dict(game=row['game'], item=item, kind='value_disagreement', chosen_src=block[item]['src'],
                              candidates=[dict(src=c['src'], value=c['value']) for c in block[item]['candidates']]))
    s, o = block['self_delta'], block['opp_delta']
    if s is not None and o is not None and s != -o:
        found.append(dict(game=row['game'], item='deltas', kind='not_zero_sum', self_delta=s, opp_delta=o))
    gain = row.get('rating_self_gain')
    if is_int(gain) and s is not None and gain != s:
        found.append(dict(game=row['game'], item='self_gain', kind='self_gain_vs_delta', wire_gain=gain, self_delta=s))
    outcome = row.get('outcome')
    if s not in (None, 0) and ((outcome == 'win' and s < 0) or (outcome == 'loss' and s > 0)):
        found.append(dict(game=row['game'], item='self_delta', kind='outcome_vs_delta', outcome=outcome, self_delta=s))
    return found


# ---- the report ------------------------------------------------------------------------------

PUBLIC_KEYS = ('phase', 'terminal', 'outcome', 'stop_reason', 'opp_kind', 'seat_id', 'first_mover', 'decisions',
               'updates', 'duration_s', 'gf_final', 'end_gf', 'resumes', 'started_at', 'ended_at')
CONDITION_KEYS = ('model', 'ckpt_sha8', 'code_rev', 'policy', 'search', 'dream')


def classify(row, exempt):
    if row['game'] in exempt:
        return 'preliminary_unapplied'
    started = 'started_at' in row
    if row['phase'] == 'finished':
        return 'completed'
    if row['phase'] in OPEN_PHASES:
        return 'in_progress' if started else 'open_unmatched'
    return 'undecided_after_start' if started else 'undecided_unmatched'


def build_report(events, diagnostics=None, supplement=None):
    rows = fold_games(events)
    exempt = unapplied_preliminary(events)
    by_game = {row['game']: row for row in rows}
    games, mismatches, previous_after = [], [], {}
    for row in rows:
        klass = classify(row, exempt)
        entry = dict(game=row['game'], klass=klass, counted=row['game'] not in exempt,
                     **{k: row.get(k) for k in PUBLIC_KEYS})
        if klass not in ('preliminary_unapplied', 'open_unmatched', 'undecided_unmatched') and 'started_at' in row:
            block = rating_block(row, by_game.get(row['game'] + 1), supplement,
                                 previous_after.get(row['game'] - 1))
            entry['rating'] = block
            previous_after[row['game']] = block['self_after']['value']
            mismatches.extend(rating_mismatches(row, block))
        games.append(entry)

    counted = [g for g in games if g['counted']]
    of = lambda *klasses: [g for g in counted if g['klass'] in klasses]
    completed = of('completed')
    known = [g for g in completed if g['outcome'] in ('win', 'loss', 'draw')]
    wins = sum(g['outcome'] == 'win' for g in known)
    losses = sum(g['outcome'] == 'loss' for g in known)
    draws = sum(g['outcome'] == 'draw' for g in known)
    participated = [g for g in counted if g['started_at'] is not None]
    # Closed after the match started, but with no known result: finished with an unknown outcome,
    # stopped, or lost to a restart. They are real (if undecided) games.
    undecided_started = [g for g in participated if g['klass'] in ('undecided_after_start', 'completed')
                         and g['outcome'] not in ('win', 'loss', 'draw')]
    closed = len(known) + len(undecided_started)
    kinds = {}
    for g in participated:
        bucket = kinds.setdefault(g['opp_kind'] or 'unknown', dict(participated=0, wins=0, losses=0, draws=0,
                                                                   undecided=0, in_progress=0))
        bucket['participated'] += 1
        if g['klass'] == 'in_progress':
            bucket['in_progress'] += 1
        elif g['outcome'] in ('win', 'loss', 'draw'):
            bucket[dict(win='wins', loss='losses', draw='draws')[g['outcome']]] += 1
        else:
            bucket['undecided'] += 1
    for bucket in kinds.values():
        decided = bucket['wins'] + bucket['losses'] + bucket['draws']
        bucket['win_rate_completed'] = wilson(bucket['wins'], decided)
    conditions = {}
    reserves = {}
    for event in events:
        if event['ev'] == 'reserve':
            reserves[event['game']] = event
    for game, event in sorted(reserves.items()):
        key = tuple(event.get(k) for k in CONDITION_KEYS)
        conditions.setdefault(key, []).append(game)
    warnings = []
    if diagnostics:
        if diagnostics['invalid_lines']:
            warnings.append('invalid_lines')
        if diagnostics['ignored_tail_bytes']:
            warnings.append('unterminated_tail_ignored')
    if len(conditions) > 1:
        warnings.append('mixed_conditions')
    if [g['game'] for g in games] != list(range(1, len(games) + 1)):
        warnings.append('game_numbers_not_contiguous')
    if len(counted) > HARD_LIMIT:
        warnings.append('over_hard_limit')
    if any(g['game'] in (1, 2) and g['game'] not in exempt for g in games):
        warnings.append('preliminary_games_not_verified')
    if any(g['klass'] in ('in_progress', 'open_unmatched') for g in games):
        warnings.append('game_in_progress')

    series = [dict(game=g['game'], outcome=g['outcome'], self_before=g['rating']['self_before']['value'],
                   self_after=g['rating']['self_after']['value'], opp_before=g['rating']['opp_before']['value'],
                   opp_after=g['rating']['opp_after']['value'], self_delta=g['rating']['self_delta'],
                   opp_delta=g['rating']['opp_delta']) for g in counted if 'rating' in g]
    after = [s['self_after'] for s in series if s['self_after'] is not None]
    deltas = [s['self_delta'] for s in series if s['self_delta'] is not None]
    first_before = next((s['self_before'] for s in series if s['self_before'] is not None), None)
    endpoints = after[-1] - first_before if after and first_before is not None else None
    if endpoints is not None and len(deltas) == len(series) and sum(deltas) != endpoints \
            and all(s['self_before'] is not None for s in series):
        # With every game known, the per-game deltas must add up to the endpoint difference.
        mismatches.append(dict(game=None, item='net_change', kind='net_change', sum_of_deltas=sum(deltas),
                               endpoints=endpoints))
    if mismatches:
        warnings.append('rating_mismatch')
    opp_before = [s['opp_before'] for s in series if s['opp_before'] is not None]
    exempt_rows = [g for g in games if not g['counted']]
    return dict(
        report_schema=REPORT_SCHEMA,
        source=dict(diagnostics or {}, events=len(events)),
        cap=dict(hard_limit=HARD_LIMIT, game_number_max=GAME_NUMBER_MAX, counted=len(counted),
                 remaining=max(0, HARD_LIMIT - len(counted)), game_numbers=len(games),
                 highest_game_number=max((g['game'] for g in games), default=0),
                 budgets_recorded=sorted({e['budget'] for e in events if e['ev'] == 'reserve'})),
        preliminary_unapplied=dict(
            games=[g['game'] for g in exempt_rows], count=len(exempt_rows),
            basis='pinned_ledger_events' if exempt_rows else 'none',
            stop_reasons=sorted({g['stop_reason'] for g in exempt_rows}),
            decisions=sum(g['decisions'] or 0 for g in exempt_rows),
            note='entry clicked, application button 戦う not pressed, no opponent; excluded from every count below'),
        counts=dict(counted=len(counted), participated=len(participated), completed=len(known),
                    wins=wins, losses=losses, draws=draws, undecided_after_start=len(undecided_started),
                    undecided_unmatched=len(of('undecided_unmatched')), in_progress=len(of('in_progress')),
                    open_unmatched=len(of('open_unmatched')), closed_with_match=closed),
        win_rate=dict(
            completed=wilson(wins, len(known)),
            decided_only=wilson(wins, wins + losses),
            unknown_as_loss=wilson(wins, closed),
            unknown_as_win=wilson(wins + len(undecided_started), closed),
            note='95% Wilson; draws are not wins; completed = known outcomes of games that really started'),
        by_opp_kind=dict(sorted(kinds.items())),
        conditions=[dict(zip(CONDITION_KEYS, key), games=games_) for key, games_ in conditions.items()],
        games=games,
        rating=dict(
            priority=dict(self_before=list(SELF_BEFORE_PRIORITY), self_after=list(SELF_AFTER_PRIORITY),
                          opp=list(OPP_PRIORITY), wire_display_offset=WIRE_DISPLAY_OFFSET),
            series=series,
            self=dict(first_before=first_before, last_after=after[-1] if after else None,
                      net_change=endpoints, sum_of_known_deltas=sum(deltas) if deltas else None,
                      games_with_unknown_after=sum(s['self_after'] is None for s in series),
                      min_after=min(after) if after else None, max_after=max(after) if after else None),
            opponents=dict(known=len(opp_before), mean_before=round(sum(opp_before) / len(opp_before), 2)
                           if opp_before else None),
            mismatches=mismatches),
        warnings=warnings)


def report_from_path(path, supplement_path=None, supplement_game=DEFAULT_SUPPLEMENT_GAME):
    events, diagnostics = read_events(path)
    supplement = read_supplement(supplement_path, supplement_game) if supplement_path else None
    report = build_report(events, diagnostics, supplement)
    if supplement_path:
        report['source']['supplement'] = dict(game=supplement_game, usable=supplement is not None)
    return report


def write_report(path, text, ledger_path):
    """Atomic 0600 write that can never touch the ledger directory."""
    target, ledger = Path(path).absolute(), Path(ledger_path).absolute()
    if target.resolve().parent == ledger.resolve().parent:
        raise ValueError('out_inside_ledger_directory')
    tmp = target.with_name(target.name + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(text)
    os.replace(tmp, target)


def build_parser():
    parser = argparse.ArgumentParser(description='Read-only anonymous campaign report of the ranked ledger')
    parser.add_argument('--ledger', type=Path, default=DEFAULT_LEDGER, help='ledger.jsonl (read only, no lock)')
    parser.add_argument('--supplement', type=Path, default=DEFAULT_SUPPLEMENT,
                        help='hand-written display-rating supplement for one game (numbers only)')
    parser.add_argument('--supplement-game', type=int, default=DEFAULT_SUPPLEMENT_GAME)
    parser.add_argument('--no-supplement', action='store_true')
    parser.add_argument('--out', type=Path, help='also write the JSON here (never inside the ledger directory)')
    parser.add_argument('--indent', type=int, default=None, help='pretty-print with this indent')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.ledger.is_file():
        print('error:ledger_missing'); return 2
    supplement = None if args.no_supplement or not args.supplement.is_file() else args.supplement
    report = report_from_path(args.ledger, supplement, args.supplement_game)
    report['generated_at'] = int(time.time())
    text = json.dumps(report, sort_keys=True, allow_nan=False, indent=args.indent) + '\n'
    if args.out is not None:
        try:
            write_report(args.out, text, args.ledger)
        except (OSError, ValueError):
            print('error:out_not_writable'); return 2
    sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
