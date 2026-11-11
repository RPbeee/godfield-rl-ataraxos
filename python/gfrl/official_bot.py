"""UI bot for training and two-player password rooms. Imports stay offline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import re
import shutil
import stat
import subprocess
import sys
import threading
import tempfile
import time

from .official import ROOT, OfficialSession, OfficialStreams, exception_detail, parser_module
from .official_ui import UIUnsupported, require, prepare, execute, ready, wait_visible, PUBLIC_PATHS


class ReceiveAdvanced(Exception):
    def __init__(self, result):
        self.result = result


class Probe:
    def __init__(self, command, timeout=30):
        self.timeout = timeout
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.lines = queue.Queue()
        self.broken = False
        def reader():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        threading.Thread(target=reader, daemon=True).start()
        ready = self.receive()
        if ready.get('ready') is not True:
            self.broken = True
            self.process.terminate(); self.process.wait(timeout=5)
            raise UIUnsupported('probe_not_ready')

    def receive(self):
        try:
            line = self.lines.get(timeout=self.timeout)
            value = json.loads(line) if line is not None else None
            require(isinstance(value, dict) and not value.get('error'), 'probe_error')
            return value
        except Exception:
            self.broken = True
            self.process.terminate()
            raise UIUnsupported('probe_failure') from None

    def send(self, command):
        require(not self.broken, 'probe_stopped')
        try:
            self.process.stdin.write(json.dumps(command, ensure_ascii=False)+'\n')
            self.process.stdin.flush()
        except (OSError, ValueError):
            self.broken = True
            raise UIUnsupported('probe_input_failure') from None
        reply = self.receive()
        if command['kind'] == 'close' and 'capture' in reply:
            reply = self.receive()
        return reply

    def close(self):
        try:
            if not self.broken:
                self.send(dict(kind='close'))
        finally:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
            self.process.stdin.close(); self.process.stdout.close()


def private_directory(path):
    path = Path(path).absolute()
    require(not path.is_symlink(), 'output_symlink')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


class Journal:
    def __init__(self, directory):
        self.directory = directory
        self.file = open(os.open(directory / 'moves.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'w')

    def write(self, value):
        from .official_search import empty_diagnostics
        advice = value.get('advice', {})
        value = dict(value, **{key: advice.get(key, default)
            for key, default in dict(search_status='not_a_decision', **empty_diagnostics()).items()})
        self.file.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n'); self.file.flush()

    def close(self):
        self.file.close()


def poll_state(probe, predicate, timeout=20):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        state = probe.send(dict(kind='state'))['state']
        if predicate(state):
            return state
        time.sleep(.2)
    raise UIUnsupported('visible_state_timeout')


def settings(state):
    body = state['body']
    return ('G.F.' not in body and '戦いを始める' in body and
            re.search(r'修行者\s*2\s*人', body) is not None and
            re.search(r'終末の時\s*なし(?:\s|$)', body) is not None)


def wait_login(probe, predicate, timeout):
    last = None
    def check(state):
        nonlocal last
        last = state
        return predicate(state)
    try:
        return poll_state(probe, check, timeout)
    except UIUnsupported as error:
        if str(error) == 'visible_state_timeout' and last is not None and '生誕する' in last['body']:
            print('名前が使用中の可能性。別の名前で', file=sys.stderr)
            raise UIUnsupported('login_not_accepted') from None
        raise


def enter_training(probe, name, *, login_timeout=20):
    state = poll_state(probe, lambda s: '生誕する' in s['body'] or 'コンピュータと対戦' in s['body'] or settings(s))
    if '生誕する' in state['body']:
        probe.send(dict(kind='fillName', text=name))
        probe.send(dict(kind='clickText', text='生誕する'))
        state = wait_login(probe, lambda s: 'コンピュータと対戦' in s['body'], login_timeout)
    if not settings(state):
        probe.send(dict(kind='clickText', text='修行'))
    return poll_state(probe, settings)


def room_screen(state):
    return all(t in state['body'] for t in ('隠れ乱闘', '個人戦', 'チーム戦', '終末の時'))


def room_transition(state):
    # The room header/footer remain while the central room/battle UI is empty.
    # A lobby also mentions 隠れ乱闘, so require it to be the heading.
    body = state['body']
    return (re.match(r'\s*(?:←\s*)?隠れ乱闘(?:\s|$)', body) is not None
            and not any(t in body for t in ('個人戦', 'チーム戦', '終末の時',
                                            '戦いを始める', 'G.F.', '修行', '真剣タイマン')))


def wait_room_return(probe, room_wait):
    deadline = time.monotonic()+room_wait
    while time.monotonic() < deadline:
        state = probe.send(dict(kind='state'))['state']
        if room_screen(state):
            return state
        require(room_transition(state) or 'G.F.' in state['body'], 'room_start_screen_unknown')
        time.sleep(min(.2, max(0, deadline-time.monotonic())))
    raise UIUnsupported('room_wait_timeout')


def room_host(state):
    return room_screen(state) and '戦いを始める' in state['body']


def room_members(state, name, *, joined=True):
    require(room_screen(state), 'room_screen_missing')
    # Only the central nine individual slots identify participants. Header and
    # right-hand room occupants are deliberately outside this rectangle.
    rows = [r for r in state.get('texts', []) if 580 <= r['x'] <= 860
            and any(abs(r['y'] + r['h']/2 - (153+35*i)) <= 12 for i in range(9))]
    own = sum(r['text'] == name for r in rows)
    require(own == 1 if joined else own <= 1, 'room_own_participant_not_unique')
    require(len(rows) <= 2, 'room_participant_count')
    return rows


def room_no_end(state):
    # The open selection menu contains なし as well; reject its GF options.
    return (room_screen(state) and 'G.F.' not in state['body']
            and re.search(r'終末の時\s*なし(?:\s|$)', state['body']) is not None)


def previous_room_game(probe, name, state, baseline, room_wait, *, strict_events=False):
    """Leave an observed spectator game; never resume a previous own game."""
    parser = parser_module()
    sanitize = parser.sanitize_game
    own = []
    def inspect(game, viewer_id=None, **options):
        own.append(any(p.get('name') == name for p in game.get('players', [])))
        return sanitize(game, viewer_id, **options)
    parser.sanitize_game = inspect
    boundaries = {OfficialStreams.stream_key(c['file']): c['bytes'] for c in baseline}
    deadline = time.monotonic()+room_wait
    clicked = False
    while True:
        spectators = []
        for c in capture(probe):
            own.clear()
            try:
                records = parser.parse_prefix(checked_capture(c), lenient_events=not strict_events)['games']
            except parser.ParseError as error:
                raise UIUnsupported('unsupported_receive', exception_detail(error, parser.ParseError)) from None
            fresh = [(r, participant) for r, participant in zip(records, own)
                     if r['frame_end_byte'] > boundaries.get(OfficialStreams.stream_key(c['file']), 0)]
            if fresh:
                require(not fresh[-1][1], 'room_previous_game_in_progress')
                spectators.append(True)
        if room_screen(state):
            # The foreign game's records must not enter the next game's parser.
            baseline[:] = capture(probe)
            return state
        require(room_transition(state) or 'G.F.' in state['body'] or '戦いを終わる' in state['body'],
                'room_start_screen_unknown')
        if spectators and '戦いを終わる' in state['body'] and not clicked:
            probe.send(dict(kind='clickText', text='戦いを終わる'))
            clicked = True
        require(time.monotonic() < deadline, 'room_wait_timeout')
        time.sleep(min(.2, max(0, deadline-time.monotonic())))
        state = probe.send(dict(kind='state'))['state']


def prepare_room(probe, name, timeout=20, *, room_wait=1800, baseline=None, strict_events=False):
    state = poll_state(probe, lambda s: room_screen(s) or 'G.F.' in s['body']
                       or '戦いを終わる' in s['body'], timeout)
    if not room_screen(state):
        state = previous_room_game(probe, name, state, baseline if baseline is not None else [],
                                   room_wait, strict_events=strict_events)
    room_members(state, name, joined=False)
    if not room_no_end(state) and room_host(state):
        probe.send(dict(kind='clickText', text='終末の時'))
        poll_state(probe, lambda s: room_screen(s) and 'G.F.1' in s['body'] and 'なし' in s['body'], timeout)
        probe.send(dict(kind='clickText', text='なし'))
        state = poll_state(probe, room_no_end, timeout)
    if not room_no_end(state):
        deadline = time.monotonic()+room_wait
        while not room_no_end(state):
            require(time.monotonic() < deadline, 'room_end_time_not_none')
            time.sleep(.2)
            state = probe.send(dict(kind='state'))['state']
            if not room_screen(state):
                if room_transition(state):
                    continue
                require('G.F.' in state['body'], 'room_start_screen_unknown')
                return state  # The received game still has its own end-time check.
            room_members(state, name, joined=False)
    rows = room_members(state, name, joined=False)
    if not any(r['text'] == name for r in rows):
        probe.send(dict(kind='clickPoint', x=400, y=173))
    def joined(s):
        require(room_no_end(s), 'room_end_time_changed')
        rows = room_members(s, name, joined=False)
        return sum(r['text'] == name for r in rows) == 1
    return poll_state(probe, joined, timeout)


def enter_room(probe, name, mode='room', password=None, *, timeout=20, room_wait=1800,
               login_timeout=None, baseline=None, strict_events=False):
    login_timeout = timeout if login_timeout is None else login_timeout
    if mode == 'training':
        return enter_training(probe, name, login_timeout=login_timeout)
    require(mode == 'room' and isinstance(password, str) and bool(password), 'room_password_required')
    def lobby(s):
        return all(t in s['body'] for t in ('修行', '隠れ乱闘', '真剣タイマン'))
    state = poll_state(probe, lambda s: '生誕する' in s['body'] or lobby(s) or room_screen(s), timeout)
    if '生誕する' in state['body']:
        probe.send(dict(kind='fillName', text=name))
        probe.send(dict(kind='clickText', text='生誕する'))
        state = wait_login(probe, lobby, login_timeout)
    if not room_screen(state):
        probe.send(dict(kind='clickText', text='隠れ乱闘'))
        poll_state(probe, lambda s: all(t in s['body'] for t in ('部屋の合言葉', '唱える'))
                   and len(s.get('inputs', [])) == 1, timeout)
        probe.send(dict(kind='fillName', text=password))
        probe.send(dict(kind='clickText', text='唱える'))
    return prepare_room(probe, name, timeout, room_wait=room_wait, baseline=baseline, strict_events=strict_events)


def named_room_parser(name, baseline, viewer=None):
    # A private parser instance keeps transient raw-name checks out of the
    # shared parser and projections. Only numeric seat IDs escape this hook.
    parser = parser_module()
    sanitize = parser.sanitize_game
    parse = parser.parse_prefix
    boundaries = {OfficialStreams.stream_key(c['file']): c['bytes'] for c in baseline}
    matches = []
    def inspect(game, viewer_id=None, **options):
        seats = [p.get('id') for p in game.get('players', []) if p.get('name') == name]
        matches.append((seats, len(game.get('players', [])), game.get('tiebreakGF')))
        return sanitize(game, viewer_id, **options)
    parser.sanitize_game = inspect
    def checked(raw, viewer_id=None, *, boundary=0, **options):
        matches.clear()
        parsed = parse(raw, viewer_id, **options)
        for record, (seats, count, end) in zip(parsed['games'], matches):
            if record['frame_end_byte'] <= boundary:
                continue
            require(count == 2, 'room_game_not_two_players')
            require(len(seats) == 1 and (viewer is None or seats[0] == viewer), 'viewer_not_unique')
            require(end is None, 'room_game_end_time')
            record['room_viewer'] = seats[0]
        matches.clear()
        return parsed
    parser.parse_prefix = checked
    parser.parse_capture = lambda raw, viewer_id, boundary, **options: checked(raw, viewer_id, boundary=boundary, **options)
    return parser, boundaries


def wait_room_start(probe, name, baseline, *, room_start, room_wait, strict_events=False):
    parser, boundaries = named_room_parser(name, baseline)
    deadline = time.monotonic()+room_wait
    clicked = False
    notified = False
    end_pending = False
    while time.monotonic() < deadline:
        state = probe.send(dict(kind='state'))['state']
        if room_screen(state):
            host = room_host(state)
            end_pending = not room_no_end(state)
            require(not host or not end_pending, 'room_end_time_changed')
            members = room_members(state, name)
            if room_start == 'bot' and not host and not notified:
                print('部屋主ではないため、部屋主が戦いを始めるのを待ちます。', file=sys.stderr)
                notified = True
            if room_start == 'bot' and host and not end_pending and len(members) == 2 and not clicked:
                probe.send(dict(kind='clickText', text='戦いを始める'))
                clicked = True
        else:
            require(room_transition(state) or 'G.F.' in state['body'], 'room_start_screen_unknown')
        viewers = set()
        for c in capture(probe):
            raw = checked_capture(c)
            try:
                records = parser.parse_prefix(raw, boundary=boundaries.get(OfficialStreams.stream_key(c['file']), 0),
                                              lenient_events=not strict_events)['games']
            except parser.ParseError as error:
                raise UIUnsupported('unsupported_receive', exception_detail(error, parser.ParseError)) from None
            viewers.update(r['room_viewer'] for r in records if 'room_viewer' in r)
        require(len(viewers) <= 1, 'viewer_not_unique')
        if viewers:
            return viewers.pop()
        time.sleep(.2)
    raise UIUnsupported('room_end_time_not_none' if end_pending else 'room_wait_timeout')


def checked_capture(c):
    import hashlib
    require(not c.get('capture_error'), 'capture_failed')
    raw = Path(c['file']).read_bytes()
    require(len(raw) == c['bytes'] and (not c.get('sha256') or hashlib.sha256(raw).hexdigest() == c['sha256']),
            'capture_integrity')
    return raw


def capture(probe):
    result = probe.send(dict(kind='capture'))
    return result['capture']['captures']


def accepted(probe, streams, update, timeout=20, *, retry=None, retry_after=6):
    started = time.monotonic()
    deadline = started+timeout
    retried = False
    while time.monotonic() < deadline:
        result = streams.poll(capture(probe))
        require(result['status'] != 'unsupported', 'unsupported_receive', result.get('detail', {}))
        if result.get('public', {}).get('update', update) > update:
            return result
        if retry is not None and not retried and time.monotonic()-started >= retry_after:
            retried = True
            retry()
        time.sleep(min(.2, max(0, deadline-time.monotonic())))
    raise UIUnsupported('acceptance_timeout')


def summary(public):
    return {k: public[k] for k in ('update', 'gf', 'phase', 'actor', 'viewer', 'is_over')}


def run(probe, advisor, journal, *, name, games=1, mode='training', password=None,
        dry_run=False, timeout=20, settle_timeout=30, on_stop='save',
        room_start='human', room_wait=1800, turn_wait=900, strict_events=False, chance_delay=3):
    counts = dict(games=0, wins=0, losses=0, draws=0)
    current_record = None
    successful_kinds = set()
    try:
        require(mode != 'room' or on_stop == 'save', 'room_abandon_unsupported')
        room_baseline = capture(probe) if mode == 'room' else None
        enter_room(probe, name, mode, password, timeout=settle_timeout, room_wait=room_wait,
                   login_timeout=timeout, baseline=room_baseline, strict_events=strict_events)
        for game_index in range(games):
            baseline = room_baseline if mode == 'room' else capture(probe)
            if mode == 'room':
                viewer = wait_room_start(probe, name, baseline, room_start=room_start, room_wait=room_wait, strict_events=strict_events)
            else:
                probe.send(dict(kind='clickText', text='戦いを始める'))
                poll_state(probe, lambda s: 'G.F.' in s['body'])
                viewer = getattr(probe, 'viewer', None)
            # The parser infers the unique human seat in training; the bot checks
            # the same roster before constructing the fixed-viewer session.
            if viewer is None:
                records = capture(probe)
                parser = __import__('gfrl.official', fromlist=['parser_module']).parser_module()
                humans = set()
                for c in records:
                    try:
                        parsed = parser.parse_prefix(Path(c['file']).read_bytes(), lenient_events=not strict_events)
                    except parser.ParseError as error:
                        raise UIUnsupported('unsupported_receive', exception_detail(error, parser.ParseError)) from None
                    for r in parsed['games']:
                        if r['frame_end_byte'] > {OfficialStreams.stream_key(x['file']): x['bytes'] for x in baseline}.get(OfficialStreams.stream_key(c['file']), 0):
                            humans.update(p['id'] for p in r['game']['players'] if not p.get('isBot', False))
                require(len(humans) == 1, 'viewer_not_unique')
                viewer = humans.pop()
            streams = OfficialStreams(OfficialSession(viewer), baseline, lenient_events=not strict_events)
            reported_events = 0
            if mode == 'room':
                streams.parser, _ = named_room_parser(name, baseline, viewer)
            result = accepted(probe, streams, -1, timeout)
            while True:
                require(result['status'] != 'unsupported', 'unsupported_receive', result.get('detail', {}))
                public = result['public']
                replacements = result.get('replaced_events', [])
                if len(replacements) > reported_events:
                    print('warning:replaced_events ' + json.dumps(replacements[reported_events:], sort_keys=True), file=sys.stderr)
                    reported_events = len(replacements)
                if result['status'] == 'finished':
                    own = next(p for p in public['players'] if p['id'] == viewer)
                    other = next(p for p in public['players'] if p['id'] != viewer)
                    outcome = 'draws' if own['dead'] == other['dead'] else 'losses' if own['dead'] else 'wins'
                    counts['games'] += 1; counts[outcome] += 1
                    journal.write(dict(status='finished', state=summary(public), outcome=outcome, replaced_events=replacements))
                    if mode == 'room' and counts['games'] >= games:
                        return dict(counts, status='complete')
                    end_state = poll_state(probe, lambda s: '戦いを終わる' in s['body']
                                           or (mode == 'room' and room_screen(s)))
                    if mode == 'room':
                        room_baseline = capture(probe)
                    if '戦いを終わる' in end_state['body']:
                        probe.send(dict(kind='clickText', text='戦いを終わる'))
                    if mode == 'room':
                        wait_room_return(probe, room_wait)
                        if game_index+1 < games:
                            prepare_room(probe, name, settle_timeout, room_wait=room_wait)
                    else:
                        poll_state(probe, settings)
                    break
                if result['status'] != 'decision':
                    result = accepted(probe, streams, public['update'], turn_wait if mode == 'room' else timeout)
                    continue
                advice = advisor(result)
                current_record = dict(state=summary(public), advice=advice, clicks=[], accepted=False,
                                      replaced_events=replacements)
                for key in ('fallback_samples', 'selection_exhausted', 'dream_last_resort'):
                    if key in advice:
                        current_record[key] = advice[key]
                require(advice['status'] == 'advice', 'unsupported_advice',
                        dict(unsupported=advice.get('unsupported', []),
                             action=dict(kind=advice.get('action', {}).get('kind'),
                                         models=[c['model'] for c in advice.get('action', {}).get('cards', [])])))
                def refresh():
                    latest = streams.poll(capture(probe))
                    require(latest['status'] != 'unsupported', 'unsupported_receive', latest.get('detail', {}))
                    if latest.get('public', {}).get('update', public['update']) > public['update']:
                        raise ReceiveAdvanced(latest)

                try:
                    if any(e['action'] in ('hit', 'miss') for e in public.get('events', [])):
                        deadline = time.monotonic() + chance_delay
                        while time.monotonic() < deadline:
                            refresh()
                            time.sleep(min(.5, max(0, deadline-time.monotonic())))
                    state = probe.send(dict(kind='state'))['state']
                    state = wait_visible(probe, state, lambda s: ready(s, public, name),
                                         timeout=settle_timeout, refresh=refresh)
                    plan = prepare(state, public, advice['action'], name,
                                   dream_last_resort=advice.get('dream_last_resort') is True)
                    if not dry_run:
                        plan = execute(probe, state, public, advice['action'], name,
                                commands=current_record['clicks'], settle_timeout=settle_timeout,
                                refresh=refresh, dream_last_resort=advice.get('dream_last_resort') is True)
                except ReceiveAdvanced as advanced:
                    current_record.update(status='superseded', received_update=advanced.result['public']['update'])
                    journal.write(current_record); current_record = None
                    result = advanced.result
                    continue
                if dry_run:
                    current_record.update(status='dry_run', planned_clicks=plan['commands'])
                    journal.write(current_record)
                    print(json.dumps(current_record, ensure_ascii=False))
                    current_record = None
                    return dict(counts, status='dry_run')
                current_record['confirm_method'] = 'pane'
                if plan.get('selection_retries'):
                    current_record['selection_retries'] = plan['selection_retries']
                def retry_confirmation():
                    from .official_ui import same_confirmation
                    s = probe.send(dict(kind='state'))['state']
                    if same_confirmation(s, public, advice['action'], name, plan):
                        # Recheck the receive after reading the visible scene.
                        latest = streams.poll(capture(probe))
                        require(latest['status'] != 'unsupported', 'unsupported_receive', latest.get('detail', {}))
                        if latest.get('public', {}).get('update', public['update']) > public['update']:
                            raise ReceiveAdvanced(latest)
                        current_record['confirm_method'] = 'pane_retry'
                        current_record['confirmation_retries'] = 1
                        current_record['clicks'].append(plan['confirm'])
                        probe.send(plan['confirm'])
                try:
                    result = accepted(probe, streams, public['update'], timeout, retry=retry_confirmation)
                except UIUnsupported as error:
                    if (str(error) != 'acceptance_timeout' or public['phase'] != 1
                            or advice['action']['kind'] != 'defend' or advice['action']['cards']):
                        raise
                    from .official_ui import allow_button
                    s = probe.send(dict(kind='state'))['state']
                    try:
                        button = allow_button(s, public, advice['action'], name, plan)
                    except UIUnsupported:
                        raise error
                    latest = streams.poll(capture(probe))
                    require(latest['status'] != 'unsupported', 'unsupported_receive', latest.get('detail', {}))
                    if latest.get('public', {}).get('update', public['update']) > public['update']:
                        result = latest
                    else:
                        current_record['confirm_method'] = 'allow_button'
                        current_record['clicks'].append(button)
                        probe.send(button)
                        result = accepted(probe, streams, public['update'], timeout)
                except ReceiveAdvanced as advanced:
                    result = advanced.result
                current_record.update(status='accepted', accepted=True, received_update=result['public']['update'])
                kind = advice['action']['kind']
                if kind not in successful_kinds:
                    current_record['first_success_kind'] = kind
                    successful_kinds.add(kind)
                journal.write(current_record); current_record = None
        return dict(counts, status='complete')
    except Exception as error:
        code = str(error) if isinstance(error, UIUnsupported) else 'bot_input_or_model_failure'
        detail = error.detail if isinstance(error, UIUnsupported) else exception_detail(error, fixed_codes=False, local_only=False)
        # Advice is an external/model boundary; a failing payload must not be
        # copied wholesale into stopped rows (even if it contains a name).
        stopped = dict(status='stopped', reason=code, detail=detail)
        if current_record is not None:
            stopped.update(state=current_record['state'], clicks=current_record['clicks'], accepted=False)
            recorded_advice = current_record.get('advice')
            recorded_advice = recorded_advice if isinstance(recorded_advice, dict) else {}
            a = recorded_advice.get('action')
            a = a if isinstance(a, dict) else {}
            cards = a.get('cards')
            cards = cards if isinstance(cards, list) else []
            # Persist only validated public scalar fields, never model prose or names.
            stopped['action'] = dict(kind=a.get('kind') if a.get('kind') in
                ('play','defend','discard','pray','exchange','sell','buy','accept','decline','sacrifice') else None,
                cards=[dict(model=c['model'], item_id=c['item_id']) for c in cards[:18]
                       if isinstance(c, dict) and type(c.get('model')) is int and 1 <= c['model'] <= 296
                       and type(c.get('item_id')) is int],
                target=a.get('target') if type(a.get('target')) is int else None)
            if recorded_advice.get('dream_last_resort') is True:
                stopped['dream_last_resort'] = True
                stopped['warnings'] = ['dream_actual_execution_not_verified_from_public_faces']
            for key in ('confirm_method', 'fallback_samples', 'selection_exhausted'):
                if key in current_record:
                    stopped[key] = current_record[key]
            if isinstance(recorded_advice.get('mask_exclusions'), list):
                stopped['mask_exclusions'] = [dict(reason=e.get('reason') if e.get('reason') in (
                    'dream_actual_execution_not_verified_from_public_faces',
                    'danger_wide_requires_unobservable_mortar_absence') else 'unsupported_card',
                    item_ids=[i for i in e.get('item_ids', []) if type(i) is int])
                    for e in recorded_advice['mask_exclusions'] if isinstance(e, dict) and isinstance(e.get('item_ids'), list)]
                stopped['mask_exclusions'].extend(
                    dict(reason=e['reason'], kind=e['kind'])
                    for e in recorded_advice['mask_exclusions'] if isinstance(e, dict)
                    and e.get('kind') in ('discard', 'sacrifice')
                    and e.get('reason') == f"ui_kind_unobserved:{e['kind']}")
            if current_record.get('confirmation_retries'):
                stopped['confirmation_retries'] = current_record['confirmation_retries']
        journal.write(stopped)
        print(json.dumps(dict(reason=code, detail=detail), ensure_ascii=False, allow_nan=False), file=sys.stderr)
        # Keep observation failures independent so one broken probe command
        # cannot prevent saving the other artifact.
        for sample in range(1, 4 if code == 'fog_response_ui_unobserved' else 2):
            if sample > 1:
                time.sleep(5)
            suffix = '' if sample == 1 else f'-{sample}'
            password_dialog = mode == 'room'  # unknown screen: protect the input
            try:
                stopped_state = probe.send(dict(kind='state'))['state']
                if mode == 'room':
                    password_dialog = '部屋の合言葉' in stopped_state['body']
                    # Geometry remains useful without recording names/passwords
                    # from the room, input dialog, or a battle's text nodes.
                    stopped_state = dict(body='[redacted]', inputs=[],
                        texts=[dict(text='[redacted]', **{k: r[k] for k in ('x', 'y', 'w', 'h')})
                               for r in stopped_state.get('texts', [])],
                        images=[r for r in stopped_state.get('images', []) if r.get('path') in PUBLIC_PATHS],
                        fogBands=stopped_state.get('fogBands', []),
                        targetArrow=stopped_state.get('targetArrow'))
                destination = journal.directory / f'stopped-state{suffix}.json'
                with open(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'w') as out:
                    json.dump(stopped_state, out, ensure_ascii=False, allow_nan=False)
                    out.write('\n')
            except Exception:
                pass
            try:
                if password_dialog:
                    # Never preserve a plaintext password in the failure image.
                    # The witnessed dialog has one input; fillName fails closed
                    # if this is no longer true. No submit/navigation is attempted.
                    probe.send(dict(kind='fillName', text='[redacted]'))
                    poll_state(probe, lambda s: len(s.get('inputs', [])) == 1
                               and s['inputs'][0].get('value') == '[redacted]', settle_timeout)
                snapshot = probe.send(dict(kind='screenshot'))
                source = Path(snapshot['screenshot'])
                if source.is_file():
                    # Keep the requested failure image; received prefixes and other
                    # probe artifacts live only in the temporary private directory.
                    destination = journal.directory / f'stopped-screenshot{suffix}.png'
                    with open(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as out:
                        with source.open('rb') as inp:
                            shutil.copyfileobj(inp, out)
            except Exception:
                pass
        try:
            if on_stop == 'abandon' and mode == 'training':
                probe.send(dict(kind='clickPoint', x=240, y=23))
                poll_state(probe, lambda s: '戦いをあきらめる' in s['body'])
                probe.send(dict(kind='clickText', text='戦いをあきらめる'))
                poll_state(probe, lambda s: 'コンピュータと対戦' in s['body'])
        except Exception:
            pass
        return dict(counts, status='stopped', reason=code)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['training', 'room'], default='training')
    passwords = parser.add_mutually_exclusive_group()
    passwords.add_argument('--password')
    passwords.add_argument('--password-file', type=Path)
    parser.add_argument('--room-start', choices=['human', 'bot'], default='human')
    parser.add_argument('--room-wait', type=float, default=1800)
    parser.add_argument('--turn-wait', type=float, default=900)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--search', action='store_true')
    from .official_search import add_search_arguments, search_config, search_agent
    add_search_arguments(parser)
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--playwright', type=Path)
    parser.add_argument('--name', default='GFRLBot')
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--settle-timeout', type=float, default=30,
                        help='Seconds to wait for each visible scene/click to settle')
    parser.add_argument('--strict-events', action='store_true', help='Stop on unknown event schemas')
    parser.add_argument('--chance-delay', type=float, default=3, help='Seconds to wait after hit/miss')
    parser.add_argument('--on-stop', choices=['save', 'abandon'], default='save')
    parser.add_argument('--replay', type=Path, help='Offline live1 directory; never starts Node/network')
    args = parser.parse_args(argv)
    try:
        search_config(args)
    except ValueError as error:
        parser.error(str(error))
    if (args.games < 1 or not 0 < args.timeout < float('inf')
            or not 0 < args.settle_timeout < float('inf')
            or not 0 <= args.chance_delay < float('inf')
            or not 0 < args.room_wait < float('inf') or not 0 < args.turn_wait < float('inf')
            or (not args.replay and not args.playwright)):
        parser.error('requires positive limits and --playwright (or --replay)')
    if args.mode == 'room' and (args.on_stop == 'abandon' or args.replay):
        parser.error('room does not support abandon or replay')
    if args.password_file is not None:
        try:
            with open(os.open(args.password_file, os.O_RDONLY | os.O_NOFOLLOW), 'r') as source:
                info = os.fstat(source.fileno())
                require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600,
                        'password_file_permissions')
                args.password = source.read().rstrip('\r\n')
        except Exception:
            parser.error('password file must be a readable regular file with mode 0600')
    if args.mode == 'room' and not args.password:
        parser.error('room requires --password or --password-file')
    probe = journal = scratch = None
    try:
        import torch
        from .actor import Actor
        from .evaluation import load_policy_model, _policy_rng
        from .official_advisor import advise_supported
        torch.set_num_threads(args.threads)
        model = load_policy_model(args.checkpoint, args.device)
        agent = None
        if args.search:
            agent = search_agent(model, args)
        def advisor(result):
            nonlocal agent
            if agent is None:
                agent = Actor(model, result['env'], args.device, sample=not args.greedy)
            return advise_supported(result, agent, search=args.search)
        directory = private_directory(args.out_dir)
        journal = Journal(directory)
        if args.replay:
            from .official_replay import LiveReplayProbe, replay_advice
            probe = LiveReplayProbe(args.replay, directory)
            # Replay all recorded decision positions; model proposals do not
            # cause or explain the human's recorded subsequent updates.
            report = replay_advice(probe, advisor, journal)
        else:
            scratch = tempfile.TemporaryDirectory(prefix='probe-', dir=directory)
            probe = Probe(['node', str(ROOT/'scripts/official_probe.mjs'),
                           '--playwright', str(args.playwright), '--out-dir', scratch.name])
            with _policy_rng(0, (agent,) if agent else ()):
                report = run(probe, advisor, journal, name=args.name, games=args.games,
                             mode=args.mode, password=args.password, dry_run=args.dry_run,
                             timeout=args.timeout, settle_timeout=args.settle_timeout, on_stop=args.on_stop,
                             room_start=args.room_start, room_wait=args.room_wait, turn_wait=args.turn_wait,
                             strict_events=args.strict_events, chance_delay=args.chance_delay)
        print(json.dumps(report, ensure_ascii=False))
        return 1 if report['status'] == 'stopped' else 0
    except Exception:
        print('error:official_bot_initialization_failure')
        return 1
    finally:
        if probe is not None:
            try:
                probe.close()
            except Exception:
                pass
        if journal is not None:
            journal.close()
        if scratch is not None:
            scratch.cleanup()
