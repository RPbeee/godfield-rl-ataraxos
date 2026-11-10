"""Training UI bot. Importing this module never starts a browser or network."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import tempfile
import time

from .official import ROOT, OfficialSession, OfficialStreams
from .official_ui import UIUnsupported, require, prepare, execute, ready, wait_visible


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


def enter_training(probe, name):
    state = poll_state(probe, lambda s: '生誕する' in s['body'] or 'コンピュータと対戦' in s['body'] or settings(s))
    if '生誕する' in state['body']:
        probe.send(dict(kind='fillName', text=name))
        probe.send(dict(kind='clickText', text='生誕する'))
        state = poll_state(probe, lambda s: 'コンピュータと対戦' in s['body'])
    if not settings(state):
        probe.send(dict(kind='clickText', text='修行'))
    return poll_state(probe, settings)


def enter_room(probe, name, mode, password=None):
    require(mode == 'training', 'room_mode_not_implemented')
    return enter_training(probe, name)


def capture(probe):
    result = probe.send(dict(kind='capture'))
    return result['capture']['captures']


def accepted(probe, streams, update, timeout=20):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        result = streams.poll(capture(probe))
        require(result['status'] != 'unsupported', 'unsupported_receive')
        if result.get('public', {}).get('update', update) > update:
            return result
        time.sleep(.2)
    raise UIUnsupported('acceptance_timeout')


def summary(public):
    return {k: public[k] for k in ('update', 'gf', 'phase', 'actor', 'viewer', 'is_over')}


def run(probe, advisor, journal, *, name, games=1, mode='training', password=None,
        dry_run=False, timeout=20, settle_timeout=30, on_stop='save'):
    counts = dict(games=0, wins=0, losses=0, draws=0)
    current_record = None
    successful_kinds = set()
    try:
        enter_room(probe, name, mode, password)
        for _ in range(games):
            baseline = capture(probe)
            probe.send(dict(kind='clickText', text='戦いを始める'))
            poll_state(probe, lambda s: 'G.F.' in s['body'])
            # The parser infers the unique human seat in training; the bot checks
            # the same roster before constructing the fixed-viewer session.
            viewer = getattr(probe, 'viewer', None)
            if viewer is None:
                records = capture(probe)
                parser = __import__('gfrl.official', fromlist=['parser_module']).parser_module()
                humans = set()
                for c in records:
                    for r in parser.parse_prefix(Path(c['file']).read_bytes())['games']:
                        if r['frame_end_byte'] > {OfficialStreams.stream_key(x['file']): x['bytes'] for x in baseline}.get(OfficialStreams.stream_key(c['file']), 0):
                            humans.update(p['id'] for p in r['game']['players'] if not p.get('isBot', False))
                require(len(humans) == 1, 'viewer_not_unique')
                viewer = humans.pop()
            streams = OfficialStreams(OfficialSession(viewer), baseline)
            result = accepted(probe, streams, -1, timeout)
            while True:
                require(result['status'] != 'unsupported', 'unsupported_receive')
                public = result['public']
                if result['status'] == 'finished':
                    own = next(p for p in public['players'] if p['id'] == viewer)
                    other = next(p for p in public['players'] if p['id'] != viewer)
                    outcome = 'draws' if own['dead'] == other['dead'] else 'losses' if own['dead'] else 'wins'
                    counts['games'] += 1; counts[outcome] += 1
                    journal.write(dict(status='finished', state=summary(public), outcome=outcome))
                    poll_state(probe, lambda s: '戦いを終わる' in s['body'])
                    probe.send(dict(kind='clickText', text='戦いを終わる'))
                    poll_state(probe, settings)
                    break
                if result['status'] != 'decision':
                    result = accepted(probe, streams, public['update'], timeout)
                    continue
                advice = advisor(result)
                current_record = dict(state=summary(public), advice=advice, clicks=[], accepted=False)
                require(advice['status'] == 'advice', 'unsupported_advice')
                def refresh():
                    latest = streams.poll(capture(probe))
                    require(latest['status'] != 'unsupported', 'unsupported_receive')
                    if latest.get('public', {}).get('update', public['update']) > public['update']:
                        raise ReceiveAdvanced(latest)

                try:
                    state = probe.send(dict(kind='state'))['state']
                    state = wait_visible(probe, state, lambda s: ready(s, public, name),
                                         timeout=settle_timeout, refresh=refresh)
                    plan = prepare(state, public, advice['action'], name)
                    if not dry_run:
                        execute(probe, state, public, advice['action'], name,
                                commands=current_record['clicks'], settle_timeout=settle_timeout,
                                refresh=refresh)
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
                result = accepted(probe, streams, public['update'], timeout)
                current_record.update(status='accepted', accepted=True, received_update=result['public']['update'])
                kind = advice['action']['kind']
                if kind not in successful_kinds:
                    current_record['first_success_kind'] = kind
                    successful_kinds.add(kind)
                journal.write(current_record); current_record = None
        return dict(counts, status='complete')
    except Exception as error:
        code = str(error) if isinstance(error, UIUnsupported) else 'bot_input_or_model_failure'
        if current_record is not None:
            current_record.update(status='stopped', reason=code); journal.write(current_record)
        journal.write(dict(status='stopped', reason=code))
        # Keep observation failures independent so one broken probe command
        # cannot prevent saving the other artifact.
        try:
            stopped_state = probe.send(dict(kind='state'))['state']
            destination = journal.directory / 'stopped-state.json'
            with open(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'w') as out:
                json.dump(stopped_state, out, ensure_ascii=False, allow_nan=False)
                out.write('\n')
        except Exception:
            pass
        try:
            snapshot = probe.send(dict(kind='screenshot'))
            source = Path(snapshot['screenshot'])
            if source.is_file():
                # Keep the requested failure image; received prefixes and other
                # probe artifacts live only in the temporary private directory.
                destination = journal.directory / 'stopped-screenshot.png'
                with open(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as out:
                    with source.open('rb') as inp:
                        shutil.copyfileobj(inp, out)
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
    parser.add_argument('--password')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--playwright', type=Path)
    parser.add_argument('--name', default='GFRLBot')
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--settle-timeout', type=float, default=30,
                        help='Seconds to wait for each visible scene/click to settle')
    parser.add_argument('--on-stop', choices=['save', 'abandon'], default='save')
    parser.add_argument('--replay', type=Path, help='Offline live1 directory; never starts Node/network')
    args = parser.parse_args(argv)
    if (args.games < 1 or not 0 < args.timeout < float('inf')
            or not 0 < args.settle_timeout < float('inf') or args.mode != 'training'
            or (not args.replay and not args.playwright)):
        parser.error('training requires positive limits and --playwright (or --replay)')
    probe = journal = scratch = None
    try:
        import torch
        from .actor import Actor
        from .evaluation import load_policy_model, _policy_rng
        from .official_advisor import advise
        torch.set_num_threads(1)
        model = load_policy_model(args.checkpoint, 'cpu')
        agent = None
        if args.search:
            from .search import SearchAgent, SearchConfig, load_search_belief
            agent = SearchAgent(model, load_search_belief('uniform', None, 'cpu'), 'cpu',
                                config=SearchConfig(select='argmax' if args.greedy else 'sample'), seed=0, num_threads=1)
            agent.actor.sample = not args.greedy
        def advisor(result):
            nonlocal agent
            if agent is None:
                agent = Actor(model, result['env'], 'cpu', sample=not args.greedy)
            return advise(result, agent, search=args.search)
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
            probe = Probe(['node', str(ROOT/'third_party/goodfield/scripts/official_headless_probe.mjs'),
                           '--playwright', str(args.playwright), '--out-dir', scratch.name])
            with _policy_rng(0, (agent,) if agent else ()):
                report = run(probe, advisor, journal, name=args.name, games=args.games,
                             mode=args.mode, password=args.password, dry_run=args.dry_run,
                             timeout=args.timeout, settle_timeout=args.settle_timeout, on_stop=args.on_stop)
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
