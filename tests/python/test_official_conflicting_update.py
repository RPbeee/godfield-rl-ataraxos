"""Offline public-only conflict diagnostics and snapshot ownership."""
from copy import deepcopy
import json

import pytest

from gfrl.official import (OfficialSession, OfficialStreams, conflict_detail,
                           exception_detail, projection_differences)
from gfrl.official_bot import Journal, private_directory, run
from test_official_bot import GAMES, FakeProbe, advisor, capture_file


def test_conflicting_resend_paths_only(tmp_path):
    first = deepcopy(GAMES[0])
    new = deepcopy(first)
    new['players'][0]['hp'] -= 1
    new['players'][1]['name'] = 'PRIVATE_NAME'
    new['players'][1]['items'] = [dict(id=987654321, modelId=296)]
    streams = OfficialStreams(OfficialSession(1))
    streams.poll([capture_file(tmp_path, [first])])
    detail = streams.poll([capture_file(tmp_path, [new], seq=2)])['detail']
    assert detail == dict(error='ValueError', code='conflicting_update', update=1,
                         differences=[dict(path='players[0].status.hp', kind='changed')])
    assert streams.poll([])['detail'] == detail
    assert all(s not in json.dumps(detail) for s in ('PRIVATE_NAME', '987654321', '296'))


def test_identical_resends_keep_original_context_and_snapshots(tmp_path):
    streams = OfficialStreams(OfficialSession(1))
    streams.poll([capture_file(tmp_path, GAMES)])
    snapshot = deepcopy(streams.seen)
    # Results and session are mutable, but neither may change cached projections.
    streams.latest['public']['players'][0]['status']['hp'] = 0
    streams.session.previous['hand'].clear()
    assert streams.seen == snapshot
    assert streams.poll([capture_file(tmp_path, GAMES, seq=2)])['status'] == 'finished'
    assert not streams.failed
    assert streams.seen == snapshot


def test_difference_markers_and_bound_use_only_public_keys():
    a = dict(players=[dict(status=dict(hp=40, mp=10))], pending=None, events=[])
    b = dict(players=[dict(status=dict(hp=39, cp=20))], pending={}, events=[dict(action='pray')])
    assert projection_differences(a, b) == [
        dict(path='events[0]', kind='added'), dict(path='pending', kind='type'),
        dict(path='players[0].status.cp', kind='added'),
        dict(path='players[0].status.hp', kind='changed'),
        dict(path='players[0].status.mp', kind='removed')]
    assert projection_differences(dict(events=[0]*100), dict(events=[1]*100)) == [
        dict(path=f'events[{i}]', kind='changed') for i in range(32)]


@pytest.mark.parametrize('differences', [None, 'private text', [dict(path=12, kind='changed')],
    [dict(path='events[0]', kind='private text')], [dict(path='events[0]', kind='changed', value='private')],
    [dict(path='events[0]', kind='changed')]*33])
def test_conflict_detail_rejects_invalid_fields(differences):
    payload = dict(update=True, differences=differences, raw='PRIVATE')
    assert conflict_detail(payload) == {}
    assert exception_detail(ValueError('conflicting_update', payload)) == dict(
        error='ValueError', code='conflicting_update')


def test_conflict_survives_stop_journal_stderr_and_final_summary(tmp_path, capsys):
    class ConflictingProbe(FakeProbe):
        def send(self, command):
            if command['kind'] == 'capture' and self.phase == 'battle':
                changed = deepcopy(self.games[0])
                changed['players'][0]['hp'] -= 1
                return dict(capture=dict(captures=[capture_file(
                    self.directory, [self.games[0], changed])]))
            return super().send(command)
    directory = private_directory(tmp_path/'log')
    journal = Journal(directory)
    report = run(ConflictingProbe(tmp_path), advisor, journal, name='SyntheticName', settle_timeout=0)
    journal.close()
    row = json.loads((directory/'moves.jsonl').read_text())
    assert report['status'] == row['status'] == 'stopped'
    assert report['reason'] == row['reason'] == 'unsupported_receive'
    assert report['detail'] == row['detail'] == dict(error='ValueError', code='conflicting_update',
        update=1, differences=[dict(path='players[0].status.hp', kind='changed')])
    assert json.loads(capsys.readouterr().err)['detail'] == report['detail']
