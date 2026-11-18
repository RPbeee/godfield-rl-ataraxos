"""Offline live1 replay. No browser, subprocess, network or source file writes.

Legacy probe states lacked text geometry. This specific witnessed recording's
own name bands are reconstructed explicitly (left,left,right as observed), not
inferred by the live bot. Replay proposals are never claimed to be accepted.
"""
from copy import deepcopy
import json
from pathlib import Path

from .official import OfficialSession, OfficialStreams
from .official_ui import prepare, UIUnsupported


class LiveReplayProbe:
    # Response offsets and own pane centers verified against live1 screenshots.
    POSITIONS = {1: (8, 350), 2: (37, 690), 3: (40, 350), 4: (48, 350), 5: (58, 690), 6: (64, 690)}

    def __init__(self, source, directory=None):
        self.source = Path(source)
        self.responses = [json.loads(line) for line in (self.source/'resp.jsonl').read_text().splitlines()]
        self.captures = [r['capture']['captures'] for r in self.responses if 'capture' in r]
        self.viewer = 1

    def state_for(self, public):
        offset, x = self.POSITIONS[public['update']]
        state = deepcopy(self.responses[offset]['state'])
        # The recording has a footer name just after GF; do not store it.
        name = state['body'].split('G.F.', 1)[1].splitlines()[1]
        state['texts'] = [dict(text=name, x=x-30, y=60, w=60, h=20)]
        # u3/u4 states were captured after the first selection; construct the
        # corresponding unselected pane without touching the recorded hand.
        state['images'] = [r for r in state['images'] if not (r['path'].startswith('/images/items/')
                           and r['x'] == x-145 and 103 <= r['y'] <= 403)]
        return state, name

    def close(self):
        pass


def replay_advice(probe, advisor, journal):
    streams = OfficialStreams(OfficialSession(probe.viewer))
    shown = set(); proposals = supported = 0
    for captures in probe.captures:
        result = streams.poll(captures)
        if result['status'] == 'unsupported':
            return dict(status='stopped', reason='unsupported_replay_receive')
        public = result.get('public')
        if not public or public['update'] in shown:
            continue
        shown.add(public['update'])
        if result['status'] != 'decision':
            continue
        advice = advisor(result); proposals += 1
        row = dict(status='replay_proposal', update=public['update'], advice=advice,
                   accepted=None, recorded_followup_is_not_acceptance=True)
        state, name = probe.state_for(public)
        try:
            if advice['status'] != 'advice':
                raise UIUnsupported('unsupported_advice')
            row['planned_clicks'] = prepare(state, public, advice['action'], name,
                allow_dream=advice.get('dream_played') is True,
                dream_last_resort=advice.get('dream_last_resort') is True)['commands']
            supported += 1
        except UIUnsupported as error:
            row['ui_unsupported'] = str(error)
        journal.write(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'advice'} | {
            'explanation': advice.get('explanation'), 'action': advice.get('action')}, ensure_ascii=False))
    return dict(status='replay_complete', positions=proposals, convertible=supported,
                final_status=streams.latest['status'])
