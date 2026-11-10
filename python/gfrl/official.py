"""Offline received-game boundary. Only project_game sees privileged wire inventories.

No transport, browser, account, or networking code lives in this module.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np

from . import _core
from .model import OBS_FIELDS

ROOT = Path(__file__).resolve().parents[2]
CURSES = ('none', 'cold', 'fever', 'hell', 'heaven', 'fog', 'dream', 'flash', 'darkcloud')
KINDS = ('play', 'defend', 'discard', 'pray', 'exchange', 'sell', 'buy', 'accept', 'decline', 'sacrifice')
DTYPES = dict(obs_type=np.int8, obs_aux_card=np.int16, obs_valid=bool,
              obs_cards=np.int16, obs_cat=np.int8, obs_num=np.float32)


def parser_module():
    spec = importlib.util.spec_from_file_location('_official_webchannel',
        ROOT / 'third_party/goodfield/scripts/parse_official_webchannel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_games(path, viewer, document_sequence=None):
    """Prefix, parser output, wire fixture, JSON array, or JSONL; retain order."""
    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        try:
            data = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        except (ValueError, UnicodeError):
            data = parser_module().parse_prefix(raw, viewer)
    if isinstance(data, dict):
        data = data.get('games', data.get('updates', [data]))
    if not isinstance(data, list):
        raise ValueError('invalid_record_container')
    sequences = {r.get('document_sequence') for r in data if 'game' in r and 'document_sequence' in r}
    if len(sequences) > 1 and document_sequence is None:
        raise ValueError('multiple_documents_select_sequence')
    if document_sequence is not None and document_sequence not in sequences:
        raise ValueError('document_sequence_not_found')
    # Each yielded object is still privileged. Call project_game immediately.
    for record in data:
        if document_sequence is None or record.get('document_sequence') == document_sequence:
            yield record.get('game', record)


def _integer(x, minimum=0, maximum=9007199254740991):
    if type(x) is not int or not minimum <= x <= maximum:
        raise ValueError('invalid_public_integer')
    return x


def _face(item):
    return _integer(item.get('fakeModelId') or item['modelId'], 1, 296)


def project_game(game, viewer, *, prior=None):
    """Destructive information boundary by construction (no raw references escape).

    Never copy foreign unused items, IDs, lengths, fake flags, or placeholders.
    Own hidden actual models are discarded too; model IDs in use events are
    already revealed by use. Names/identity/transport metadata never cross.
    """
    players = game['players']
    if len(players) != 2 or len({p['id'] for p in players}) != 2:
        raise ValueError('unsupported_roster')
    own = next(p for p in players if p['id'] == viewer)
    ended = bool(game.get('isOver', False))
    fog = 'fog' in own.get('curses', []) and not own.get('dead', False) and not ended
    out = dict(viewer=_integer(viewer), owner=_integer(game['attackTurnPlayerId']),
               gf=_integer(game['gf']), update=_integer(game['updateCount']),
               tiebreak_gf=game.get('tiebreakGF'), is_over=ended, players=[], hand=[],
               actor=None, phase=2 if ended else 0, pending=None, purchase=None,
               events=[], unsupported=[])
    if out['tiebreak_gf'] is not None:
        _integer(out['tiebreak_gf'])
    for p in players:
        pid = _integer(p['id']); dead = bool(p.get('dead', False))
        hidden = fog and pid != viewer and not dead
        if p.get('team', 0) != 0:
            out['unsupported'].append('team_game')
        status = None
        if not hidden:
            curses, seen = [], set()
            for curse in p.get('curses', []):
                CURSES.index(curse)
                if curse in seen:
                    continue
                seen.add(curse)
                if curse in CURSES[1:5]:
                    curses = [c for c in curses if c not in CURSES[1:5]]
                curses.append(curse)
            curses.sort(key=CURSES.index)
            status = dict(hp=_integer(p.get('hp', 40), 0, 99),
                          mp=_integer(p.get('mp', 10), 0, 99),
                          cp=_integer(p.get('cp', 20), 0, 99), curses=curses,
                          guardian=p.get('guardian') or 'none',
                          confused=not dead and bool(p.get('confusionCount', 0)))
        # This is the sole inspection of an opponent's inventory. The predicate
        # reads used only, then transfers face only for the public used pane.
        used = sorted(_face(i) for i in p.get('items', []) if i.get('used') is True)
        out['players'].append(dict(id=pid, dead=dead, fogged=hidden, status=status, used=used))
        if pid == viewer:
            for item in p.get('items', []):
                if item.get('modelId') == 0:
                    continue
                out['hand'].append(dict(id=_integer(item['id']), face=_face(item),
                                        disguised=bool(item.get('fakeModelId')), used=bool(item.get('used', False))))
    if len({h['id'] for h in out['hand']}) != len(out['hand']):
        raise ValueError('ambiguous_own_item_ids')
    if len(out['hand']) > 18:
        out['unsupported'].append('hand_over_capacity')
    # Explicit per-event allowlist. Foreign gift/replacement payloads and their
    # counts are not forwarded, even though the transport can contain them.
    sale_target = (prior or {}).get('pending')
    sale_target = sale_target.get('target') if sale_target else None
    sale_source = (prior or {}).get('pending')
    sale_source = sale_source.get('source') if sale_source else None
    for event in game.get('events', []):
        action = event['action']; e = dict(action=action)
        if action == 'setTargetPlayer':
            sale_target = event['playerId']
            sale_source = game['attackTurnPlayerId']
        if 'playerId' in event:
            e['player'] = _integer(event['playerId'])
        if action == 'sell':
            if sale_target is None or sale_source is None:
                out['unsupported'].append('sale_recipient_unavailable')
            else:
                e.update(buyer=sale_target, seller=sale_source)
                # Commodity contents are released only to the receiving viewer.
                if sale_target == viewer:
                    e['face'] = _face(event['item'])
        elif action in ('gift', 'replaceItems'):
            if event.get('playerId') == viewer:
                e['faces'] = [_face(i) for i in ([event['item']] if action == 'gift' else event.get('items', [])) if i.get('modelId')]
        elif action in ('useAttackItems', 'useDefenseItems', 'removeSomething'):
            e['faces'] = [_face(i) for i in event.get('items', [])]
            # Use is public; do not retain the item's received token or ID.
            e['reveals'] = [dict(face=_face(i), model=i['modelId']) for i in event.get('items', []) if i.get('fakeModelId')]
        elif action in ('nextAttack', 'attackByGuardian'):
            a = event['attack']
            e['attack'] = {k: deepcopy(a[k]) for k in ('playerId', 'targetPlayerId', 'itemModelIds', 'atk', 'guardian', 'reversed') if k in a}
        elif action in ('dealDamage', 'boostHP', 'boostMP', 'exchange'):
            for key in ('damage', 'hp', 'mp', 'cp'):
                if key in event:
                    e[key] = _integer(event[key], -9007199254740991)
        elif action in ('addCurse', 'setCurseOfEverybody'):
            e['curse'] = event['curse']
        elif action == 'removeCurses':
            e['curses'] = list(event.get('curses', []))
        elif action == 'setGuardian':
            e['guardian'] = event['guardian']
        elif action in ('buy', 'canNotBuy', 'setBought', 'useDevilItem'):
            e['model'] = event.get('item', {}).get('modelId', 0)
            if 'bought' in event:
                e['bought'] = event['bought']
        elif action == 'phenomenon':
            e['model'] = event['itemModelId']
        elif action not in ('startGame', 'advanceGF', 'setTargetPlayer', 'disease', 'die', 'endGame',
                             'safe', 'block', 'selfCurse', 'redraw', 'counterAttack', 'reflect', 'bounce', 'danger', 'miss'):
            out['unsupported'].append('unknown_event')
            e = dict(action='unsupported')
        out['events'].append(e)
    attacks = game.get('attacks', [])
    if attacks and not ended:
        if len(attacks) != 1:
            out['unsupported'].append('multiple_pending_frames')
        a = attacks[0]
        models = [_integer(m, 1, 296) for m in a.get('itemModelIds', [])]
        source = _integer(a.get('playerId', 0)); target = a.get('targetPlayerId', 0)
        if not models or not target or source not in [p['id'] for p in players] or target not in [p['id'] for p in players]:
            out['unsupported'].append('ambiguous_pending_actor')
        else:
            out['phase'] = 1; out['actor'] = target
            out['pending'] = dict(origin=source, source=source, target=target, models=models,
                                  guardian=a.get('guardian') or 'none', forced_hit=False,
                                  reflections=0, bounces=0, counter=False, death=False)
            if a.get('atk'):
                out['pending']['base_override'] = _integer(a['atk'])
            if a.get('buyingItemModelId'):
                out['phase'] = 3; out['actor'] = source; out['pending'] = None
                out['purchase'] = dict(origin=source, buyer=source, seller=target,
                                       model=_integer(a['buyingItemModelId'], 1, 296))
            if a.get('reversed'):
                out['unsupported'].append('redirect_requires_history')
    if out['phase'] == 0:
        out['actor'] = out['owner']
    if out['actor'] == viewer and out['phase'] == 0 and own.get('confusionCount', 0):
        out['unsupported'].append('confused_main')
    return out


class OfficialSession:
    """One document/game, one fixed viewer. No wire data is retained."""
    def __init__(self, viewer, history_len=64):
        self.viewer, self.history_len = viewer, history_len
        self.previous = None
        self.events = []
        self.decisions = 0
        self.history_complete = False
        self.pending_meta = None

    def ingest(self, game):
        try:
            prior = self.previous
            if prior and isinstance(game, dict) and game.get('updateCount') == prior['update']:
                prior = self.projection_prior
            public = project_game(game, self.viewer, prior=prior)
            trial = deepcopy(self)
            result = trial.ingest_public(public)
            trial.projection_prior = deepcopy(prior)
            self.__dict__.update(trial.__dict__)
            return result
        except Exception:
            return dict(status='unsupported', unsupported=['invalid_or_unknown_public_state'])

    def ingest_public(self, public):
        p = deepcopy(public)
        if self.previous and p['update'] <= self.previous['update']:
            if p == self.previous.get('_source'):
                return dict(status='duplicate', unsupported=[])
            return dict(status='unsupported', unsupported=['out_of_order_or_conflicting_update'])
        if self.previous and p['update'] != self.previous['update'] + 1:
            self.history_complete = False
            self.pending_meta = None
        if self.previous and any(e['action'] == 'startGame' for e in p['events']):
            return dict(status='unsupported', unsupported=['new_game_requires_new_session'])
        if not self.previous:
            self.history_complete = (any(e['action'] == 'startGame' for e in p['events'])
                                     and not any(e['action'] == 'useAttackItems' for e in p['events']))
        before = deepcopy(self.previous) if self.previous else deepcopy(p)
        before.pop('_source', None)
        # Wire updates can contain multiple accepted decisions. A public ledger
        # retains each use in order, never the private gift generation inputs.
        # Effects are applied below to the event-time visible status panes.
        working = deepcopy(before)
        working['decisions'] = self.decisions
        action = None
        target = before.get('pending', {}).get('target') if before.get('pending') else before['owner']
        actor = before.get('pending', {}).get('target') if before.get('pending') else before['owner']
        owner = before['owner']; gf = before['gf']
        phase = before['phase']; launch = self.pending_meta

        def flush():
            nonlocal action
            if action is None:
                return
            self.decisions += 1
            working['decisions'] = self.decisions
            after = deepcopy(working)
            self.events.append(dict(before=action.pop('before'), after=after,
                                    decision=dict(action, decision=self.decisions)))
            action = None

        def pane(pid):
            return next((x['status'] for x in working['players'] if x['id'] == pid), None)

        for e in p['events']:
            kind = e['action']
            if kind == 'advanceGF':
                flush()
                owner = e['player']; actor = owner
                # Initial advance sets GF1; subsequent advances increment GF.
                gf = 1 if any(x['action'] == 'startGame' for x in p['events']) and not self.previous else gf + 1
                phase = 0
            elif kind in ('useAttackItems', 'useDefenseItems'):
                flush()
                response = kind == 'useDefenseItems'
                actor = target if response else owner
                phase = 1 if response else 0
                faces = e['faces']
                command_kind = 1 if response else (3 if not faces else {3: 4, 4: 5, 5: 6}.get(faces[0], 0))
                hidden = bool(next((x['fogged'] for x in working['players'] if x['id'] == actor), False)) and command_kind in (2, 9)
                action = dict(before=deepcopy(working), gf=gf, phase=phase, source=0,
                              actor=actor, kind=command_kind, faces=[] if hidden else faces,
                              faces_hidden=hidden, target=None, allocation=None)
                # Paid MP and actual disguised effects cannot always be decoded
                # from use markers alone; final panes are never reverse-inferred.
                if e['reveals']:
                    self.history_complete = False
                if not response:
                    launch = dict(origin=actor, source=actor, counter=False, death=False,
                                  reflections=0, bounces=0)
            elif kind == 'setTargetPlayer':
                target = e['player']
                if action and action['phase'] == 0 and action['kind'] in (0, 5, 6):
                    action['target'] = target
            elif kind == 'nextAttack':
                a = e['attack']; actor = a['playerId']
                launch = dict(origin=actor, source=actor, counter=bool(a.get('atk')), death=False,
                              reflections=0, bounces=0)
            elif kind == 'reflect' and launch:
                launch['reflections'] += 1
                launch['source'], target = target, launch['source']
            elif kind == 'bounce' and launch:
                launch['bounces'] += 1
                self.history_complete = False  # target must be explicitly witnessed
            elif kind in ('dealDamage', 'boostHP', 'boostMP'):
                s = pane(e.get('player', target))
                if s is not None:
                    key = 'mp' if kind == 'boostMP' else 'hp'
                    delta = -e['damage'] if kind == 'dealDamage' else e[key]
                    s[key] = min(99, max(0, s[key] + delta))
            elif kind == 'exchange':
                s = pane(actor)
                if s is not None:
                    alloc = {key: e.get(key, 0) for key in ('hp', 'mp', 'cp')}
                    s.update(alloc)
                    if action:
                        action['allocation'] = alloc; action['target'] = actor
            elif kind == 'sell':
                models = before.get('pending', {}).get('models', []) if before.get('pending') else []
                if len(models) >= 2 and models[0] == 4:
                    price = _core.play_catalog()[models[1]]['price']
                    buyer, seller = pane(e.get('buyer')), pane(e.get('seller'))
                    if buyer is not None:
                        debt = price
                        for resource in ('cp', 'mp', 'hp'):
                            paid = min(buyer[resource], debt)
                            buyer[resource] -= paid; debt -= paid
                    if seller is not None:
                        seller['cp'] = min(99, seller['cp'] + price)
                else:
                    self.history_complete = False
            elif kind == 'die':
                for x in working['players']:
                    if x['id'] == e['player']:
                        x['dead'] = True
            elif kind in ('gift', 'replaceItems', 'startGame', 'safe', 'block', 'danger', 'miss', 'counterAttack', 'disease'):
                pass
            elif kind in ('addCurse', 'removeCurses', 'setGuardian', 'setCurseOfEverybody'):
                # Event-time fog changes need a full visibility journal. Do not
                # reconstruct hidden panes using the received end-state truth.
                self.history_complete = False
            elif kind == 'endGame':
                flush()
            else:
                self.history_complete = False
        flush()
        p['decisions'] = self.decisions
        p['last_decision'] = deepcopy(self.events[-1]['decision']) if self.events else None
        if p['pending']:
            pending = p['pending']
            if launch:
                pending.update(launch)
                if 'redirect_requires_history' in p['unsupported']:
                    p['unsupported'].remove('redirect_requires_history')
            if pending['models'][0] in (58, 94, 202) and 'base_override' not in pending:
                p['unsupported'].append('dynamic_attack_value_unavailable')
            if pending['guardian'] in ('earth', 'moon'):
                p['unsupported'].append('earth_moon_payload')
            # DarkCloud is public only when the activation target pane is visible.
            s = next((x['status'] for x in p['players'] if x['id'] == pending['target']), None)
            if pending['reflections'] or pending['bounces']:
                if _core.play_catalog()[pending['models'][0]]['hit_rate']:
                    p['unsupported'].append('redirect_chance_activation_unavailable')
                pending['forced_hit'] = False  # supported deterministic payloads only
            else:
                pending['forced_hit'] = None if s is None else bool(_core.play_catalog()[pending['models'][0]]['hit_rate'] and 'darkcloud' in s['curses'])
        # Status delta failures expose missing event semantics, not private data.
        for x, y in zip(working['players'], p['players']):
            if x['status'] != y['status'] or x['fogged'] != y['fogged']:
                self.history_complete = False
        # Keep only bounded public records. Ring trimming is done in C++ History.
        self.events = self.events[-max(1, self.history_len):]
        self.pending_meta = launch
        self.previous = deepcopy(p)
        self.previous['_source'] = deepcopy(public)
        if p['unsupported']:
            return dict(status='unsupported', public=p, unsupported=p['unsupported'],
                        history_complete=self.history_complete)
        if p['actor'] != self.viewer or p['is_over']:
            return dict(status='finished' if p['is_over'] else 'waiting', public=p,
                        unsupported=p['unsupported'], history_complete=self.history_complete)
        try:
            env = OfficialDecision(p, self.events, self.history_len)
        except (ValueError, RuntimeError, IndexError):
            return dict(status='unsupported', public=p, unsupported=['unsupported_selection'],
                        history_complete=self.history_complete)
        return dict(status='decision', public=p, env=env, unsupported=[],
                    history_complete=self.history_complete)


class OfficialDecision:
    """Actor-compatible immutable observation plus a pure selection cursor.

    step() advances substeps only. It never executes the chosen command.
    """
    num_envs = 1

    def __init__(self, public, events=(), history_len=64):
        self.public = deepcopy(public)
        self.events = deepcopy(list(events))
        self.history_len = history_len
        self.cursor = _core._OfficialCursor(self.public)
        observation = _core._official_observation(self.public, self.events, history_len)
        for key in OBS_FIELDS:
            a = np.asarray(observation[key], dtype=DTYPES[key])[None]
            a.setflags(write=False)
            setattr(self, key, a)
        self._refresh()

    def _refresh(self):
        state = self.cursor.state()
        self.complete = state.pop('complete')
        for key, value in state.items():
            dtype = bool if key == 'mask' else np.int16 if key == 'substep' else np.int8
            setattr(self, key, np.asarray([value], dtype=dtype))

    def step(self, choices):
        self.cursor.choose(int(choices[0])); self._refresh()

    def obs_schema(self):
        return _core.obs_schema(self.history_len)

    def play_view(self, row, seat):
        if row != 0 or self.public['players'][seat]['id'] != self.public['viewer']:
            raise ValueError('only_fixed_viewer_available')
        return _core._official_board(self.public)

    def play_hand(self, row, seat):
        return self.play_view(row, seat)['hand']

    def action_support(self):
        """Execution gaps detectable without inspecting any hidden actual model."""
        c = self.cursor.command()
        selected = [self.public['hand'][i] for i in c['slots']]
        faces = [h['face'] for h in selected]
        reasons = []
        if 94 in faces and 232 in faces:
            reasons.append('danger_wide_requires_unobservable_mortar_absence')
        if any(h['disguised'] for h in selected):
            reasons.append('dream_actual_execution_not_verified_from_public_faces')
        if any(p['status'] and p['status']['guardian'] in ('earth', 'moon') for p in self.public['players']):
            reasons.append('earth_moon_general_execution')
        return reasons

    def semantic_action(self):
        c = self.cursor.command()
        selected = [self.public['hand'][slot] for slot in c['slots']]
        operations = [dict(op='kind', kind=KINDS[c['kind']])]
        operations += [dict(op='select_card', model=h['face'], item_id=h['id']) for h in selected]
        operations.append(dict(op='finish_cards'))
        operations.append(dict(op='target', player_id=c['target']))
        if c['allocation']:
            operations.append(dict(op='allocate', **c['allocation']))
        operations.append(dict(op='confirm'))
        return dict(kind=KINDS[c['kind']], cards=[dict(model=h['face'], item_id=h['id']) for h in selected],
                    target=c['target'], allocation=c['allocation'], operations=operations,
                    expected_update=self.public['update'], expected_gf=self.public['gf'],
                    viewer=self.public['viewer'])


class OfficialStreams:
    """Merge cumulative prefixes from every probe stream into one session.

    Parse only changed stream snapshots, ingest each update once in ascending
    order. Cache only public projections, hashes and stream byte boundaries.
    A baseline taken before starting a new game excludes older complete frames.
    Never skip malformed/unknown streams and continue with a subset.
    """
    def __init__(self, session, baseline=()):
        self.session = session
        self.parser = parser_module()
        self.baseline = {self.stream_key(c['file']): c['bytes'] for c in baseline}
        self.hashes = {}
        self.seen = {}
        self.context = {}
        self.latest = None
        self.failed = False

    @staticmethod
    def stream_key(file):
        import re
        match = re.search(r'^(.*)-\d+-stream-(\d+)-prefix\.bin$', Path(file).name)
        return match.groups() if match else (str(file), '')

    def poll(self, captures):
        if self.failed:
            return dict(status='unsupported', unsupported=['stream_reader_stopped'])
        try:
            games = []
            for c in captures:
                if c.get('capture_error'):
                    raise ValueError('capture_failed')
                key = self.stream_key(c['file'])
                if self.hashes.get(key) == c.get('sha256') and c.get('sha256'):
                    continue
                raw = Path(c['file']).read_bytes()
                import hashlib
                digest = hashlib.sha256(raw).hexdigest()
                if len(raw) != c['bytes'] or (c.get('sha256') and digest != c['sha256']):
                    raise ValueError('capture_integrity')
                records = self.parser.parse_prefix(raw, self.session.viewer)['games']
                if len({r['document_sequence'] for r in records if r['frame_end_byte'] > self.baseline.get(key, 0)}) > 1:
                    raise ValueError('multiple_documents')
                games += [r['game'] for r in records if r['frame_end_byte'] > self.baseline.get(key, 0)]
                self.hashes[key] = digest
            for game in sorted(games, key=lambda g: g['updateCount']):
                u = _integer(game['updateCount'])
                if u in self.seen:
                    projected = project_game(game, self.session.viewer, prior=self.context[u])
                    if projected != self.seen[u]:
                        raise ValueError('conflicting_update')
                    continue
                prior = self.session.previous
                projected = project_game(game, self.session.viewer, prior=prior)
                result = self.session.ingest(game)
                if result['status'] == 'unsupported':
                    self.failed = True
                    return result
                self.context[u] = dict(pending=deepcopy(prior.get('pending'))) if prior else None
                self.seen[u] = projected
                self.latest = result
            return self.latest or dict(status='waiting', unsupported=[])
        except Exception:
            # Parser exceptions and hostile schema never include raw diagnostics.
            self.failed = True
            return dict(status='unsupported', unsupported=['invalid_or_unknown_stream'])
