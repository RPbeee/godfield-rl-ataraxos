"""Offline received-game boundary. Only project_game sees privileged wire inventories.

No transport, browser, account, or networking code lives in this module.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from . import _core
from .model import OBS_FIELDS

ROOT = Path(__file__).resolve().parents[2]
CURSES = ('none', 'cold', 'fever', 'hell', 'heaven', 'fog', 'dream', 'flash', 'darkcloud')
KINDS = ('play', 'defend', 'discard', 'pray', 'exchange', 'sell', 'buy', 'accept', 'decline', 'sacrifice')
OBSERVED_UI_KINDS = frozenset(('play', 'defend', 'exchange', 'sell', 'buy', 'pray', 'accept', 'decline'))
DTYPES = dict(obs_type=np.int8, obs_aux_card=np.int16, obs_valid=bool,
              obs_cards=np.int16, obs_cat=np.int8, obs_num=np.float32)


def parser_module():
    spec = importlib.util.spec_from_file_location('_official_webchannel',
        ROOT / 'third_party/goodfield/scripts/parse_official_webchannel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Messages are never serialized: only codes owned by this boundary are allowed.
VALUE_ERROR_CODES = frozenset({
    'capture_failed', 'capture_integrity', 'multiple_documents', 'conflicting_update',
    'invalid_public_integer', 'unsupported_roster', 'ambiguous_own_item_ids',
})

# Only mask/observation construction needs these facts. All other reasons
# (including unknown future codes) remain fatal at the public-state boundary.
DECISION_ONLY_UNSUPPORTED = frozenset({
    'hand_over_capacity', 'confused_main', 'redirect_requires_history',
    'dynamic_attack_value_unavailable', 'earth_moon_payload',
    'redirect_chance_activation_unavailable',
    'dying_players_decision_unavailable',
    'dynamic_hit_rate_unavailable', 'death_attack_decision_unavailable',
})


def projection_differences(first, new):
    """At most 32 structural differences; callers supply public projections only.

    Never stringify leaves (including IDs, card faces, or event text).
    """
    differences = []
    def add(path, kind):
        differences.append(dict(path=path, kind=kind))
    def walk(a, b, path):
        if len(differences) >= 32:
            return
        if type(a) is not type(b):
            add(path, 'type')
        elif isinstance(a, dict):
            for key in sorted(a.keys() | b.keys()):
                if len(differences) >= 32:
                    break
                child = f'{path}.{key}' if path else key
                if key not in a:
                    add(child, 'added')
                elif key not in b:
                    add(child, 'removed')
                else:
                    walk(a[key], b[key], child)
        elif isinstance(a, list):
            for i in range(max(len(a), len(b))):
                if len(differences) >= 32:
                    break
                child = f'{path}[{i}]'
                if i >= len(a):
                    add(child, 'added')
                elif i >= len(b):
                    add(child, 'removed')
                else:
                    walk(a[i], b[i], child)
        elif a != b:
            add(path, 'changed')
    walk(first, new, '')
    return differences


def conflict_detail(detail):
    """Whitelist the conflict diagnostic shape at exception/log boundaries."""
    safe = {}
    if not isinstance(detail, dict):
        return safe
    update = detail.get('update')
    if type(update) is int and 0 <= update <= 9007199254740991:
        safe['update'] = update
    differences = detail.get('differences')
    if (isinstance(differences, list) and len(differences) <= 32
            and all(isinstance(d, dict) and set(d) == {'path', 'kind'}
                    and isinstance(d['path'], str) and len(d['path']) <= 256
                    and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*(?:\[\d+\]|\.[A-Za-z_][A-Za-z_0-9]*)*', d['path'])
                    and isinstance(d['kind'], str)
                    and d['kind'] in ('added', 'removed', 'changed', 'type')
                    for d in differences)):
        safe['differences'] = deepcopy(differences)
    return safe


def exception_detail(error, parse_error_type=None, *, fixed_codes=True, local_only=True):
    detail = dict(error=type(error).__name__)
    if parse_error_type is not None and isinstance(error, parse_error_type):
        detail.update(code=str(error), parser=error.detail)
    elif (fixed_codes and type(error) is ValueError and error.args
          and isinstance(error.args[0], str) and error.args[0] in VALUE_ERROR_CODES):
        detail['code'] = error.args[0]
        if detail['code'] == 'conflicting_update' and len(error.args) == 2:
            detail.update(conflict_detail(error.args[1]))
    else:
        frame = error.__traceback__
        last = None
        while frame is not None:
            file = Path(frame.tb_frame.f_code.co_filename).resolve()
            if (not local_only or file.is_relative_to(ROOT / 'python/gfrl')
                    or file.is_relative_to(ROOT / 'third_party/goodfield')):
                last = dict(file=file.name, line=frame.tb_lineno)
            frame = frame.tb_next
        if last:
            detail.update(last)
    return detail


def unsupported_detail(public):
    """Use only the projection, which has already discarded private inventories."""
    facts = public.get('detail', [])
    pending = public.get('pending') or {}
    result = [deepcopy(d) for d in facts if d['code'] == 'multiple_pending_frames']
    for code in public['unsupported']:
        matches = [d for d in facts if d['code'] == code]
        if matches:
            result.extend(matches)
        else:
            d = dict(code=code, phase=public['phase'])
            if pending:
                d.update(models=list(pending['models']), guardian=pending['guardian'],
                         reflections=pending['reflections'], bounces=pending['bounces'])
            result.append(d)
    return result


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
               events=[], unsupported=[], detail=[])
    if out['tiebreak_gf'] is not None:
        _integer(out['tiebreak_gf'])
    for p in players:
        pid = _integer(p['id']); dead = bool(p.get('dead', False))
        hidden = fog and pid != viewer and not dead
        if p.get('team', 0) != 0:
            out['unsupported'].append('team_game')
            out['detail'].append(dict(code='team_game', team_present=True))
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
        out['detail'].append(dict(code='hand_over_capacity', count=len(out['hand']), capacity=18))
    # Explicit per-event allowlist. Foreign gift/replacement payloads and their
    # counts are not forwarded, even though the transport can contain them.
    sale_target = (prior or {}).get('pending')
    sale_target = sale_target.get('target') if sale_target else None
    sale_source = (prior or {}).get('pending')
    sale_source = sale_source.get('source') if sale_source else None
    for event in game.get('events', []):
        action = event['action']; e = dict(action=action)
        if action == 'unknownEvent':
            out['events'].append(dict(action='unknownEvent'))
            continue
        if action == 'setTargetPlayer':
            sale_target = event['playerId']
            sale_source = game['attackTurnPlayerId']
        if 'playerId' in event:
            e['player'] = _integer(event['playerId'])
        if action == 'sell':
            if sale_target is None or sale_source is None:
                out['unsupported'].append('sale_recipient_unavailable')
                out['detail'].append(dict(code='sale_recipient_unavailable',
                    target_missing=sale_target is None, source_missing=sale_source is None))
            else:
                e.update(buyer=sale_target, seller=sale_source)
                # Commodity contents are released only to the receiving viewer.
                if sale_target == viewer:
                    e['face'] = _face(event['item'])
                if event.get('overflowItem') is not None:
                    item = event['overflowItem']
                    # Fanout of foreign unused eviction truth is unverified.
                    e['overflow_used'] = item['used']
                    e['overflow_unobserved'] = sale_target != viewer and not item['used']
                    if not e['overflow_unobserved'] and not item.get('fakeModelId'):
                        e['overflow_model'] = _face(item)
                    if item.get('fakeModelId'):
                        e['overflow_unobserved'] = True
        elif action == 'revive':
            # ks.k displays the consumed item (including a fake flip) to all.
            # No instance ID or other inventory entry crosses this boundary.
            item = event['item']
            e.update(model=_integer(item['modelId'], 1, 296), face=_face(item))
        elif action in ('counterAttack', 'selfCurse'):
            if 'index' in event:
                e['index'] = _integer(event['index'], 0, 17)
        elif action in ('gift', 'replaceItems'):
            if event.get('playerId') == viewer:
                e['faces'] = [_face(i) for i in ([event['item']] if action == 'gift' else event.get('items', [])) if i.get('modelId')]
        elif action in ('useAttackItems', 'useDefenseItems', 'removeSomething'):
            e['faces'] = [_face(i) for i in event.get('items', [])]
            # Use is public; do not retain the item's received token or ID.
            e['reveals'] = [dict(face=_face(i), model=i['modelId']) for i in event.get('items', []) if i.get('fakeModelId')]
        elif action in ('removeItems', 'removeUsedMiracles'):
            # Removal displays and flips each card publicly, including foreign
            # unused cards. Never copy the rest of that player's inventory.
            e['models'] = [_integer(i['modelId'], 1, 296) for i in event['items']]
            e['faces'] = [_face(i) for i in event['items']]
        elif action in ('nextAttack', 'attackByGuardian'):
            a = event['attack']
            e['attack'] = {k: deepcopy(a[k]) for k in ('playerId', 'targetPlayerId', 'itemModelIds', 'atk', 'guardian', 'reversed') if k in a}
        elif action in ('dealDamage', 'dealDarkDamage', 'boostHP', 'boostMP', 'exchange', 'takeCP', 'boostCP', 'boostCPOfEverybody'):
            for key in ('damage', 'hp', 'mp', 'cp'):
                if key in event:
                    e[key] = _integer(event[key], -9007199254740991)
        elif action in ('addCurse', 'setCurseOfEverybody', 'upgradeDisease'):
            e['curse'] = event['curse']
            if 'natural' in event:
                e['natural'] = event['natural']
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
                             'safe', 'block', 'upgradeHeaven', 'collectCPOfEverybody', 'redraw', 'reflect', 'bounce', 'danger', 'miss', 'hit', 'removeGuardian', 'pray'):
            out['unsupported'].append('unknown_event')
            if isinstance(action, str) and action in parser_module().GAME_STRINGS:
                out['detail'].append(dict(code='unknown_event', action=action))
            e = dict(action='unsupported')
        out['events'].append(e)
    attacks = game.get('attacks', [])
    if attacks and not ended:
        if len(attacks) != 1:
            # official-state-view.md, v48: the client's fresh active pane uses
            # only attacks[0]. nextAttack supplies subsequent active payloads;
            # do not infer execution/history from the remaining queue.
            guardians = ('none', 'mercury', 'venus', 'earth', 'moon',
                         'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto')
            out['detail'].append(dict(code='multiple_pending_frames', count=len(attacks),
                frames=[dict(playerId=_integer(a.get('playerId', 0)),
                             targetPlayerId=None if a.get('targetPlayerId') is None else _integer(a['targetPlayerId']),
                             itemModelIds=[_integer(m, 1, 296) for m in a.get('itemModelIds', [])],
                             has_atk='atk' in a,
                             guardian=a.get('guardian') if a.get('guardian') in guardians else 'none')
                        for a in attacks]))
        a = attacks[0]
        models = [_integer(m, 1, 296) for m in a.get('itemModelIds', [])]
        source = _integer(a.get('playerId', 0)); target = a.get('targetPlayerId')
        if not models or target is None or source not in [p['id'] for p in players] or target not in [p['id'] for p in players]:
            out['unsupported'].append('ambiguous_pending_actor')
            out['detail'].append(dict(code='ambiguous_pending_actor',
                models=models, models_empty=not models, target_missing=not bool(target),
                source_in_roster=source in [p['id'] for p in players],
                target_in_roster=target in [p['id'] for p in players]))
        else:
            out['phase'] = 1; out['actor'] = target
            out['pending'] = dict(origin=source, source=source, target=target, models=models,
                                  guardian=a.get('guardian') or 'none', forced_hit=False,
                                  reflections=0, bounces=0, counter=181 <= models[0] <= 188, death=False)
            # Trade values, resource counters and 202's signed HP delta
            # are not attack power: preserve native zero-power responses.
            if 'atk' in a and models[0] not in (4, 5, 187, 188, 202):
                out['pending']['base_override'] = _integer(a['atk'])
            if models[0] in (187, 188, 202):
                out['pending']['power'] = 0
                if 'atk' in a:
                    out['pending']['amount'] = _integer(a['atk'], -9007199254740991)
            if models[0] in (4, 5):
                out['pending']['power'] = 0
            # p6/fu/gf2 display a positive DTO rate, otherwise the catalog rate.
            # The Match-view adapter reconstructs catalog rates only. A display
            # override is not proof of a native activation/death/redirect state.
            catalog_rate = _core.play_catalog()[models[0]]['hit_rate']
            rate = _integer(a['hitRate'], 0, 100) if 'hitRate' in a else 0
            if rate > 0 and rate != catalog_rate:
                out['unsupported'].append('dynamic_hit_rate_unavailable')
                out['detail'].append(dict(code='dynamic_hit_rate_unavailable',
                    models=models, hit_rate=rate, catalog_hit_rate=catalog_rate,
                    reason='native_pending_uses_catalog_hit_rate'))
            # Native death bows carry both rate=75 and death=true. Neither a
            # 30-power DTO nor a 75% label establishes that execution provenance.
            if models[0] == 112 and (a.get('atk', 0) == 30 or rate == 75):
                out['unsupported'].append('death_attack_decision_unavailable')
                out['detail'].append(dict(code='death_attack_decision_unavailable',
                    models=models, reason='native_death_state_not_reconstructible'))
            if a.get('buyingItemModelId'):
                out['phase'] = 3; out['actor'] = source; out['pending'] = None
                out['purchase'] = dict(origin=source, buyer=source, seller=target,
                                       model=_integer(a['buyingItemModelId'], 1, 296))
            if a.get('reversed'):
                out['unsupported'].append('redirect_requires_history')
    if out['phase'] == 0:
        out['actor'] = out['owner']
    # Pinned v2 client tn ignores this server field. It is public roster
    # metadata, not a dead flag or evidence of a particular 112/208 schedule.
    # Native PendingView.death/hit_rate and the unresolved death queue cannot
    # be reconstructed from IDs alone. Never manufacture an observation/mask.
    dying = game.get('dyingPlayerIds', [])
    if dying:
        out['dying_players'] = [_integer(pid) for pid in dying]
        out['unsupported'].append('dying_players_decision_unavailable')
        out['detail'].append(dict(code='dying_players_decision_unavailable',
            phase=out['phase'], dying_players=list(out['dying_players']),
            reason='native_death_state_not_reconstructible'))
    if out['actor'] == viewer and out['phase'] == 0 and own.get('confusionCount', 0):
        out['unsupported'].append('confused_main')
        out['detail'].append(dict(code='confused_main', phase=0, own_confused=True))
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
        except Exception as error:
            return dict(status='unsupported', unsupported=['invalid_or_unknown_public_state'],
                        detail=exception_detail(error))

    def ingest_public(self, public):
        p = deepcopy(public)
        # Proofs are session-owned and must be established again from this
        # update, even if a caller builds a synthetic update from prior output.
        p.pop('_search_launch', None)
        p.pop('_search_fallback', None)
        public_source = deepcopy(p)
        if self.previous and p['update'] <= self.previous['update']:
            if p == self.previous.get('_source'):
                return dict(status='duplicate', unsupported=[])
            return dict(status='unsupported', unsupported=['out_of_order_or_conflicting_update'],
                        detail=dict(code='out_of_order_or_conflicting_update', older=p['update'] < self.previous['update']))
        if self.previous and p['update'] != self.previous['update'] + 1:
            self.history_complete = False
            self.pending_meta = None
        if self.previous and any(e['action'] == 'startGame' for e in p['events']):
            return dict(status='unsupported', unsupported=['new_game_requires_new_session'],
                        detail=dict(code='new_game_requires_new_session', start_game=True))
        if not self.previous:
            self.history_complete = (any(e['action'] == 'startGame' for e in p['events'])
                                     and not any(e['action'] == 'useAttackItems' for e in p['events']))
        if any(e['action'] == 'unknownEvent' for e in p['events']):
            self.history_complete = False
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
        removal_model = (before.get('pending') or {}).get('models', [None])[0]

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
                    working.pop('removal', None)
                    working.pop('sale', None)
                    working['guardian_changes'] = []
                    removal_model = faces[0] if faces else None
                    launch = dict(origin=actor, source=actor, counter=False, death=False,
                                  reflections=0, bounces=0)
                    # The final received MP has already been spent. Only an
                    # observed, contiguous launch can establish pre-spend MP.
                    if (faces and faces[0] == 58 and not e['reveals'] and self.previous
                            and p['update'] == self.previous['update'] + 1 and pane(actor) is not None
                            and not any(x['action'] in ('unknownEvent', 'unsupported') for x in p['events'])):
                        remaining_mp = pane(actor)['mp'] - _core._official_magic_cost(faces)
                        if remaining_mp >= 0:
                            launch.update(base_override=2 * remaining_mp, value_models=list(faces))
            elif kind == 'pray':
                # Some clients also send an empty useAttackItems marker.
                if action is None:
                    action = dict(before=deepcopy(working), gf=gf, phase=0, source=0,
                                  actor=owner, kind=3, faces=[], faces_hidden=False,
                                  target=None, allocation=None)
                    working.pop('removal', None)
                    working.pop('sale', None)
                    working['guardian_changes'] = []
                    launch = None
                elif action['kind'] != 3:
                    self.history_complete = False
            elif kind == 'setTargetPlayer':
                target = e['player']
                if action and action['phase'] == 0 and action['kind'] in (0, 5, 6):
                    action['target'] = target
            elif kind == 'nextAttack':
                a = e['attack']; actor = a['playerId']
                removal_model = a.get('itemModelIds', [None])[0]
                launch = dict(origin=actor, source=actor,
                              counter=removal_model in range(181, 189), death=False,
                              reflections=0, bounces=0)
            elif kind == 'reflect' and launch:
                launch['reflections'] += 1
                launch['source'], target = target, launch['source']
            elif kind == 'bounce' and launch:
                launch['bounces'] += 1
                self.history_complete = False  # target must be explicitly witnessed
            elif kind == 'dealDarkDamage':
                # Receiver ignores damage and zeroes the current pane. Death,
                # guardian departure and revival are separate public events.
                s = pane(target)
                if action is None or s is None:
                    self.history_complete = False
                if s is not None:
                    s['hp'] = 0
            elif kind == 'revive':
                # There is no native revival token. A plain 208 consumption
                # contributes through the separate boostHP/decision delta.
                # Disguised consumption needs a native Reveal journal too.
                if action is None or e['model'] != 208 or e['face'] != e['model']:
                    self.history_complete = False
            elif kind == 'upgradeHeaven':
                # cl is an animation only, not a disease/HP/dead mutation.
                if action is None:
                    self.history_complete = False
            elif kind == 'collectCPOfEverybody':
                # Native 294 is an external phenomenon decision, with its own
                # source and journal. This marker alone cannot recover that
                # anchor (or event-time fogged CP). Keep the public recipient.
                self.history_complete = False
            elif kind in ('dealDamage', 'boostHP', 'boostMP'):
                s = pane(e.get('player', target))
                if s is not None:
                    key = 'mp' if kind == 'boostMP' else 'hp'
                    delta = -e['damage'] if kind == 'dealDamage' else e[key]
                    s[key] = min(99, max(0, s[key] + delta))
            elif kind in ('boostCP', 'takeCP', 'boostCPOfEverybody'):
                recipients = ([x['id'] for x in working['players'] if not x['dead']]
                              if kind == 'boostCPOfEverybody' else [target])
                source = launch['source'] if launch else None
                if action is None or (kind == 'takeCP' and source is None):
                    self.history_complete = False
                else:
                    for pid in recipients:
                        s = pane(pid)
                        if s is not None:
                            s['cp'] = min(99, max(0, s['cp'] + (-e['cp'] if kind == 'takeCP' else e['cp'])))
                    if kind == 'takeCP' and pane(source) is not None:
                        s = pane(source)
                        s['cp'] = min(99, max(0, s['cp'] + e['cp']))
            elif kind == 'upgradeDisease':
                s = pane(e['player'])
                if action is None:
                    self.history_complete = False
                if s is not None:
                    s['curses'] = sorted([c for c in s['curses'] if c not in CURSES[1:5]] +
                                         [e['curse']], key=CURSES.index)
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
                    if launch and e.get('buyer') is not None and e.get('seller') is not None:
                        working['sale'] = dict(origin=launch['origin'], source=e['seller'],
                            target=e['buyer'], model=models[1], price=price,
                            overflow_model=e.get('overflow_model'))
                    else:
                        self.history_complete = False
                    if e.get('overflow_unobserved'):
                        self.history_complete = False
                    elif e.get('overflow_model') is not None:
                        player = next(x for x in working['players'] if x['id'] == e['buyer'])
                        if e.get('overflow_used') and e['overflow_model'] in player['used']:
                            player['used'].remove(e['overflow_model'])
                else:
                    self.history_complete = False
            elif kind in ('removeItems', 'removeUsedMiracles'):
                model = 205 if kind == 'removeItems' else 206
                recipient = e.get('player', target)
                if action is None or e['models'] != e['faces']:
                    self.history_complete = False
                if launch and removal_model == model:
                    working['removal'] = dict(origin=launch['origin'], source=launch['source'],
                                              target=recipient, model=model, models=e['models'])
                else:
                    # A midgame receipt cannot identify the origin/source from
                    # the recipient alone. Keep the public result, omit history.
                    self.history_complete = False
                player = next(x for x in working['players'] if x['id'] == recipient)
                if kind == 'removeUsedMiracles':
                    for face in e['faces']:
                        if face in player['used']:
                            player['used'].remove(face)
                        else:
                            self.history_complete = False
                # Event-time own inventory order and refills are not journaled;
                # the received final pane below remains authoritative.
            elif kind == 'removeGuardian':
                s = pane(e['player'])
                if s is None or s['guardian'] == 'none' or action is None:
                    self.history_complete = False
                if s is not None:
                    if s['guardian'] != 'none':
                        working.setdefault('guardian_changes', []).append(dict(
                            player=e['player'], before=s['guardian'], after='none', reason='departure'))
                    s['guardian'] = 'none'
            elif kind == 'die':
                for x in working['players']:
                    if x['id'] == e['player']:
                        x['dead'] = True
            elif kind in ('gift', 'replaceItems', 'startGame', 'safe', 'block', 'danger', 'miss', 'hit', 'counterAttack', 'selfCurse', 'disease'):
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
        p['guardian_changes'] = deepcopy(working.get('guardian_changes', []))
        if 'sale' in working:
            p['sale'] = deepcopy(working['sale'])
        if 'removal' in working:
            p['removal'] = deepcopy(working['removal'])
        p['decisions'] = self.decisions
        p['last_decision'] = deepcopy(self.events[-1]['decision']) if self.events else None
        if any(d['code'] == 'multiple_pending_frames' for d in p['detail']):
            # The event tail need not describe the head of this queue. Never
            # transfer its redirect/provenance/dynamic value onto another frame.
            launch = None
            self.history_complete = False
        if p['pending']:
            pending = p['pending']
            if launch:
                pending.update({k: v for k, v in launch.items() if k not in ('base_override', 'value_models')})
                if ('base_override' not in pending and launch.get('value_models') == pending['models']
                        and 'base_override' in launch):
                    pending['base_override'] = launch['base_override']
                if 'redirect_requires_history' in p['unsupported']:
                    p['unsupported'].remove('redirect_requires_history')
            # Native 202 publishes signed amount; 58 publishes launch MP power.
            # 94 responds at catalog power 30 (plus visible modifiers); a
            # mortar's 99-power branch resolves immediately without response.
            if pending['models'][0] == 202 and 'amount' not in pending:
                magnitude = abs(_core.play_catalog()[202]['ability_value'])
                pending.update(power=0, amount=-magnitude)
                p.setdefault('warnings', []).append('assumed_dynamic_value: 202')
            elif pending['models'][0] == 58 and 'base_override' not in pending:
                p['unsupported'].append('dynamic_attack_value_unavailable')
            if pending['guardian'] in ('earth', 'moon') and not _core._official_guardian_payload_supported(
                    pending['guardian'], pending['models']):
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
        # Executable proof is deliberately narrower than observation/history.
        # Require a witnessed, contiguous pre-launch main boundary. End-state
        # resources are never inverted to invent pre-spend panes.
        prior = self.previous
        if p['phase'] != 0:
            p['_search_fallback'] = ('purchase_public_escrow_unavailable' if p['phase'] == 2
                                     else 'response_public_journal_unavailable')
            uses = [e for e in p['events'] if e['action'] == 'useAttackItems']
            allowed = {'useAttackItems', 'setTargetPlayer', 'hit', 'miss'}
            if p.get('pending') and (p['pending']['counter'] or p['pending']['death']
                    or p['pending']['guardian'] != 'none'):
                p['_search_fallback'] = 'response_special_launch'
            elif p.get('pending') and (p['pending']['reflections'] or p['pending']['bounces']):
                p['_search_fallback'] = 'response_redirect_journal_unavailable'
            elif (prior and p['update'] == prior['update'] + 1 and prior['phase'] == 0
                    and prior['actor'] == prior['owner'] and prior['owner'] != self.viewer
                    and prior['gf'] == p['gf'] and prior['hand'] == p['hand']
                    and len(uses) == 1 and uses[0]['faces'] and not uses[0]['reveals']
                    and all(e['action'] in allowed for e in p['events'])
                    and p.get('pending') and uses[0]['faces'] == p['pending']['models']):
                anchor = deepcopy(prior)
                anchor.pop('_source', None)
                anchor.pop('_search_launch', None)
                p['_search_fallback'] = 'response_public_launch_replay_mismatch'
                p['_search_launch'] = dict(before=anchor, models=uses[0]['faces'],
                    target=p['pending']['target'])
        # Retain a witnessed ordinary launch across reflection-only updates.
        # Any body, bounce, redraw, reveal, counter or missing update breaks the
        # proof. Actors come from the previous public recipient and each reflect.
        if (p['phase'] == 1 and prior and '_search_launch' in prior
                and p['update'] == prior['update'] + 1 and p.get('pending')
                and not p['pending']['counter'] and not p['pending']['death']
                and not p['pending']['bounces'] and p['pending']['guardian'] == 'none'
                and all(e['action'] in {'useDefenseItems', 'reflect'} for e in p['events'])):
            journal = deepcopy(prior['_search_launch'])
            responder = prior['pending']['target']
            responses = journal.setdefault('responses', [])
            valid = True
            waiting_reflect = False
            for event in p['events']:
                if event['action'] == 'useDefenseItems':
                    if waiting_reflect or event['reveals'] or not event['faces']:
                        valid = False
                    responses.append(dict(actor=responder, models=event['faces']))
                    waiting_reflect = True
                elif not waiting_reflect:
                    valid = False
                else:
                    responder = next(x['id'] for x in p['players'] if x['id'] != responder)
                    waiting_reflect = False
            if valid and not waiting_reflect and responses:
                p['_search_launch'] = journal
                p['_search_fallback'] = 'response_public_launch_replay_mismatch'
        # A single received update may finish our command, advance GF, then
        # launch the enemy's command. The public ledger already applied those
        # effects forward. Use its last pre-launch boundary only when every
        # preceding effect was understood and no earlier enemy command occurred.
        if (p['phase'] == 1 and '_search_launch' not in p and prior
                and p['update'] == prior['update'] + 1 and self.history_complete
                and self.events and p.get('pending') and not p['pending']['counter']
                and not p['pending']['death'] and not p['pending']['reflections']
                and not p['pending']['bounces'] and p['pending']['guardian'] == 'none'):
            final = self.events[-1]
            decision = final['decision']
            markers = [i for i, e in enumerate(p['events']) if e['action'] == 'useAttackItems']
            if (markers and decision['phase'] == 0 and decision['kind'] == 0
                    and decision['actor'] == p['owner'] != self.viewer
                    and decision['faces'] == p['pending']['models']
                    and not p['events'][markers[-1]]['reveals']
                    and all(e['action'] in {'setTargetPlayer', 'hit', 'miss'}
                            for e in p['events'][markers[-1]+1:])
                    and any(e['decision']['actor'] == self.viewer for e in self.events[:-1]
                            if e['decision']['decision'] > prior['decisions'])
                    and all(e['decision']['actor'] != p['owner'] for e in self.events[:-1]
                            if e['decision']['decision'] > prior['decisions'])):
                anchor = deepcopy(final['before'])
                for key in ('_source', '_search_launch', '_search_fallback'):
                    anchor.pop(key, None)
                anchor.update(phase=0, owner=p['owner'], actor=p['owner'], gf=decision['gf'],
                              pending=None, purchase=None, hand=deepcopy(p['hand']),
                              decisions=decision['decision']-1)
                p['_search_launch'] = dict(before=anchor, models=decision['faces'],
                                          target=p['pending']['target'])
                p['_search_fallback'] = 'response_public_launch_replay_mismatch'
        self.previous = deepcopy(p)
        self.previous['_source'] = public_source
        fatal = [code for code in p['unsupported']
                 if p['actor'] == self.viewer or code not in DECISION_ONLY_UNSUPPORTED]
        if fatal:
            return dict(status='unsupported', public=p, unsupported=p['unsupported'], detail=unsupported_detail(p),
                        history_complete=self.history_complete)
        if p['actor'] != self.viewer or p['is_over']:
            return dict(status='finished' if p['is_over'] else 'waiting', public=p,
                        unsupported=p['unsupported'], detail=unsupported_detail(p),
                        history_complete=self.history_complete)
        try:
            env = OfficialDecision(p, self.events, self.history_len)
        except (ValueError, RuntimeError, IndexError) as error:
            detail = exception_detail(error)
            frames = [d for d in p['detail'] if d['code'] == 'multiple_pending_frames']
            return dict(status='unsupported', public=p, unsupported=['unsupported_selection'],
                        detail=frames + [dict(code='unsupported_selection', **detail)] if frames else detail,
                        history_complete=self.history_complete)
        return dict(status='decision', public=p, env=env, unsupported=[],
                    detail=unsupported_detail(p),
                    history_complete=self.history_complete)


class OfficialDecision:
    """Actor-compatible immutable observation plus a pure selection cursor.

    step() advances substeps only. It never executes the chosen command.
    """
    num_envs = 1

    def __init__(self, public, events=(), history_len=64, *, excluded_ids=(),
                 exclude_enemy_default_self=False, excluded_kinds=()):
        self.public = deepcopy(public)
        self.events = deepcopy(list(events))
        self.history_len = history_len
        self.restrictions = dict(
            exclude_enemy_default_self=exclude_enemy_default_self,
            excluded_kinds=[KINDS.index(k) for k in excluded_kinds], excluded_slots=[
            i for i, h in enumerate(self.public['hand']) if h['id'] in excluded_ids])
        self.cursor = _core._OfficialCursor(dict(self.public, **self.restrictions))
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
        reasons = []
        faces = [h['face'] for h in selected]
        kind = KINDS[c['kind']]
        if kind not in OBSERVED_UI_KINDS:
            reasons.append(f'ui_kind_unobserved:{kind}')
        catalog = _core.play_catalog()
        if c['kind'] in (0, 5, 6) and c['target'] == self.public['viewer'] and faces and (
                catalog[faces[0]]['targets_enemy_by_default']):
            reasons.append('self_target_of_enemy_default_unobserved')
        # The gfrl mask forbids 94+232 in one action; keep this defensive guard.
        if 94 in faces and 232 in faces:
            reasons.append('danger_wide_requires_unobservable_mortar_absence')
        if any(h['disguised'] for h in selected):
            reasons.append('dream_actual_execution_not_verified_from_public_faces')
        disguised_faces = {h['face'] for h in self.public['hand'] if h['disguised']}
        if any(not h['disguised'] and h['face'] in disguised_faces for h in selected):
            reasons.append('ambiguous_disguised_face')
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
    Never skip malformed/unknown streams and continue with a subset. An
    explicit lenient_events option discards only unrecognized event elements.
    """
    def __init__(self, session, baseline=(), *, lenient_events=False):
        self.session = session
        self.lenient_events = lenient_events
        self.replaced_events = []
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
            return dict(status='unsupported', unsupported=['stream_reader_stopped'],
                        detail=self.failure_detail)
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
                if hasattr(self.parser, 'parse_capture'):
                    records = self.parser.parse_capture(raw, self.session.viewer, self.baseline.get(key, 0),
                        **({'lenient_events': True} if self.lenient_events else {}))['games']
                else:
                    records = self.parser.parse_prefix(raw, self.session.viewer,
                        **({'lenient_events': True} if self.lenient_events else {}))['games']
                if len({r['document_sequence'] for r in records if r['frame_end_byte'] > self.baseline.get(key, 0)}) > 1:
                    raise ValueError('multiple_documents')
                games += [(r['game'], r.get('replaced_events', [])) for r in records
                          if r['frame_end_byte'] > self.baseline.get(key, 0)]
                self.hashes[key] = digest
            for game, replacements in sorted(games, key=lambda pair: pair[0]['updateCount']):
                u = _integer(game['updateCount'])
                if u in self.seen:
                    projected = project_game(game, self.session.viewer, prior=self.context[u])
                    if projected != self.seen[u]:
                        raise ValueError('conflicting_update', dict(update=u,
                            differences=projection_differences(self.seen[u], projected)))
                    continue
                prior = self.session.previous
                projected = project_game(game, self.session.viewer, prior=prior)
                result = self.session.ingest(game)
                if result['status'] == 'unsupported':
                    self.failed = True
                    self.failure_detail = result.get('detail', {})
                    return result
                if replacements:
                    self.replaced_events.extend(dict(update=u, **e) for e in replacements)
                result['replaced_events'] = deepcopy(self.replaced_events)
                self.context[u] = dict(pending=deepcopy(prior.get('pending'))) if prior else None
                # Keep the comparison snapshot independent of session/result
                # ownership, including future ingest implementation changes.
                self.seen[u] = deepcopy(projected)
                self.latest = result
            return self.latest or dict(status='waiting', unsupported=[])
        except Exception as error:
            self.failed = True
            self.failure_detail = exception_detail(error, self.parser.ParseError)
            return dict(status='unsupported', unsupported=['invalid_or_unknown_stream'],
                        detail=self.failure_detail)
