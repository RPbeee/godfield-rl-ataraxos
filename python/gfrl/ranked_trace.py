"""Anonymous numeric-only turn/decision trace for official ranked campaigns.

Only research game numbers, relative seats (self/opponent booleans), public card
model IDs (1..296), public status integers, fixed enums, and model value/policy
summary numbers are ever written. Names, user IDs, room keys, URLs, chat,
screenshots, raw receive bytes, item instance IDs, and foreign unused cards
never enter or leave this module.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .official import CURSES, KINDS

TRACE_SCHEMA = 1
GUARDIANS = frozenset(('none', 'mercury', 'venus', 'earth', 'mars', 'jupiter',
                       'saturn', 'uranus', 'neptune', 'pluto', 'moon'))
CURSE_SET = frozenset(CURSES)
KIND_SET = frozenset(KINDS)
EVENT_ACTIONS = frozenset((
    'startGame', 'gift', 'advanceGF', 'useAttackItems', 'setTargetPlayer', 'dealDamage',
    'addCurse', 'disease', 'useDefenseItems', 'removeCurses', 'die', 'endGame', 'boostHP',
    'boostMP', 'exchange', 'buy', 'canNotBuy', 'setBought', 'useDevilItem', 'removeSomething',
    'selfCurse', 'redraw', 'replaceItems', 'safe', 'block', 'setGuardian', 'attackByGuardian',
    'counterAttack', 'nextAttack', 'bounce', 'reflect', 'phenomenon', 'setCurseOfEverybody',
    'danger', 'miss', 'hit', 'removeGuardian', 'removeItems', 'removeUsedMiracles', 'sell',
    'pray', 'takeCP', 'boostCP', 'boostCPOfEverybody', 'upgradeDisease', 'upgradeHeaven',
    'dealDarkDamage', 'revive', 'collectCPOfEverybody', 'unknownEvent', 'unsupported',
))
POLICY_ONLY_CODES = frozenset((
    'dying_players_decision_unavailable', 'redirect_chance_activation_unavailable',
    'redirect_requires_history', 'dynamic_hit_rate_unavailable',
    'death_attack_decision_unavailable', 'multiple_pending_frames',
    'confusion_count_unavailable',
))


def _is_int(v, lo=-10 ** 9, hi=10 ** 9):
    return type(v) is int and lo <= v <= hi


def _is_card(v):
    return type(v) is int and 1 <= v <= 296


def _is_prob(v):
    return type(v) in (int, float) and math.isfinite(v) and 0.0 <= float(v) <= 1.0


def _is_finite(v, lo=-1e6, hi=1e6):
    return type(v) in (int, float) and math.isfinite(v) and lo <= float(v) <= hi


def _is_bool(v):
    return type(v) is bool


def _is_opt_bool(v):
    return v is None or type(v) is bool


def _is_opt_int(v, lo=-10 ** 9, hi=10 ** 9):
    return v is None or _is_int(v, lo, hi)


def _round4(v):
    return round(float(v), 4)


def _rel(pid, viewer):
    if pid is None:
        return None
    return pid == viewer


def _sanitize_player(p):
    status = p.get('status')
    out = dict(
        dead=bool(p['dead']),
        fogged=bool(p['fogged']),
        used=[int(m) for m in p.get('used', []) if _is_card(m)],
    )
    if status is None:
        out.update(hp=None, mp=None, cp=None, guardian=None, confused=None, curses=None)
    else:
        g = status.get('guardian', 'none')
        out.update(
            hp=int(status['hp']),
            mp=int(status['mp']),
            cp=int(status['cp']),
            guardian=g if g in GUARDIANS else 'none',
            confused=bool(status['confused']),
            curses=[c for c in status.get('curses', []) if c in CURSE_SET],
        )
    return out


def _sanitize_pending(pending, viewer):
    if pending is None:
        return None
    g = pending.get('guardian', 'none')
    return dict(
        origin_self=_rel(pending.get('origin'), viewer),
        source_self=_rel(pending.get('source'), viewer),
        target_self=pending['target'] == viewer,
        models=[int(m) for m in pending.get('models', []) if _is_card(m)],
        guardian=g if g in GUARDIANS else 'none',
        power=pending.get('power'),
        amount=pending.get('amount'),
        base_override=pending.get('base_override'),
        hit_rate=pending.get('hit_rate'),
        reflections=int(pending.get('reflections', 0)),
        bounces=int(pending.get('bounces', 0)),
        counter=bool(pending.get('counter', False)),
        death=bool(pending.get('death', False)),
        forced_hit=pending.get('forced_hit'),
    )


def _sanitize_purchase(purchase, viewer):
    if purchase is None:
        return None
    return dict(
        origin_self=purchase['origin'] == viewer,
        buyer_self=purchase['buyer'] == viewer,
        seller_self=purchase['seller'] == viewer,
        model=int(purchase['model']),
    )


def _sanitize_event(e, viewer):
    act = e.get('action')
    out = dict(action=act if act in EVENT_ACTIONS else 'unsupported')
    if 'player' in e:
        out['player_self'] = _rel(e['player'], viewer)
    if 'buyer' in e:
        out['buyer_self'] = _rel(e['buyer'], viewer)
    if 'seller' in e:
        out['seller_self'] = _rel(e['seller'], viewer)
    for key in ('faces', 'models'):
        if key in e and isinstance(e[key], list):
            out[key] = [int(m) for m in e[key] if _is_card(m)]
    if 'reveals' in e and isinstance(e['reveals'], list):
        out['reveals'] = [[int(r['face']), int(r['model'])] for r in e['reveals']
                          if isinstance(r, dict) and _is_card(r.get('face')) and _is_card(r.get('model'))]
    for key in ('damage', 'hp', 'mp', 'cp', 'index'):
        if key in e and _is_int(e[key]):
            out[key] = e[key]
    for key in ('model', 'face', 'overflow_model'):
        if key in e and _is_card(e[key]):
            out[key] = e[key]
    for key in ('bought', 'natural', 'overflow_used', 'overflow_unobserved'):
        if key in e and _is_bool(e[key]):
            out[key] = e[key]
    if 'curse' in e and e['curse'] in CURSE_SET:
        out['curse'] = e['curse']
    if 'curses' in e and isinstance(e['curses'], list):
        out['curses'] = [c for c in e['curses'] if c in CURSE_SET]
    if 'guardian' in e and e['guardian'] in GUARDIANS:
        out['guardian'] = e['guardian']
    if 'attack' in e and isinstance(e['attack'], dict):
        a = e['attack']
        att = dict(
            source_self=_rel(a.get('playerId'), viewer),
            target_self=_rel(a.get('targetPlayerId'), viewer),
            models=[int(m) for m in a.get('itemModelIds', []) if _is_card(m)],
        )
        if _is_int(a.get('atk')):
            att['atk'] = a['atk']
        if a.get('guardian') in GUARDIANS:
            att['guardian'] = a['guardian']
        if _is_bool(a.get('reversed')):
            att['reversed'] = a['reversed']
        out['attack'] = att
    return out


def build_update_row(game, public, *, t):
    """Build an anonymous, numeric/enum-only snapshot of one received public update."""
    viewer = public['viewer']
    own = next(p for p in public['players'] if p['id'] == viewer)
    opp = next(p for p in public['players'] if p['id'] != viewer)
    row = dict(
        v=TRACE_SCHEMA,
        ev='update',
        game=int(game),
        t=int(t),
        update=int(public['update']),
        gf=int(public['gf']),
        tiebreak_gf=public.get('tiebreak_gf'),
        phase=int(public['phase']),
        is_over=bool(public['is_over']),
        owner_self=public['owner'] == viewer,
        actor_self=_rel(public.get('actor'), viewer),
        self=_sanitize_player(own),
        opp=_sanitize_player(opp),
        hand=[[int(h['face']), int(bool(h['disguised'])), int(bool(h['used']))]
              for h in public.get('hand', [])],
        pending=_sanitize_pending(public.get('pending'), viewer),
        purchase=_sanitize_purchase(public.get('purchase'), viewer),
        events=[_sanitize_event(e, viewer) for e in public.get('events', [])],
        policy_only=[c for c in public.get('_policy_only', []) if c in POLICY_ONLY_CODES],
    )
    return validate_trace_row(row)


def build_decision_row(game, public, advice, *, decision_idx, policy_stats=None, t):
    """Build an anonymous record of one accepted bot decision and its model estimates."""
    viewer = public['viewer']
    act = advice['action']
    alloc = act.get('allocation')
    clean_alloc = (dict(hp=int(alloc['hp']), mp=int(alloc['mp']), cp=int(alloc['cp']))
                   if isinstance(alloc, dict) else None)
    stats = policy_stats or {}
    row = dict(
        v=TRACE_SCHEMA,
        ev='decision',
        game=int(game),
        t=int(t),
        update=int(public['update']),
        gf=int(public['gf']),
        phase=int(public['phase']),
        decision_idx=int(decision_idx),
        action=dict(
            kind=act['kind'],
            models=[int(c['model']) for c in act.get('cards', [])],
            target_self=_rel(act.get('target'), viewer),
            allocation=clean_alloc,
        ),
        dream_played=bool(advice.get('dream_played', False)),
        dream_last_resort=bool(advice.get('dream_last_resort', False)),
        value=_round4(stats['value']) if 'value' in stats and stats['value'] is not None else None,
        value_probs=[_round4(x) for x in stats['value_probs']] if stats.get('value_probs') is not None else None,
        action_logp=_round4(stats['action_logp']) if stats.get('action_logp') is not None else None,
        stage_probs=[_round4(x) for x in stats['stage_probs']] if stats.get('stage_probs') is not None else None,
        stage_entropies=[_round4(x) for x in stats['stage_entropies']] if stats.get('stage_entropies') is not None else None,
    )
    return validate_trace_row(row)


_PLAYER_KEYS = frozenset(('dead', 'fogged', 'used', 'hp', 'mp', 'cp', 'guardian', 'confused', 'curses'))
_PENDING_KEYS = frozenset(('origin_self', 'source_self', 'target_self', 'models', 'guardian', 'power',
                           'amount', 'base_override', 'hit_rate', 'reflections', 'bounces', 'counter',
                           'death', 'forced_hit'))
_PURCHASE_KEYS = frozenset(('origin_self', 'buyer_self', 'seller_self', 'model'))
_EVENT_KEYS = frozenset(('action', 'player_self', 'buyer_self', 'seller_self', 'faces', 'models',
                         'reveals', 'damage', 'hp', 'mp', 'cp', 'index', 'model', 'face',
                         'overflow_model', 'bought', 'natural', 'overflow_used', 'overflow_unobserved',
                         'curse', 'curses', 'guardian', 'attack'))
_ATTACK_KEYS = frozenset(('source_self', 'target_self', 'models', 'atk', 'guardian', 'reversed'))
_UPDATE_KEYS = frozenset(('v', 'ev', 'game', 't', 'update', 'gf', 'tiebreak_gf', 'phase', 'is_over',
                          'owner_self', 'actor_self', 'self', 'opp', 'hand', 'pending', 'purchase',
                          'events', 'policy_only'))
_ACTION_KEYS = frozenset(('kind', 'models', 'target_self', 'allocation'))
_DECISION_KEYS = frozenset(('v', 'ev', 'game', 't', 'update', 'gf', 'phase', 'decision_idx', 'action',
                            'dream_played', 'dream_last_resort', 'value', 'value_probs', 'action_logp',
                            'stage_probs', 'stage_entropies'))


def _check_player(p):
    if not isinstance(p, dict) or set(p) != _PLAYER_KEYS:
        raise ValueError('trace_bad_player')
    if not (_is_bool(p['dead']) and _is_bool(p['fogged'])
            and isinstance(p['used'], list) and all(_is_card(m) for m in p['used'])):
        raise ValueError('trace_bad_player')
    if p['hp'] is None:
        if any(p[k] is not None for k in ('mp', 'cp', 'guardian', 'confused', 'curses')):
            raise ValueError('trace_bad_player')
    else:
        if not (_is_int(p['hp'], 0, 99) and _is_int(p['mp'], 0, 99) and _is_int(p['cp'], 0, 99)
                and p['guardian'] in GUARDIANS and _is_bool(p['confused'])
                and isinstance(p['curses'], list) and all(c in CURSE_SET for c in p['curses'])):
            raise ValueError('trace_bad_player')


def _check_event(e):
    if not isinstance(e, dict) or set(e) - _EVENT_KEYS or e.get('action') not in EVENT_ACTIONS:
        raise ValueError('trace_bad_event')
    for k in ('player_self', 'buyer_self', 'seller_self'):
        if k in e and not _is_opt_bool(e[k]):
            raise ValueError('trace_bad_event')
    for k in ('faces', 'models'):
        if k in e and not (isinstance(e[k], list) and all(_is_card(m) for m in e[k])):
            raise ValueError('trace_bad_event')
    if 'reveals' in e:
        if not (isinstance(e['reveals'], list)
                and all(isinstance(r, list) and len(r) == 2 and _is_card(r[0]) and _is_card(r[1])
                        for r in e['reveals'])):
            raise ValueError('trace_bad_event')
    for k in ('damage', 'hp', 'mp', 'cp', 'index'):
        if k in e and not _is_int(e[k]):
            raise ValueError('trace_bad_event')
    for k in ('model', 'face', 'overflow_model'):
        if k in e and not _is_card(e[k]):
            raise ValueError('trace_bad_event')
    for k in ('bought', 'natural', 'overflow_used', 'overflow_unobserved'):
        if k in e and not _is_bool(e[k]):
            raise ValueError('trace_bad_event')
    if 'curse' in e and e['curse'] not in CURSE_SET:
        raise ValueError('trace_bad_event')
    if 'curses' in e and not (isinstance(e['curses'], list) and all(c in CURSE_SET for c in e['curses'])):
        raise ValueError('trace_bad_event')
    if 'guardian' in e and e['guardian'] not in GUARDIANS:
        raise ValueError('trace_bad_event')
    if 'attack' in e:
        a = e['attack']
        if (not isinstance(a, dict) or set(a) - _ATTACK_KEYS
                or not _is_opt_bool(a.get('source_self')) or not _is_opt_bool(a.get('target_self'))
                or not (isinstance(a.get('models'), list) and all(_is_card(m) for m in a['models']))
                or ('atk' in a and not _is_int(a['atk']))
                or ('guardian' in a and a['guardian'] not in GUARDIANS)
                or ('reversed' in a and not _is_bool(a['reversed']))):
            raise ValueError('trace_bad_event')


def validate_trace_row(row):
    """Fail-closed allowlist check: no free text, names, raw IDs, URLs or unknown keys."""
    if not isinstance(row, dict) or row.get('v') != TRACE_SCHEMA:
        raise ValueError('trace_bad_row')
    ev = row.get('ev')
    if ev == 'update':
        if set(row) != _UPDATE_KEYS:
            raise ValueError('trace_bad_keys')
        if not (_is_int(row['game'], 1, 10 ** 6) and _is_int(row['t'], 0, 4102444800)
                and _is_int(row['update'], 0, 10 ** 7) and _is_int(row['gf'], 0, 10 ** 6)
                and _is_opt_int(row['tiebreak_gf'], 0, 10 ** 6) and _is_int(row['phase'], 0, 4)
                and _is_bool(row['is_over']) and _is_bool(row['owner_self'])
                and _is_opt_bool(row['actor_self'])):
            raise ValueError('trace_bad_value')
        _check_player(row['self'])
        _check_player(row['opp'])
        if not (isinstance(row['hand'], list) and len(row['hand']) <= 32
                and all(isinstance(h, list) and len(h) == 3 and _is_card(h[0])
                        and h[1] in (0, 1) and h[2] in (0, 1) and type(h[1]) is int and type(h[2]) is int
                        for h in row['hand'])):
            raise ValueError('trace_bad_hand')
        if row['pending'] is not None:
            p = row['pending']
            if (not isinstance(p, dict) or set(p) != _PENDING_KEYS
                    or not (_is_opt_bool(p['origin_self']) and _is_opt_bool(p['source_self'])
                            and _is_bool(p['target_self']) and isinstance(p['models'], list)
                            and all(_is_card(m) for m in p['models']) and p['guardian'] in GUARDIANS
                            and _is_opt_int(p['power']) and _is_opt_int(p['amount'])
                            and _is_opt_int(p['base_override']) and _is_opt_int(p['hit_rate'], 0, 100)
                            and _is_int(p['reflections'], 0, 100) and _is_int(p['bounces'], 0, 100)
                            and _is_bool(p['counter']) and _is_bool(p['death'])
                            and _is_opt_bool(p['forced_hit']))):
                raise ValueError('trace_bad_pending')
        if row['purchase'] is not None:
            q = row['purchase']
            if (not isinstance(q, dict) or set(q) != _PURCHASE_KEYS
                    or not (_is_bool(q['origin_self']) and _is_bool(q['buyer_self'])
                            and _is_bool(q['seller_self']) and _is_card(q['model']))):
                raise ValueError('trace_bad_purchase')
        if not isinstance(row['events'], list):
            raise ValueError('trace_bad_events')
        for e in row['events']:
            _check_event(e)
        if not (isinstance(row['policy_only'], list) and all(c in POLICY_ONLY_CODES for c in row['policy_only'])):
            raise ValueError('trace_bad_policy_only')
        return row
    if ev == 'decision':
        if set(row) != _DECISION_KEYS:
            raise ValueError('trace_bad_keys')
        if not (_is_int(row['game'], 1, 10 ** 6) and _is_int(row['t'], 0, 4102444800)
                and _is_int(row['update'], 0, 10 ** 7) and _is_int(row['gf'], 0, 10 ** 6)
                and _is_int(row['phase'], 0, 4) and _is_int(row['decision_idx'], 1, 10 ** 6)
                and _is_bool(row['dream_played']) and _is_bool(row['dream_last_resort'])):
            raise ValueError('trace_bad_value')
        a = row['action']
        if (not isinstance(a, dict) or set(a) != _ACTION_KEYS or a['kind'] not in KIND_SET
                or not (isinstance(a['models'], list) and all(_is_card(m) for m in a['models']))
                or not _is_opt_bool(a['target_self'])):
            raise ValueError('trace_bad_action')
        if a['allocation'] is not None:
            al = a['allocation']
            if (not isinstance(al, dict) or set(al) != {'hp', 'mp', 'cp'}
                    or not all(_is_int(al[k], 0, 99) for k in ('hp', 'mp', 'cp'))):
                raise ValueError('trace_bad_allocation')
        if row['value'] is not None and not _is_finite(row['value'], -1.0001, 1.0001):
            raise ValueError('trace_bad_stats')
        if row['value_probs'] is not None and not (
                isinstance(row['value_probs'], list) and len(row['value_probs']) == 3
                and all(_is_prob(x) for x in row['value_probs'])):
            raise ValueError('trace_bad_stats')
        if row['action_logp'] is not None and not _is_finite(row['action_logp'], -1000.0, 0.0001):
            raise ValueError('trace_bad_stats')
        if row['stage_probs'] is not None and not (
                isinstance(row['stage_probs'], list) and len(row['stage_probs']) <= 64
                and all(_is_prob(x) for x in row['stage_probs'])):
            raise ValueError('trace_bad_stats')
        if row['stage_entropies'] is not None and not (
                isinstance(row['stage_entropies'], list) and len(row['stage_entropies']) <= 64
                and all(_is_finite(x, -0.0001, 10.0) for x in row['stage_entropies'])):
            raise ValueError('trace_bad_stats')
        return row
    raise ValueError('trace_bad_ev')


class TracingActor:
    """Thin Actor wrapper that captures value/policy stats without changing choices."""
    def __init__(self, actor):
        self.actor = actor
        self._value = None
        self._value_probs = None
        self._stage_probs = []
        self._stage_entropies = []
        self._logp_sum = 0.0
        self._active_env = None

    @property
    def sample(self):
        return self.actor.sample

    @sample.setter
    def sample(self, value):
        self.actor.sample = value

    def reset_stats(self):
        self._value = None
        self._value_probs = None
        self._stage_probs = []
        self._stage_entropies = []
        self._logp_sum = 0.0
        self._active_env = None

    def pop_stats(self):
        if self._value is None and not self._stage_probs:
            return None
        out = dict(
            value=self._value,
            value_probs=self._value_probs,
            action_logp=self._logp_sum if self._stage_probs else None,
            stage_probs=list(self._stage_probs),
            stage_entropies=list(self._stage_entropies),
        )
        self.reset_stats()
        return out

    def __call__(self, env, rows=None):
        if env is not self._active_env:
            self.reset_stats()
            self._active_env = env
        self.actor._bind(env)
        res = self.actor.act(rows)
        row = 0 if rows is None else int(np.asarray(rows)[0])
        if self._value is None:
            logits = torch.as_tensor(res.value_logits[row], dtype=torch.float64)
            probs = torch.softmax(logits, dim=-1).tolist()
            self._value_probs = probs
            self._value = float(probs[0] - probs[1])
        lp_sel = float(res.log_prob[row])
        all_lp = np.asarray(res.log_probs[row], dtype=np.float64)
        mask = np.asarray(env.mask[row], dtype=bool)
        valid_lp = all_lp[mask]
        probs_stage = np.exp(valid_lp)
        entropy = float(-np.sum(probs_stage * np.where(probs_stage > 0, valid_lp, 0.0)))
        self._stage_probs.append(float(math.exp(lp_sel)))
        self._stage_entropies.append(max(0.0, entropy))
        self._logp_sum += lp_sel
        return res.choices


class RankedTrace:
    """Append-only 0600 JSONL writer for anonymous turn/decision traces."""
    def __init__(self, directory, *, clock=time.time):
        self.directory = Path(directory).absolute()
        self.clock = clock
        self.path = self.directory / 'anon_turns.jsonl'
        self.fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self._last_update = None

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def _append(self, row):
        if self.fd is None:
            return
        data = (json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode()
        os.write(self.fd, data)
        os.fsync(self.fd)

    def record_update(self, game, public):
        if game is None:
            return
        key = (int(game), int(public['update']))
        if key == self._last_update:
            return
        try:
            row = build_update_row(game, public, t=int(self.clock()))
            self._append(row)
            self._last_update = key
        except Exception:
            pass

    def record_decision(self, game, public, advice, *, decision_idx, policy_stats=None):
        if game is None:
            return
        try:
            row = build_decision_row(game, public, advice, decision_idx=decision_idx,
                                     policy_stats=policy_stats, t=int(self.clock()))
            self._append(row)
        except Exception:
            pass
