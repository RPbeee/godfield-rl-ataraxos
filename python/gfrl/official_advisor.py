"""Offline checkpoint advice; semantic actions are proposals, never clicks."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from . import _core
from .actor import Actor
from .evaluation import load_policy_model, _policy_rng
from .official import OfficialSession, read_games
from .play import Presenter, KINDS
from .search import _SUPPORT
from .official_search import (add_search_arguments, search_config, search_agent, root_matches,
                              empty_diagnostics, public_report, policy_at_prefix)
from .actor import synchronize


def advise(result, agent, *, search=False):
    env = result['env']; p = env.public
    seat = next(i for i, x in enumerate(p['players']) if x['id'] == p['viewer'])
    presenter = Presenter(seat)
    # Presenter historically assumes player IDs 0/1. Replace only its identity
    # mapping, preserving all the shared card/board/action descriptions.
    presenter.who = lambda pid: ('環境' if pid is None else 'あなた' if pid == p['viewer'] else '相手')
    board = presenter.board(env.play_view(0, seat), hand=env.play_hand(0, seat))
    if p['pending'] and p['pending']['models'][0] in (4, 5):
        import re
        board = re.sub(r'威力 \d+ /', '取引 /', board)
    labels = []
    search_status = 'forced_no_inference' if search and env.complete else 'disabled'
    root = None
    diagnostics = empty_diagnostics()
    if search and not env.complete:
        response_enabled = getattr(getattr(agent, 'config', None), 'search_response', False)
        if p['phase'] != 0 and not response_enabled:
            search_status = 'response_search_disabled_policy_fallback'
        else:
            try:
                root = _core._official_search_root(p, env.events, env.history_len)
                support = int(root.determinize_support(np.array([0], np.int32))[0])
                if support != 0:
                    search_status = _SUPPORT.get(support, 'unsupported_root') + '_policy_fallback'
                    root = None
                elif not root_matches(root, env):
                    root = None
                    search_status = 'public_root_mismatch_policy_fallback'
                else:
                    search_status = 'main_public_worlds' if p['phase'] == 0 else 'response_public_worlds'
            except (ValueError, RuntimeError) as error:
                root = None
                # Only fixed local codes may cross the diagnostic boundary.
                reason = {'hidden_launch_status': 'response_hidden_launch_status',
                          'unsupported_public_launch': 'response_special_launch',
                          'public_launch_decision_mismatch': 'response_public_decision_mismatch'}.get(str(error))
                search_status = (reason or (p.get('_search_fallback', 'response_public_journal_unavailable')
                                 if p['phase'] != 0 else 'unsupported_root')) + '_policy_fallback'
    while not env.complete:
        choices = presenter.options(env, env.play_hand(0, seat))
        if root is not None:
            if not root_matches(root, env):
                root = None
                search_status = 'public_prefix_mismatch_policy_fallback'
                choice = int(policy_at_prefix(agent.actor, env, np.array([0], np.int32))[0])
            else:
                first = int(root.substep[0]) == 0
                device = getattr(agent, 'device', None)
                if device is not None:
                    synchronize(device)
                started = perf_counter()
                choice = int(agent(root, np.array([0], np.int32))[0])
                if device is not None:
                    synchronize(device)
                diagnostics['search_seconds'] += perf_counter()-started
                if first:
                    diagnostics.update(public_report(agent, env))
                    if not diagnostics['search_candidates']:
                        search_status = 'no_candidates_policy_fallback'
        else:
            actor = agent.actor if search else agent
            choice = int(actor(env, np.array([0], np.int32))[0])
        labels.append(next(label for c, label in choices if c == choice))
        env.step(np.array([choice], np.int32))
        # Never execute the completed command in the executable search root.
        if root is not None and not env.complete:
            root.step(np.array([choice], np.int32))
    semantic = env.semantic_action()
    cards = ' + '.join(presenter.name(c['model']) for c in semantic['cards'])
    text = KINDS[env.cursor.command()['kind']] + (': ' + cards if cards else '')
    if semantic['target'] is not None:
        text += ' → ' + presenter.who(semantic['target'])
    unsupported = env.action_support()
    return dict(status='unsupported' if unsupported else 'advice', unsupported=unsupported, update=p['update'], gf=p['gf'], board=board,
                explanation=text, selections=labels, action=semantic,
                history_complete=result['history_complete'], search_status=search_status, **diagnostics,
                **({'detail': result['detail']} if result.get('detail') else {}),
                warnings=([] if result['history_complete'] else ['partial_history_and_decision_count'])
                         + p.get('warnings', [])
                         + (['earth_moon_general_execution'] if any(
                             x['status'] and x['status']['guardian'] in ('earth', 'moon')
                             for x in p['players']) else [])
                         + ([dict(replaced_events=result['replaced_events'])] if result.get('replaced_events') else []))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--viewer-id', type=int, required=True)
    parser.add_argument('--document-sequence', type=int, help='Select one document in a multi-document prefix')
    parser.add_argument('--checkpoint', required=True, help='checkpoint EMA (same loader as play.py)')
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--search', action='store_true')
    add_search_arguments(parser, advisor=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--history-len', type=int, default=64)
    parser.add_argument('--json', action='store_true', help='JSONL including semantic actions')
    args = parser.parse_args(argv)
    try:
        search_config(args)
    except ValueError as error:
        parser.error(str(error))
    if args.history_len < 0:
        parser.error('history-len must be nonnegative')
    torch.set_num_threads(args.threads)
    model = load_policy_model(args.checkpoint, args.device)
    session = OfficialSession(args.viewer_id, args.history_len)
    agent = None
    if args.search:
        agent = search_agent(model, args, seed=args.seed)
    try:
        with _policy_rng(args.seed, (agent,) if agent else ()):
            for game in read_games(args.input, args.viewer_id, args.document_sequence):
                result = session.ingest(game)
                if result['status'] == 'decision':
                    if agent is None:
                        agent = Actor(model, result['env'], args.device, sample=not args.greedy)
                    result = advise(result, agent, search=args.search)
                else:
                    result = {key: value for key, value in result.items() if key != 'public'}
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
                elif result['status'] == 'advice':
                    print(result['board'])
                    print('助言: ' + result['explanation'])
                    if result['warnings']:
                        print('注意: 履歴・決定数は部分記録です')
                    print('意味行動: ' + json.dumps(result['action'], ensure_ascii=False))
                else:
                    print(json.dumps(result, ensure_ascii=False))
    except Exception:
        # Never echo raw payloads, input paths, model errors, or identities.
        print('error:offline_advisor_input_or_model_error', file=sys.stderr)
        return 1
    return 0


DREAM_REASON = 'dream_actual_execution_not_verified_from_public_faces'


def advise_supported(result, agent, *, search=False, max_samples=16):
    """Prefer verified actions; relax only Dream after the safe mask is exhausted."""
    from .official import OfficialDecision, OBSERVED_UI_KINDS, KINDS as ACTION_KINDS
    advice = _advise_supported(result, agent, search=search, max_samples=max_samples)
    exclusions = advice.get('mask_exclusions', [])
    if not advice.get('selection_exhausted') or not any(e['reason'] == DREAM_REASON for e in exclusions):
        return advice
    # Preserve every other execution gap. In particular, never restore discard
    # (or another unobserved kind) while restoring disguised card selections.
    hard_ids = {i for e in exclusions if e['reason'] != DREAM_REASON for i in e.get('item_ids', [])}
    kinds = set(ACTION_KINDS) - OBSERVED_UI_KINDS
    prune_self = any(e['reason'] == 'self_target_of_enemy_default_unobserved' for e in exclusions)
    env = result['env']
    try:
        fresh = dict(result, env=OfficialDecision(env.public, env.events, env.history_len,
            excluded_ids=hard_ids, excluded_kinds=kinds, exclude_enemy_default_self=prune_self))
    except (ValueError, RuntimeError):
        return advice
    actor = agent.actor if search else agent
    recovered = _advise_supported(fresh, actor, max_samples=max_samples, allow_dream=True,
        excluded_ids=hard_ids, excluded_kinds=kinds, exclude_self=prune_self,
        initial_exclusions=exclusions)
    if search:
        recovered.update({k: advice[k] for k in empty_diagnostics()},
                         search_status='dream_last_resort_policy_fallback',
                         initial_search_status=advice['search_status'])
    return recovered


def _advise_supported(result, agent, *, search=False, max_samples=16, allow_dream=False,
                      excluded_ids=(), excluded_kinds=(), exclude_self=False, initial_exclusions=()):
    """Bot-only recovery: prune card/target/kind gaps, then retry using policy RNG."""
    from .official import OfficialDecision
    env = result['env']
    def choose(position, policy, *, search=False):
        advice = advise(position, policy, search=search)
        if allow_dream and DREAM_REASON in advice['unsupported'] and any(
                h['disguised'] and h['id'] in {c['item_id'] for c in advice['action']['cards']}
                for h in env.public['hand']):
            advice['unsupported'] = [r for r in advice['unsupported'] if r != DREAM_REASON]
            advice['status'] = 'unsupported' if advice['unsupported'] else 'advice'
            advice['warnings'].append(DREAM_REASON)
            advice['dream_last_resort'] = True
        if initial_exclusions:
            advice['mask_exclusions'] = list(initial_exclusions)
        return advice
    advice = choose(result, agent, search=search)
    if advice['status'] != 'unsupported':
        return advice
    search_diagnostics = {k: advice[k] for k in empty_diagnostics()}
    searched_status = advice['search_status']
    actor = agent.actor if search else agent
    excluded = set(excluded_ids)
    excluded_kinds = set(excluded_kinds)
    exclude_enemy_default_self = exclude_self
    exclusions = list(initial_exclusions)
    while advice['status'] == 'unsupported':
        reasons = advice.get('unsupported', [])
        selected = {c['item_id'] for c in advice['action']['cards']}
        blocked_by_reason = []
        if 'dream_actual_execution_not_verified_from_public_faces' in reasons:
            blocked_by_reason.append(dict(reason='dream_actual_execution_not_verified_from_public_faces',
                item_ids=[h['id'] for h in env.public['hand'] if h['disguised'] and h['id'] in selected]))
        if 'danger_wide_requires_unobservable_mortar_absence' in reasons:
            blocked_by_reason.append(dict(reason='danger_wide_requires_unobservable_mortar_absence',
                item_ids=[c['item_id'] for c in advice['action']['cards'] if c['model'] == 232]))
        new = {i for e in blocked_by_reason for i in e['item_ids']} - excluded
        prune_self = ('self_target_of_enemy_default_unobserved' in reasons
                      and not exclude_enemy_default_self)
        kind = advice['action']['kind']
        prune_kind = f'ui_kind_unobserved:{kind}' in reasons and kind not in excluded_kinds
        kind_exclusions = [dict(reason=f'ui_kind_unobserved:{kind}', kind=kind)] if prune_kind else []
        if not new and not prune_self and not prune_kind:
            break
        excluded.update(new)
        if prune_kind:
            excluded_kinds.add(kind)
        try:
            fresh = dict(result, env=OfficialDecision(env.public, env.events, env.history_len,
                excluded_ids=excluded, excluded_kinds=excluded_kinds,
                exclude_enemy_default_self=exclude_enemy_default_self or prune_self))
        except (ValueError, RuntimeError):
            # No completion survives the accumulated exclusions. Rolling back
            # the last exclusion would sample that same unsupported card again.
            advice['mask_exclusions'] = exclusions + ([dict(
                reason='self_target_of_enemy_default_unobserved', target='self')] if prune_self else []) + [dict(reason=e['reason'],
                item_ids=sorted(set(e['item_ids']) & new))
                for e in blocked_by_reason if set(e['item_ids']) & new] + kind_exclusions
            advice['selection_exhausted'] = True
            return advice
        exclude_enemy_default_self |= prune_self
        if prune_self:
            exclusions.append(dict(reason='self_target_of_enemy_default_unobserved', target='self'))
        exclusions.extend(dict(reason=e['reason'], item_ids=sorted(set(e['item_ids']) & new))
                          for e in blocked_by_reason if set(e['item_ids']) & new)
        exclusions.extend(kind_exclusions)
        advice = choose(fresh, actor)  # Search cannot use a different mask.
        if search:
            advice.update(search_diagnostics, search_status='ui_mask_policy_fallback',
                          initial_search_status=searched_status)
        advice['mask_exclusions'] = exclusions
    if advice['status'] == 'advice':
        return advice
    original_sample = actor.sample
    try:
        actor.sample = True
        for n in range(1, max_samples+1):
            fresh = dict(result, env=OfficialDecision(env.public, env.events, env.history_len,
                excluded_ids=excluded, excluded_kinds=excluded_kinds,
                exclude_enemy_default_self=exclude_enemy_default_self))
            advice = choose(fresh, actor)
            if search:
                advice.update(search_diagnostics, search_status='ui_retry_policy_fallback',
                              initial_search_status=searched_status)
            advice['fallback_samples'] = n
            if exclusions:
                advice['mask_exclusions'] = exclusions
            if advice['status'] == 'advice':
                return advice
        return advice
    finally:
        actor.sample = original_sample
