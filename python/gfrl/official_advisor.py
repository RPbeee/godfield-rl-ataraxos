"""Offline checkpoint advice; semantic actions are proposals, never clicks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from . import _core
from .actor import Actor
from .evaluation import load_policy_model, _policy_rng
from .official import OfficialSession, read_games
from .play import Presenter, KINDS
from .search import SearchAgent, SearchConfig, load_search_belief


def advise(result, agent, *, search=False):
    env = result['env']; p = env.public
    seat = next(i for i, x in enumerate(p['players']) if x['id'] == p['viewer'])
    presenter = Presenter(seat)
    # Presenter historically assumes player IDs 0/1. Replace only its identity
    # mapping, preserving all the shared card/board/action descriptions.
    presenter.who = lambda pid: ('環境' if pid is None else 'あなた' if pid == p['viewer'] else '相手')
    board = presenter.board(env.play_view(0, seat), hand=env.play_hand(0, seat))
    labels = []
    search_status = 'forced_no_inference' if search and env.complete else 'disabled'
    root = None
    if search and not env.complete:
        if p['phase'] == 0:
            try:
                root = _core._official_search_root(p, env.events, env.history_len)
                if int(root.determinize_support(np.array([0],np.int32))[0]) != 0:
                    root = None
                    search_status = 'unsupported_root_policy_fallback'
                else:
                    search_status = 'main_public_worlds'
            except (ValueError, RuntimeError):
                search_status = 'unsupported_root_policy_fallback'
        else:
            search_status = 'response_public_journal_unavailable_policy_fallback'
    while not env.complete:
        choices = presenter.options(env, env.play_hand(0, seat))
        if root is not None:
            choice = int(agent(root, np.array([0], np.int32))[0])
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
                history_complete=result['history_complete'], search_status=search_status,
                warnings=[] if result['history_complete'] else ['partial_history_and_decision_count'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--viewer-id', type=int, required=True)
    parser.add_argument('--document-sequence', type=int, help='Select one document in a multi-document prefix')
    parser.add_argument('--checkpoint', required=True, help='checkpoint EMA (same loader as play.py)')
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--belief', default='uniform', help='uniform / b1 / belief checkpoint; oracle forbidden')
    parser.add_argument('--fit')
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--history-len', type=int, default=64)
    parser.add_argument('--json', action='store_true', help='JSONL including semantic actions')
    args = parser.parse_args(argv)
    if args.belief == 'oracle':
        parser.error('oracle is forbidden for official advice')
    if args.history_len < 0 or min(args.worlds, args.candidates, args.depth) < 1:
        parser.error('invalid history/search sizes')
    torch.set_num_threads(1)
    model = load_policy_model(args.checkpoint, 'cpu')
    session = OfficialSession(args.viewer_id, args.history_len)
    agent = None
    if args.search:
        agent = SearchAgent(model, load_search_belief(args.belief, args.fit, 'cpu'), 'cpu',
                            config=SearchConfig(worlds=args.worlds, candidates=args.candidates,
                                                depth=args.depth, select='argmax' if args.greedy else 'sample'),
                            seed=args.seed, num_threads=1)
        agent.actor.sample = not args.greedy
    try:
        with _policy_rng(args.seed, (agent,) if agent else ()):
            for game in read_games(args.input, args.viewer_id, args.document_sequence):
                result = session.ingest(game)
                if result['status'] == 'decision':
                    if agent is None:
                        agent = Actor(model, result['env'], 'cpu', sample=not args.greedy)
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
