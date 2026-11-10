"""The actual terminal strings are part of the information boundary."""
import json

import numpy as np
import pytest

from gfrl import VecEnv, _core
from gfrl.play import Presenter, QuitGame, main, parser, read_choice


def test_hidden_worlds_have_identical_terminal_strings():
    # Different empty hidden hands and future RNGs, on main and response roots.
    rng = np.random.default_rng(8)
    seen = set()
    for seed in range(12):
        env = VecEnv(1, 1, eval_pairs=1, opponent='external', eval_seed=seed,
                     track_projection=True)
        for _ in range(150):
            if int(env.substep[0]) == 0 and env.determinize_support(np.array([0], np.int32))[0] == 0:
                seat = int(env.actor_seat[0])
                p = Presenter(seat)
                before_view = env.play_view(0, seat)
                before_hand = env.play_hand(0, seat)
                before = p.board(before_view, hand=before_hand)
                options = p.options(env, before_hand)
                hidden = env.play_hand(0, 1 - seat)
                _core._test_replace_hidden_hand(env, 0, seed + 99)
                assert env.play_hand(0, 1 - seat) != hidden
                assert p.board(env.play_view(0, seat), hand=env.play_hand(0, seat)) == before
                assert p.options(env, env.play_hand(0, seat)) == options
                serialized = json.dumps(env.play_view(0, seat))
                assert 'token' not in serialized and 'rng' not in serialized
                seen.add(before_view['phase'])
                break
            env.step(np.array([rng.choice(np.flatnonzero(env.mask[0]))], np.int32))
    assert 'main' in seen
    # Reach a supported response independently; main was tested above.
    for seed in range(30):
        env = VecEnv(1, 1, eval_pairs=1, opponent='external', eval_seed=seed)
        for _ in range(250):
            if env.eval_results():
                break
            if env.substep[0] == 0 and env.play_view(0, 0)['phase'] == 'response':
                seat = int(env.actor_seat[0]); p = Presenter(seat)
                before = p.board(env.play_view(0, seat))
                _core._test_replace_hidden_hand(env, 0, 199)
                assert p.board(env.play_view(0, seat)) == before
                return
            env.step(np.array([rng.choice(np.flatnonzero(env.mask[0]))], np.int32))
    pytest.fail('no response root encountered')


def test_options_cover_every_mask_entry_and_all_stages():
    rng = np.random.default_rng(13)
    seen = set()
    for seed in range(30):
        env = VecEnv(1, 1, eval_pairs=1, opponent='external', eval_seed=seed)
        for _ in range(500):
            if env.eval_results():
                break
            seat = int(env.actor_seat[0]); p = Presenter(seat)
            hand = env.play_hand(0, seat)
            options = p.options(env, hand)
            assert [c for c, _ in options] == np.flatnonzero(env.mask[0]).tolist()
            assert len(set(label for _, label in options)) == len(options)
            seen.add(int(env.stage[0]))
            # Prefer exchange to cover both allocation stages.
            choice = 4 if env.stage[0] == 0 and env.mask[0, 4] else rng.choice([c for c, _ in options])
            env.step(np.array([choice], np.int32))
    assert seen == set(range(5))


def test_input_validation_help_board_and_quit():
    replies = iter(('x', '0', '99', 'h', 's', '2'))
    output = []
    assert read_choice([(3, 'a'), (8, 'b')], lambda: 'BOARD',
                       input_fn=lambda _: next(replies), output=output.append) == 8
    assert 'BOARD' in output
    assert sum('番号を入力' in s for s in output) == 3
    with pytest.raises(QuitGame):
        read_choice([], lambda: '', input_fn=lambda _: 'q')
    def eof(_):
        raise EOFError
    with pytest.raises(QuitGame):
        read_choice([], lambda: '', input_fn=eof)


def test_trace_preserves_play_and_records_terminal_before_reset():
    a = VecEnv(1, 1, eval_pairs=1, opponent='external', eval_seed=12)
    b = VecEnv(1, 1, eval_pairs=1, opponent='external', eval_seed=12)
    a.enable_play_trace(0, 0)
    rng = np.random.default_rng(9)
    terminal = False
    while not a.eval_done():
        assert a.play_view(0, 0) == b.play_view(0, 0)
        choice = np.array([rng.choice(np.flatnonzero(a.mask[0]))], np.int32)
        a.step(choice); b.step(choice)
        for e in a.take_play_trace(0):
            assert 'token' not in json.dumps(e)
            assert e['after']['decisions'] > e['before']['decisions']
            assert Presenter(0).event(e)
            terminal |= e['after']['terminated'] or e['after']['truncated']
        assert b.take_play_trace(0) == []
    assert terminal
    assert a.eval_results() == b.eval_results()
    assert a.stats() == b.stats()


def test_defaults_and_oracle_rejection():
    args = parser().parse_args(['--checkpoint', 'x'])
    assert (args.worlds, args.depth, args.beta, args.leaf_view, args.search_response) == (64, 3, .1, 'actor', True)
    with pytest.raises(SystemExit):
        main(['--checkpoint', 'x', '--search', '--belief', 'oracle'])


def test_catalog_names_and_effects_are_readable():
    p = Presenter(0)
    assert p.name(19) == '鎖ガマ'
    assert 'HP回復 15' in p.card(193)
    assert '・・' not in p.card(19)
