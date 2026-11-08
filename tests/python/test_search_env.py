import gc
import numpy as np
import pytest
from gfrl import SearchEnv, VecEnv
from gfrl.model import OBS_FIELDS, Model, ModelConfig
from gfrl.actor import Actor


def rows(n):
    return np.arange(n, dtype=np.int32)


def roots(n=8):
    env = VecEnv(n, 2, 42, track_projection=True)
    assert np.all(env.determinize_support(rows(n)) == 0)
    return env


def load(src, threads=2, depth=512):
    env = SearchEnv(src.num_envs, threads, max_decisions=depth)
    status = env.load_worlds(src, rows(src.num_envs), src.label_opp_hand.astype(np.int32),
                             np.arange(src.num_envs, dtype=np.uint64))
    assert np.all(status == 0)
    return env


def random_choices(env, rng):
    return np.array([rng.choice(np.flatnonzero(m)) if s >= 0 else -999
                     for m, s in zip(env.mask, env.stage)], dtype=np.int32)


def test_root_observe_buffers_and_reload():
    src = roots()
    env = load(src)
    for key in OBS_FIELDS:
        assert np.array_equal(getattr(src, key), getattr(env, key))
    for seat in range(2):
        leaf = env.observe(rows(8), seat)
        assert set(leaf) == set(OBS_FIELDS)
        for key in OBS_FIELDS:
            assert not leaf[key].flags.writeable
            if seat == int(src.actor_seat[0]):
                assert np.array_equal(leaf[key], getattr(env, key))
    leaf = env.observe(rows(8), env.searcher_seat.astype(np.int32))
    saved = {k: getattr(env, k).copy() for k in OBS_FIELDS}
    env.observe(rows(8)[::-1].copy(), 1)
    for k in OBS_FIELDS:
        assert np.array_equal(getattr(env, k), saved[k])
    a = env.obs_num
    env.load_worlds(src, rows(8), src.label_opp_hand.astype(np.int32), np.arange(8, dtype=np.uint64)+99)
    assert np.shares_memory(a, env.obs_num)
    for k in OBS_FIELDS:
        assert np.array_equal(getattr(env, k), saved[k])
    del env
    gc.collect()
    assert np.isfinite(a).all() and leaf['obs_num'].shape[0] == 8


def test_invalid_and_shapes():
    src = roots()
    env = SearchEnv(8, 2)
    assignments = src.label_opp_hand.astype(np.int32)
    assignments[0, 0] = 1
    assignments[1, 1] = -1
    assignments[2, 1] = 19
    status = env.load_worlds(src, rows(8), assignments, np.arange(8, dtype=np.uint64))
    assert np.array_equal(status[:3], [1, 1, 1])
    assert np.all(env.episode_done[:3]) and np.all(env.stage[:3] == -1)
    for _ in range(10):
        env.step(random_choices(env, np.random.default_rng(1)))
    assert np.all(env.decisions[:3] == 0) and np.all(env.episode_done[:3])
    with pytest.raises(ValueError):
        env.observe(rows(1), 0)
    with pytest.raises((IndexError, ValueError)):
        src.determinize_support(np.array([-1], np.int32))
    with pytest.raises(ValueError):
        env.load_worlds(src, rows(7), assignments, np.arange(8, dtype=np.uint64))
    with pytest.raises(TypeError):
        env.load_worlds(src, rows(8), assignments.astype(np.uint8), np.arange(8, dtype=np.uint64))


def test_determinism_terminal_depth_and_actor():
    src = roots()
    a, b = load(src, 1), load(src, 3)
    rng = np.random.default_rng(63)
    previous = np.zeros(8, np.uint64)
    for _ in range(10000):
        if np.all(a.episode_done):
            break
        choices = random_choices(a, rng)
        a.step(choices)
        b.step(choices)
        for key in (*OBS_FIELDS, 'mask', 'stage', 'decisions', 'episode_done', 'result', 'truncated', 'outcome'):
            assert np.array_equal(getattr(a, key), getattr(b, key)), key
        assert np.all(a.decisions - previous == a.decision_done)
        previous = a.decisions.copy()
        active = np.flatnonzero(a.stage >= 0).astype(np.int32)
        if len(active) and _ % 20 == 0:
            observed = a.observe(active, a.actor_seat[active].astype(np.int32))
            for key in OBS_FIELDS:
                assert np.array_equal(observed[key], getattr(a, key)[active])
    else:
        pytest.fail('worlds never terminated')
    assert np.array_equal(a.result, a.outcome[np.arange(8), a.searcher_seat])
    state = {k: getattr(a, k).copy() for k in ('episode_done', 'result', 'decisions', 'truncated')}
    a.step(np.full(8, -999, np.int32))
    for k, expected in state.items():
        assert np.array_equal(getattr(a, k), expected)
    env = load(src, depth=2)
    actor = Actor(Model(ModelConfig(d=32, heads=4, depth=1, ff=64)), env, 'cpu')
    for _ in range(100):
        env.step(actor.act().choices)
        if np.all(env.episode_done):
            break
    assert np.all(env.decisions == 2)
    assert np.all(env.truncated)
    leaf = env.observe(rows(8), env.searcher_seat.astype(np.int32))
    assert set(leaf) == set(OBS_FIELDS)


def test_load_requires_projection_tracking():
    for src in (VecEnv(8, 2, 42), VecEnv(8, 2, 42, track_projection=False)):
        assert not src.track_projection
        with pytest.raises(ValueError, match='src with track_projection=true'):
            load(src)
    src = roots()
    assert src.track_projection
    assert not load(src).track_projection


@pytest.mark.parametrize('history_len', [0, 2, 64])
@pytest.mark.parametrize('evaluation', [False, True])
def test_tracking_preserves_trajectory(history_len, evaluation):
    options = dict(history_len=history_len)
    if evaluation:
        options.update(eval_pairs=12, opponent='external', eval_seed=101)
    a = VecEnv(8, 2, 42, **options)
    b = VecEnv(8, 2, 42, track_projection=True, **options)
    rng = np.random.default_rng(743)
    fields = (*OBS_FIELDS, 'mask', 'stage', 'actor_seat', 'substep',
              'label_opp_hand', 'label_opp_hand_count', 'episode_id',
              'decision_done', 'episode_done', 'truncated', 'outcome',
              'chosen_kind', 'chosen_cards', 'chosen_count', 'chosen_target',
              'chosen_alloc_hp')
    for _ in range(1000):
        for key in fields:
            assert np.array_equal(getattr(a, key), getattr(b, key)), key
        assert a.stats() == b.stats()
        choices = random_choices(a, rng)
        a.step(choices)
        b.step(choices)
    assert a.eval_results() == b.eval_results()
    assert a.stats()['episodes'] > 0
