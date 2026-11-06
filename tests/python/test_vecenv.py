import gc
import numpy as np
import pytest
from gfrl import MatchConfig, VecEnv
from bench_vecenv import choices

FIELDS = {
    "stage": (np.int8, ()), "actor_seat": (np.int8, ()),
    "mask": (np.bool_, (21,)), "chosen_kind": (np.int8, ()),
    "chosen_cards": (np.int8, (18,)), "chosen_count": (np.int8, ()),
    "substep": (np.int16, ()), "decision_done": (np.bool_, ()),
    "episode_done": (np.bool_, ()), "outcome": (np.int8, (2,)),
    "truncated": (np.bool_, ()), "episode_id": (np.uint64, ()),
}


def test_zero_copy_and_lifetime():
    env = VecEnv(16, 2, 7)
    arrays = {name: getattr(env, name) for name in FIELDS}
    for name, (dtype, shape) in FIELDS.items():
        a = arrays[name]
        assert isinstance(a, np.ndarray)
        assert a.dtype == dtype and a.shape == (16,) + shape
        assert not a.flags.writeable
        assert np.shares_memory(a, getattr(env, name))
    old = {k: a.copy() for k, a in arrays.items()}
    rng = np.random.default_rng(12)
    for _ in range(20):
        env.step(choices(arrays["mask"], rng))
        for name, a in arrays.items():
            assert np.array_equal(a, getattr(env, name))
    assert any(not np.array_equal(a, old[k]) for k, a in arrays.items())
    del env
    gc.collect()
    assert arrays["mask"].sum() >= 32


def test_200_episodes():
    env = VecEnv(32, 4, 43)
    rng = np.random.default_rng(44)
    results = []
    for _ in range(10000):
        env.step(choices(env.mask, rng))
        rows = env.outcome[env.episode_done].copy()
        results.extend(rows.tolist())
        assert np.all(env.outcome[env.truncated] == 0)
        if len(results) >= 200:
            break
    assert len(results) >= 200
    result = np.asarray(results)
    assert np.all(result.sum(axis=1) == 0)
    assert np.count_nonzero(result == 1) == np.count_nonzero(result == -1)
    assert np.all((result[:, 0] == 0) == (result[:, 1] == 0))
    assert env.stats()["episodes"] == len(results)


def test_invalid_transaction_and_config():
    env = VecEnv(8, 2, 42)
    before = {k: getattr(env, k).copy() for k in FIELDS}
    action = choices(env.mask, np.random.default_rng(2))
    action[-1] = 21
    with pytest.raises(ValueError, match="env 7.*21.*stage"):
        env.step(action)
    for k, a in before.items():
        np.testing.assert_array_equal(a, getattr(env, k))
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(7, dtype=np.int32))
    with pytest.raises(TypeError):
        env.step(np.zeros(8, dtype=np.int64))
    config = MatchConfig()
    config.include_fate = True
    with pytest.raises(ValueError, match="default MatchConfig"):
        VecEnv(1, 1, 42, config=config)
    with pytest.raises(ValueError):
        VecEnv(0, 1, 42)
