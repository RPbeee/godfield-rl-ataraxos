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


def test_observations():
    from gfrl import obs_schema, card_features
    features = card_features()
    assert features.shape == (297, 103)
    assert features.dtype == np.float32 and np.isfinite(features).all()
    assert not features[0].any() and not features.flags.writeable
    env = VecEnv(8, 2, 42, history_len=7)
    schema = env.obs_schema()
    assert schema == obs_schema(7)
    t, c, f = schema['T'], schema['C'], schema['F']
    assert t == 62 and len(schema['num']) == f and len(schema['cat']) == c
    fields = {'obs_type': (np.int8, (8,t)), 'obs_cards': (np.int16,(8,t,8)),
              'obs_aux_card': (np.int16,(8,t)), 'obs_cat': (np.int8,(8,t,c)),
              'obs_num': (np.float32,(8,t,f)), 'obs_valid': (np.bool_,(8,t)),
              'label_opp_hand': (np.uint8,(8,297)), 'label_opp_hand_count': (np.uint8,(8,))}
    arrays = {name:getattr(env,name) for name in fields}
    for name, (dtype, shape) in fields.items():
        assert arrays[name].shape == shape and arrays[name].dtype == dtype
        assert not arrays[name].flags.writeable
        assert np.shares_memory(arrays[name],getattr(env,name))
    before = env.obs_num.copy()
    rng = np.random.default_rng(17)
    for _ in range(40):
        old = {name:a.copy() for name,a in arrays.items()}
        env.step(choices(env.mask,rng))
        # Continuation stages preserve every observation and privileged label.
        for row in np.flatnonzero(~env.decision_done):
            for name,a in arrays.items():
                np.testing.assert_array_equal(a[row],old[name][row])
        assert np.all(env.label_opp_hand.sum(axis=1) == env.label_opp_hand_count)
        assert env.obs_cards.max() <= 296
    assert not np.array_equal(before,env.obs_num)
    del env
    gc.collect()
    assert arrays['obs_valid'][:,0].all()
    assert VecEnv(1,1,1,history_len=0).obs_type.shape == (1,55)
    with pytest.raises(ValueError):
        VecEnv(1,1,1,history_len=4097)
