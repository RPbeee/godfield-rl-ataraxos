from dataclasses import replace

import numpy as np
import pytest
import torch

from gfrl import VecEnv
from gfrl.actor import Actor, PREFIX_FIELDS
from gfrl.evaluation import evaluate_policy
from gfrl.model import Model, ModelConfig, OBS_FIELDS


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def tensors(env, fields, rows=slice(None)):
    return {key: torch.tensor(getattr(env, key)[rows].copy()) for key in fields}


@pytest.fixture
def model():
    torch.manual_seed(17)
    return Model(ModelConfig(d=32, depth=2, heads=4, ff=64)).eval()


def test_shapes_dtypes_mask(model):
    env = VecEnv(8, 2, 1)
    with torch.inference_mode():
        encoded = model.encode(tensors(env, OBS_FIELDS))
        assert encoded.shape == (8, 119, 32) and encoded.dtype == torch.float32
        prefix = tensors(env, PREFIX_FIELDS)
        for stage, width in enumerate((10, 19, 3, 21, 21)):
            prefix['stage'].fill_(stage)
            prefix['mask'].fill_(False)
            prefix['mask'][:, :width:2] = True
            lp, value = model.heads(encoded, **prefix)
            assert lp.shape == (8, 21) and value.shape == (8, 3)
            assert lp.dtype == value.dtype == torch.float32
            assert torch.isfinite(lp).all() and torch.isfinite(value).all()
            assert (lp.exp()[~prefix['mask']] == 0).all()
            torch.testing.assert_close(lp.exp().sum(-1), torch.ones(8))
        # ID zero and static feature zero contribute no card embedding.
        table = model.card_embedding.weight + model.card_projection(model.card_features)
        assert not table[0].any()


@pytest.mark.parametrize('chunk', [7, 256])
def test_packing_matches_dense_real_observations(model, chunk):
    env = VecEnv(32, 2, 41)
    rng = np.random.default_rng(42)
    observations, prefixes = [], []
    for step in range(100):
        if step % 10 == 0:
            observations.append(tensors(env, OBS_FIELDS))
            prefixes.append(tensors(env, PREFIX_FIELDS))
        env.step(np.array([rng.choice(np.flatnonzero(mask)) for mask in env.mask],
                          dtype=np.int32))
    obs = {key: torch.cat([batch[key] for batch in observations]) for key in OBS_FIELDS}
    prefix = {key: torch.cat([batch[key] for batch in prefixes]) for key in PREFIX_FIELDS}
    valid = obs['obs_valid'].bool()
    lengths = valid.sum(1)
    assert lengths.max() - lengths.min() > 15
    assert len(lengths) > chunk and len(lengths) % chunk != 0
    calls = []
    hook = model.encoder.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape[:2]))
    try:
        with torch.inference_mode():
            packed = model.encode(obs, pack=True, chunk=chunk)
            expected = sum(len(group) * int(group.max())
                           for group in lengths.sort().values.split(chunk))
            assert model.last_encode_tokens == expected < valid.numel()
            assert sum(b * t for b, t in calls) == expected
            assert len(calls) == (len(lengths) + chunk - 1) // chunk
            dense = model.encode(obs, pack=False)
            assert model.last_encode_tokens == valid.numel()
            torch.testing.assert_close(packed[valid], dense[valid], atol=1e-5, rtol=1e-5)
            assert not packed[~valid].any()
            # Exercise each conditional head using the same real observations.
            for stage in range(5):
                prefix['stage'].fill_(stage)
                lp, value = model.heads(packed, **prefix)
                dense_lp, dense_value = model.heads(dense, **prefix)
                torch.testing.assert_close(lp, dense_lp, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(value, dense_value, atol=1e-5, rtol=1e-5)
    finally:
        hook.remove()


@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
@pytest.mark.parametrize('pack', [True, False])
def test_actor_inference_dtype(model, dtype, pack):
    model.config = replace(model.config, pack=pack)
    env = VecEnv(4, 2, 43)
    actor = Actor(model, env, 'cpu', dtype=dtype, sample=False)
    for _ in range(4):
        result = actor.act()
        assert result.log_prob.dtype == result.value_logits.dtype == np.float32
        assert np.isfinite(result.log_prob).all()
        assert np.isfinite(result.value_logits).all()
        assert env.mask[np.arange(4), result.choices].all()
        env.step(result.choices)


class PublicOnly:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('label_'):
            raise AssertionError('privileged buffer read: ' + name)
        return getattr(self.env, name)


def test_legal_300_steps_no_labels(model):
    env = VecEnv(64, 4, 22)
    actor = Actor(model, PublicOnly(env), 'cpu')
    for _ in range(300):
        result = actor.act()
        assert result.choices.shape == result.log_prob.shape == (64,)
        assert result.choices.dtype == np.int32
        assert result.log_prob.dtype == result.value_logits.dtype == np.float32
        assert np.isfinite(result.value_logits).all()
        assert env.mask[np.arange(64), result.choices].all()
        env.step(result.choices)


def test_hand_equivariance(model):
    env = VecEnv(4, 2, 3)
    obs = tensors(env, OBS_FIELDS)
    prefix = tensors(env, PREFIX_FIELDS)
    prefix['stage'].fill_(1)
    prefix['chosen_kind'].fill_(0)
    prefix['chosen_count'].fill_(2)
    prefix['chosen_cards'][:, :2] = torch.tensor([0, 2])
    # Include STOP and asymmetric card mask to exercise mask permutation too.
    prefix['mask'].fill_(False)
    prefix['mask'][:, [0, 2, 3, 5, 7, 18]] = True
    perm = torch.randperm(18)
    inverse = perm.argsort()
    changed = {k: v.clone() for k, v in obs.items()}
    for key in OBS_FIELDS:
        changed[key][:, 3:21] = obs[key][:, 3:21][:, perm]
    p2 = {k: v.clone() for k, v in prefix.items()}
    p2['mask'][:, :18] = prefix['mask'][:, :18][:, perm]
    slots = prefix['chosen_cards'].long()
    p2['chosen_cards'] = torch.where(slots >= 0, inverse[slots.clamp(min=0)], slots)
    with torch.inference_mode():
        lp, v = model.heads(model.encode(obs), **prefix)
        lp2, v2 = model.heads(model.encode(changed), **p2)
    torch.testing.assert_close(lp2[:, :18], lp[:, :18][:, perm], atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(lp2[:, 18:], lp[:, 18:], atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(v2, v, atol=2e-5, rtol=1e-5)


def test_cache_matches_reencoding(model):
    env = VecEnv(16, 2, 5)
    actor = Actor(model, env, 'cpu', sample=False)
    continuations = 0
    with torch.inference_mode():
        for _ in range(40):
            fresh = np.flatnonzero(env.substep == 0)
            result = actor.act()
            assert actor.last_profile['encoded_envs'] == len(fresh)
            later = np.flatnonzero(env.substep > 0)
            if len(later):
                continuations += len(later)
                full = model.encode(tensors(env, OBS_FIELDS, later))
                cached = actor.token_cache[later]
                torch.testing.assert_close(cached, full, atol=2e-5, rtol=1e-5)
                prefix = tensors(env, PREFIX_FIELDS, later)
                lp, v = model.heads(full, **prefix)
                cp, cv = model.heads(cached, **prefix)
                torch.testing.assert_close(lp, cp, atol=2e-5, rtol=1e-5)
                torch.testing.assert_close(v, cv, atol=2e-5, rtol=1e-5)
            env.step(result.choices)
    assert continuations > 100


def test_cpu_seed_reproducible(model):
    def run():
        torch.manual_seed(123)
        env = VecEnv(8, 2, 6)
        actor = Actor(model, env, 'cpu')
        actions = []
        for _ in range(30):
            choices = actor.act().choices
            actions.append(choices.copy())
            env.step(choices)
        return np.stack(actions)
    np.testing.assert_array_equal(run(), run())


def test_evaluate_policy_and_idle(model):
    # More rows than games includes idle rows from the very first call.
    env = PublicOnly(VecEnv(8, 2, eval_pairs=2))
    actor = Actor(model, env, 'cpu', sample=False)
    while not env.eval_done():
        result = actor.act()
        idle = env.stage < 0
        assert (result.choices[idle] == 0).all()
        assert (result.log_prob[idle] == 0).all()
        assert (result.value_logits[idle] == 0).all()
        env.step(result.choices)
    result = evaluate_policy(actor, 'greedy', pairs=20, num_envs=16, num_threads=2, eval_seed=7)
    assert result['games'] == 40
    assert result['wins'] + result['losses'] + result['draws'] == 40


def test_chosen_buffers(model):
    env = VecEnv(64, 2, 9)
    actor = Actor(model, env, 'cpu')
    saw_hp = False
    for _ in range(150):
        stage = env.stage.copy()
        action = actor.act().choices
        # Prefer exchange when legal, to reach the hp->mp transition.
        # CommandKind::exchange is 4 in the native action ABI.
        action[(stage == 0) & env.mask[:, 4]] = 4
        old_hp = env.chosen_alloc_hp.copy()
        env.step(action)
        assert env.chosen_target.dtype == env.chosen_alloc_hp.dtype == np.int8
        assert np.all(env.chosen_target[env.stage <= 2] == -1)
        assert np.all(env.chosen_alloc_hp[env.stage <= 3] == -1)
        mp = env.stage == 4
        if mp.any():
            saw_hp = True
            assert np.all((env.chosen_target[mp] >= 0) & (env.chosen_target[mp] <= 2))
            assert np.all((env.chosen_alloc_hp[mp] >= 0) & (env.chosen_alloc_hp[mp] <= 20))
            transitioned = (stage == 3) & mp
            np.testing.assert_array_equal(env.chosen_alloc_hp[transitioned], action[transitioned])
            stayed = (stage == 4) & mp & ~env.decision_done
            np.testing.assert_array_equal(env.chosen_alloc_hp[stayed], old_hp[stayed])
    assert saw_hp
