"""Precision settings, finite buckets and real CPU Inductor equivalence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gfrl import VecEnv
from gfrl.actor import PREFIX_FIELDS
from gfrl.learner import Learner, LearnerConfig
from gfrl.model import Model, ModelConfig, OBS_FIELDS
from gfrl.rollout import Collector
from gfrl.runtime import (ROW_BUCKETS, WIDTH_BUCKETS, compiler_metrics,
                          configure_compile, configure_precision)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny():
    torch.manual_seed(17)
    return Model(ModelConfig(d=32, depth=1, heads=4, ff=128)).eval()


def tensors(env, fields):
    return {key: torch.tensor(getattr(env, key).copy()) for key in fields}


def test_precision_cuda_only_and_disable():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        configure_precision('cuda', True)
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        configure_precision('cpu', False)
        configure_precision('mps', False)
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        configure_precision('cuda:0', False)
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before


@pytest.mark.parametrize('pack', [False, True])
def test_bucket_equivalence_and_finite_dummy_rows(monkeypatch, pack):
    # Exercise padding independently of code generation, including every bucket
    # boundary and a chunk larger than the compiled cap.
    monkeypatch.setattr(torch, 'compile', lambda module, **kwargs: module)
    model = tiny()
    env = VecEnv(259, 2, 41)
    obs = tensors(env, OBS_FIELDS)
    obs['obs_valid'][0].zero_()
    obs['obs_valid'][1, :119] = True
    prefix = tensors(env, PREFIX_FIELDS)
    with torch.inference_mode():
        expected = model.encode(obs, pack=pack, chunk=257)
        lp, value = model.heads(expected, **prefix)
        configure_compile(model, 'cpu', 'on')
        calls, finite = [], []
        def inspect(module, inputs, output):
            calls.append(inputs[0].shape[:2])
            finite.append(bool(torch.isfinite(output).all()))
        handle = model.encoder.register_forward_hook(inspect)
        try:
            actual = model.encode(obs, pack=pack, chunk=257)
            actual_lp, actual_value = model.heads(actual, **prefix)
        finally:
            handle.remove()
        assert calls and all(r in ROW_BUCKETS and w in WIDTH_BUCKETS for r, w in calls)
        assert all(finite)
        assert not actual[0].any()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual_lp, lp, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual_value, value, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('rows, expected_rows', [(1, 32), (32, 32), (33, 64), (64, 64),
                                               (65, 128), (128, 128), (129, 256), (256, 256)])
def test_all_bucket_boundaries(rows, expected_rows):
    model = tiny()
    calls = []
    def encoder(x, mask):
        calls.append((x.shape, mask.clone()))
        assert not mask.all(1).any()
        return x + 1
    model._compiled_encoder = encoder
    for width in (1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113, 119):
        x = torch.randn(rows, width, 32)
        mask = torch.zeros(rows, width, dtype=torch.bool)
        actual = model._run_encoder(x, mask)
        torch.testing.assert_close(actual, x + 1)
        shape, padding = calls[-1]
        assert shape == (expected_rows, ((width+15)//16)*16, 32)
        assert padding[rows:, 0].eq(False).all()
        assert padding[rows:, 1:].all()


def test_cpu_compile_outputs_reuse_and_one_ppo_step():
    model = tiny()
    expected = deepcopy(model)
    env = VecEnv(8, 2, 13)
    obs, prefix = tensors(env, OBS_FIELDS), tensors(env, PREFIX_FIELDS)
    configure_compile(model, 'cpu', 'on')
    before = compiler_metrics()
    with torch.inference_mode():
        encoded = model.encode(obs)
        reference = expected.encode(obs)
        torch.testing.assert_close(encoded, reference, atol=1e-4, rtol=1e-5)
        for stage in range(5):
            prefix['stage'].fill_(stage)
            for a, b in zip(model.heads(encoded, **prefix), expected.heads(reference, **prefix)):
                torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-5)
        first = compiler_metrics()
        assert first['compile_graphs'] > before['compile_graphs']
        # Different real row count and width, but the same 32 x 16 bucket.
        smaller = {key: value[:7].clone() for key, value in obs.items()}
        smaller['obs_valid'][:, 12:] = False
        model.encode(smaller)
        second = compiler_metrics()
        assert second['compile_graphs'] == first['compile_graphs']
        assert second['compile_recompiles'] == first['compile_recompiles']

    smaller = {key: value.clone() for key, value in smaller.items()}
    model.encode(obs).square().mean().backward()
    first_grad = compiler_metrics()
    model.zero_grad(set_to_none=True)
    model.encode(smaller).square().mean().backward()
    second_grad = compiler_metrics()
    assert second_grad['compile_graphs'] == first_grad['compile_graphs']
    assert second_grad['compile_recompiles'] == first_grad['compile_recompiles']

    # Fixed behavior-policy data and identical numpy permutations give the same
    # PPO loss and optimizer result through a compiled forward/backward.
    rollout = Collector(expected, num_envs=8, threads=2, seed=123, pool_fraction=0).collect(2)
    eager = Learner(expected, LearnerConfig(minibatch_size=16, microbatch_size=8))
    compiled = Learner(deepcopy(expected), LearnerConfig(minibatch_size=16, microbatch_size=8))
    configure_compile(compiled.model, 'cpu', 'on')
    with torch.inference_mode():
        compiled.model.encode(obs)
    np.random.seed(1)
    eager_metrics = eager.update(rollout.decisions, 1)
    np.random.seed(1)
    compiled_metrics = compiled.update(rollout.decisions, 1)
    for key in ('policy_loss', 'kl_loss', 'value_loss', 'entropy', 'ratio_mean'):
        assert compiled_metrics[key] == pytest.approx(eager_metrics[key], abs=1e-4)
    for a, b in zip(compiled.model.parameters(), eager.model.parameters()):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    # Replays see in-place optimizer updates; weights are never frozen constants.
    with torch.inference_mode():
        torch.testing.assert_close(compiled.model.encode(obs), eager.model.encode(obs),
                                   atol=1e-4, rtol=1e-4)


def test_snapshot_is_eager_and_state_dict_stays_portable():
    model = tiny()
    names = list(model.state_dict())
    configure_compile(model, 'cpu', 'on')
    assert list(model.state_dict()) == names
    snapshot = deepcopy(model)
    assert snapshot._compiled_encoder is None
    assert model._compiled_encoder.encoder is model.encoder
    for a, b in zip(snapshot.parameters(), model.parameters()):
        assert a.data_ptr() != b.data_ptr()
        torch.testing.assert_close(a, b)
    configure_compile(model, 'cpu', 'off')
    assert model._compiled_encoder is None
    with pytest.raises(ValueError, match='MPS'):
        configure_compile(model, 'mps', 'on')


@pytest.mark.parametrize('initial, resumed', [('off', 'on'), ('on', 'off')])
def test_resume_runtime_options_are_caller_controlled(tmp_path, initial, resumed):
    spec = importlib.util.spec_from_file_location('train_runtime',
                Path(__file__).parents[2] / 'scripts/train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    argv = ['--d', '32', '--depth', '1', '--num-envs', '8', '--threads', '2',
            '--run-dir', str(tmp_path), '--max-hours', '1e-12']
    path = train.run(train.parse_args(argv + ['--compile', initial, '--no-tf32',
                    '--compile-mode-actor', 'default', '--compile-mode-learner', 'default']))
    train.run(train.parse_args(argv + ['--resume', '--compile', resumed, '--tf32',
              '--compile-mode-actor', 'reduce-overhead', '--compile-mode-learner', 'max-autotune']))
    state = torch.load(path, weights_only=False)
    for config in (state['config'], json.loads((tmp_path/'config.json').read_text())):
        assert config['compile'] == resumed
        assert config['tf32'] is True
        assert config['compile_mode_actor'] == 'reduce-overhead'
        assert config['compile_mode_learner'] == 'max-autotune'
