"""CPU rollout storage preserves training and bounds learner row transfers."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from gfrl.learner import Learner, LearnerConfig
from gfrl.rollout import Collector
from test_training import tiny, train_module, threads


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_storage_training_equivalence(device, monkeypatch):
    def run(storage):
        torch.manual_seed(17)
        np.random.seed(17)
        if device == 'mps':
            torch.mps.manual_seed(17)
        learner = Learner(tiny(), LearnerConfig(minibatch_size=16, microbatch_size=4,
                                               filtering=True, magnet=True), device)
        collector = Collector(learner.model, device=device, rollout_storage=storage,
                              num_envs=16, threads=2, seed=23, pool_fraction=.5,
                              decision_capacity=8, stage_capacity=16)
        original_prepare = learner._prepare
        prepared = []

        def prepare(batch, rows, *args):
            prepared.append(len(rows))
            assert batch.env.device.type == ('cpu' if storage == 'cpu' else device)
            return original_prepare(batch, rows, *args)

        monkeypatch.setattr(learner, '_prepare', prepare)
        history = []
        pool = [learner.pool_entry(0)]
        pending = False
        for iteration in range(1, 4):
            batch = collector.collect(2, pool)
            assert batch.cpu_storage == (storage == 'cpu')
            assert batch[:2].cpu_storage == batch.cpu_storage
            pending |= batch.timings['pending_decisions'] > 0
            assert collector.current.device == collector.buffer.device == batch.env.device
            assert all(v.device == batch.env.device for v in batch.stages.values())
            metrics = learner.update(batch, iteration)
            history.append({k: v for k, v in metrics.items() if not k.endswith('_s')})
            del batch
        assert pending
        assert max(prepared) <= (4 if storage == 'cpu' else 16)
        return history, deepcopy(learner.model.state_dict()), deepcopy(learner.ema.state_dict())

    expected, model, ema = run('device')
    actual, cpu_model, cpu_ema = run('cpu')
    for a, b in zip(actual, expected):
        for key in a:
            assert a[key] == pytest.approx(b[key], abs=1e-6, rel=1e-5)
    for a, b in ((model, cpu_model), (ema, cpu_ema)):
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('first,second', [('device', 'cpu'), ('cpu', 'device')])
def test_resume_switches_storage(tmp_path, first, second):
    train = train_module()
    argv = ['--device', 'cpu', '--num-envs', '16', '--threads', '2',
            '--games-per-iter', '2', '--iterations', '1', '--d', '32', '--depth', '1',
            '--eval-every', '0', '--microbatch-size', '4', '--run-dir', str(tmp_path)]
    checkpoint = train.run(train.parse_args(argv + ['--rollout-storage', first]))
    initial = torch.load(checkpoint, weights_only=False)
    assert initial['config']['rollout_storage'] == first
    train.run(train.parse_args(argv + ['--resume', '--rollout-storage', second]))
    resumed = torch.load(checkpoint, weights_only=False)
    assert resumed['iteration'] == 2
    assert resumed['optimizer_steps'] > initial['optimizer_steps']
    assert resumed['config']['rollout_storage'] == second
    assert json.loads((tmp_path/'config.json').read_text())['rollout_storage'] == second


def test_rollout_storage_defaults_and_validation():
    train = train_module()
    assert train.parse_args([]).rollout_storage == 'device'
    with pytest.raises(SystemExit):
        train.parse_args(['--rollout-storage', 'invalid'])
