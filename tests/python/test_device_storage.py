"""Host allocation regression checks; torch CPU storage stands in for device storage."""
import gc
import json
import tracemalloc
from copy import deepcopy

import numpy as np
import pytest
import torch

from gfrl import VecEnv
from gfrl.actor import PREFIX_FIELDS
from gfrl.model import OBS_FIELDS
from gfrl.rollout import Collector
from gfrl.learner import Learner, LearnerConfig
from gfrl.transfer import tensor_batch
from test_training import tiny


class NativeView(np.ndarray):
    """Reject numpy observation/prefix gathers, including transient allocations."""
    def __getitem__(self, rows):
        if self.ndim > 1 and not isinstance(rows, (slice, int)):
            raise AssertionError('host observation/prefix row selection')
        return super().__getitem__(rows)

    def astype(self, *args, **kwargs):
        if self.ndim > 1:
            raise AssertionError('host observation/prefix conversion')
        return super().astype(*args, **kwargs)


class GuardedEnv:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, key):
        if key.startswith('label_'):
            raise AssertionError('privileged label access')
        value = getattr(self.env, key)
        if key in (*OBS_FIELDS, *PREFIX_FIELDS):
            return value.view(NativeView)
        return value


def test_native_transfer_wraps_strided_views_without_copy():
    arrays = {'x': np.arange(96, dtype=np.float32).reshape(8, 12)[:, ::2]}
    tensors = tensor_batch(arrays, 'cpu')
    assert tensors['x'].data_ptr() == arrays['x'].ctypes.data
    assert tensors['x'].stride() == (12, 2)


def test_shared_transfer_once_per_step_and_learning_gathers_on_device(monkeypatch):
    import gfrl.actor as actor_module
    import gfrl.rollout as rollout_module
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = tiny()
        collector = Collector(model, num_envs=16, threads=2, pool_fraction=1.)
        pool = [Learner(deepcopy(model)).pool_entry(1)]
        transfers = []

        def transfer(arrays, device):
            for key in (*OBS_FIELDS, *PREFIX_FIELDS):
                assert arrays[key].ctypes.data == getattr(collector.env, key).ctypes.data
            transfers.append(tuple(arrays))
            return tensor_batch(arrays, device)

        def unexpected(*args, **kwargs):
            raise AssertionError('unexpected host batch assembly or repeated transfer')

        monkeypatch.setattr(rollout_module, 'tensor_batch', transfer)
        monkeypatch.setattr(actor_module, 'tensor_batch', unexpected)
        batch = collector.collect(4, pool)
        assert len(transfers) == batch.timings['steps']
        assert len(collector.actors) > 1
        learner = Learner(model, LearnerConfig(minibatch_size=16, microbatch_size=4))
        original_numpy = torch.Tensor.numpy
        numpy_shapes = []

        def numpy(tensor, *args, **kwargs):
            numpy_shapes.append(tuple(tensor.shape))
            assert tensor.ndim == 1 or tensor.shape == (len(batch), 3)
            return original_numpy(tensor, *args, **kwargs)

        monkeypatch.setattr(np, 'stack', unexpected)
        monkeypatch.setattr(torch.Tensor, 'numpy', numpy)
        learner.update(batch, 1)
        assert len(numpy_shapes) == 5  # env, game, seat, old_value, outcome only
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('n', [64, 128])
def test_collection_host_allocations_are_small_and_bounded(n):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(21)
        env = GuardedEnv(VecEnv(n, 2, 33))
        collector = Collector(tiny(), env=env, decision_capacity=8192, stage_capacity=32768)
        # Warm both output leases and torch/numpy dispatch caches.
        batch = collector.collect(n)
        batch = collector.collect(n)
        measurements = []
        tracemalloc.start()
        try:
            for iteration in range(3):
                gc.collect()
                before = tracemalloc.take_snapshot()
                baseline = tracemalloc.get_traced_memory()[0]
                tracemalloc.reset_peak()
                batch = collector.collect(n)
                current, peak = tracemalloc.get_traced_memory()
                after = tracemalloc.take_snapshot()
                # Positive snapshot deltas measure new retained Python/numpy
                # allocations; peak also catches freed transient host arrays.
                allocated = sum(max(stat.size_diff, 0) for stat in after.compare_to(before, 'traceback'))
                numpy_live = sum(trace.size for trace in after.traces
                                 if trace.domain == np.lib.tracemalloc_domain)
                row = dict(n=n, iteration=iteration+1, decisions=len(batch),
                           new_retained_bytes=allocated, peak_extra_bytes=peak-baseline,
                           numpy_live_bytes=numpy_live)
                measurements.append(row)
                print(json.dumps(row))
                assert allocated < 1_000_000 + n*4096
                assert peak-baseline < 1_000_000 + n*4096
                assert numpy_live < 500_000 + n*1024
                assert all(torch.is_tensor(value) for value in batch.columns.values())
                assert all(torch.is_tensor(value) for value in batch.stages.values())
            assert len(collector._batch_buffers) == 2
        finally:
            tracemalloc.stop()
    finally:
        torch.set_num_threads(previous)
