"""Storage reuse must preserve stage ordering, carry, and retained batches."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from gfrl.rollout import Collector, ColumnBuffer
from gfrl.transfer import tensor_batch


def test_finished_storage_reuse_and_retained_batches():
    collector = Collector.__new__(Collector)
    collector.buffer, collector.stage_buffer = ColumnBuffer(), ColumnBuffer()
    collector._batch_buffers = []

    def finish(offset):
        collector.buffer.size = collector.stage_buffer.size = 0
        collector.buffer.append(dict(env=np.arange(4), outcome=np.array([1, 2, -1, 0]),
                                     obs_num=np.arange(12).reshape(4, 3) + offset))
        collector.stage_buffer.append(dict(decision=np.array([1, 0, 3, 2, 0, 1]),
                                           choice=np.arange(6) + offset))
        collector.current = np.array([-1, 1, -1, -1])
        batch, pending = collector._finish_batch()
        assert pending == 1
        np.testing.assert_array_equal(batch.env, [0, 2, 3])
        np.testing.assert_array_equal(batch.stage_count, [2, 1, 1])
        np.testing.assert_array_equal(batch.stage_start, [0, 2, 3])
        np.testing.assert_array_equal(batch.stages['choice'], np.array([1, 4, 3, 2]) + offset)
        np.testing.assert_array_equal(batch.stages['decision'], [0, 0, 1, 2])
        assert collector.buffer.size == 1
        np.testing.assert_array_equal(collector.buffer.data['obs_num'][0], np.arange(3, 6) + offset)
        assert collector.stage_buffer.size == 2
        np.testing.assert_array_equal(collector.stage_buffer.data['decision'][:2], [0, 0])
        np.testing.assert_array_equal(collector.current, [-1, 0, -1, -1])
        return batch

    first = finish(0)
    collection_storage = (collector.buffer.data.copy(), collector.stage_buffer.data.copy())
    output_storage = tuple(buf.data.copy() for buf in collector._batch_buffers[0][:2])
    second = finish(100)
    np.testing.assert_array_equal(first.obs_num, np.arange(12).reshape(4, 3)[[0, 2, 3]])
    assert len(collector._batch_buffers) == 2
    del first
    third = finish(200)
    assert len(collector._batch_buffers) == 2
    for before, after in zip(collection_storage, (collector.buffer.data, collector.stage_buffer.data)):
        assert all(after[k] is v for k, v in before.items())
    for before, buf in zip(output_storage, collector._batch_buffers[0][:2]):
        assert all(buf.data[k] is v for k, v in before.items())
    assert np.shares_memory(third.obs_num, output_storage[0]['obs_num'])
    np.testing.assert_array_equal(second.stages['choice'], [101, 104, 103, 102])


def test_transfer_uses_blocking_to_without_pinned_allocation(monkeypatch):
    original_to = torch.Tensor.to
    calls = []

    def to(tensor, device, **kwargs):
        calls.append((device.type, kwargs))
        return original_to(tensor, 'cpu')

    def no_empty(*args, **kwargs):
        raise AssertionError('transfer must not allocate pinned storage')

    monkeypatch.setattr(torch.Tensor, 'to', to)
    monkeypatch.setattr(torch, 'empty', no_empty)
    arrays = dict(x=np.arange(12, dtype=np.float16).reshape(3, 4)[:, ::2],
                  mask=np.array([True, False]), empty=np.empty((0, 3), dtype=np.int64))
    result = tensor_batch(arrays, 'cuda')
    assert calls == [('cuda', {})] * len(arrays)
    for key, value in arrays.items():
        np.testing.assert_array_equal(result[key].numpy(), value)


def test_tsubame_wrapper_preserves_arguments_and_environment():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OMP_NUM_THREADS='7')
    code = ('import json, os, sys; '
            'print(json.dumps([sys.argv[1:], sys.prefix, '
            '[os.environ[k] for k in ("MALLOC_MMAP_THRESHOLD_", '
            '"MALLOC_TRIM_THRESHOLD_", "MALLOC_TOP_PAD_", "OMP_NUM_THREADS")]]))')
    result = subprocess.run([str(root/'scripts/tsubame_run.sh'), '-c', code,
                             'space in argument', '--flag', '$(literal)'],
                            env=env, check=True, capture_output=True, text=True)
    args, prefix, settings = json.loads(result.stdout)
    assert args == ['space in argument', '--flag', '$(literal)']
    assert Path(prefix).resolve() == (root/'.venv').resolve()
    assert settings == ['17179869184', '68719476736', '1073741824', '7']
