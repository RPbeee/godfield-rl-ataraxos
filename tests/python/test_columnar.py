"""Reference comparisons and column invariants, including carry across updates."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from gfrl.learner import Learner, LearnerConfig
from gfrl.rollout import Collector, RolloutBatch, ColumnBuffer
from gfrl.targets import compute_targets
from legacy.rollout import Collector as OldCollector
from legacy.columnar_rollout import Collector as ColumnarCollector
from legacy.targets import compute_targets as old_targets
from test_training import tiny


def keyed(records):
    groups = {}
    for d in records:
        groups.setdefault((d.env, d.game, d.seat), []).append(d)
    return groups


def compare(old, new):
    a, b = keyed(old), keyed(new)
    assert a.keys() == b.keys()
    for key in a:
        assert len(a[key]) == len(b[key])
        for x,y in zip(a[key], b[key]):
            assert x.outcome == y.outcome
            for k in x.obs:
                np.testing.assert_array_equal(x.obs[k], y.obs[k])
            np.testing.assert_allclose(x.old_value, y.old_value, atol=1e-6)
            assert len(x.stages) == len(y.stages)
            for u,v in zip(x.stages, y.stages):
                assert u.choice == v.choice
                for k in u.prefix:
                    np.testing.assert_array_equal(u.prefix[k], v.prefix[k])
                expected = np.where(u.prefix['mask'], u.old_log_probs, -np.inf).astype(np.float16)
                np.testing.assert_array_equal(expected, v.old_log_probs)
        tx, ax = old_targets(a[key])
        ty, ay = compute_targets(b[key])
        np.testing.assert_allclose(tx, ty, atol=1e-6)
        np.testing.assert_allclose(ax, ay, atol=1e-6)


def test_seeded_reference_and_carry():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(41)
        model = tiny()
        old = OldCollector(deepcopy(model), num_envs=16, threads=2, seed=19)
        new = Collector(deepcopy(model), num_envs=16, threads=2, seed=19)
        torch.manual_seed(88)
        old_one = old.collect(2)
        state = torch.get_rng_state()
        torch.manual_seed(88)
        new_one = new.collect(2)
        torch.testing.assert_close(state, torch.get_rng_state())
        compare(old_one.decisions, new_one.decisions)
        assert isinstance(new_one.decisions, RolloutBatch)
        assert new.buffer.size > 0
        # Real updates between calls; partial decisions must keep previous heads.
        with torch.no_grad():
            for x,y in zip(old.model.parameters(), new.model.parameters()):
                x.add_(.001)
                y.add_(.001)
        # Legacy set iteration orders live/retired sampling batches differently.
        # Greedy actions isolate policy equivalence across the update boundary.
        original_actor = old._actor
        def greedy_actor(*args, **kwargs):
            actor = original_actor(*args, **kwargs)
            actor.sample = False
            return actor
        old._actor = greedy_actor
        for actor in (*old.actors.values(), *new.actors.values()):
            actor.sample = False
        torch.manual_seed(89)
        old_two = old.collect(3)
        torch.manual_seed(89)
        new_two = new.collect(3)
        compare(old_two.decisions, new_two.decisions)
        for batch in (new_one.decisions, new_two.decisions):
            assert batch.obs_num.dtype == torch.float16
            assert batch.stages['old_log_probs'].dtype == torch.float16
            assert all(v.is_contiguous() and v.device.type == 'cpu'
                       for v in (*batch.columns.values(), *batch.stages.values()))
            assert torch.all(batch.outcome != 2)
            np.testing.assert_array_equal(np.bincount(batch.stages['decision'], minlength=len(batch)), batch.stage_count)
            t,a = compute_targets(batch)
            t0,a0 = old_targets(list(batch))
            np.testing.assert_allclose(t,t0,atol=1e-6)
            np.testing.assert_allclose(a,a0,atol=1e-6)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('lv,la', [(0,0),(1,1),(.8,.5),(.13,.97)])
def test_random_targets(lv,la):
    from gfrl.rollout import Decision
    rng = np.random.default_rng(7)
    records = []
    for _ in range(500):
        p = rng.dirichlet([1,1,1]).astype(np.float32)
        records.append(Decision({}, int(rng.integers(5)), int(rng.integers(2)),
                                int(rng.integers(4)), p))
    for rows in keyed(records).values():
        outcome = int(rng.integers(-1,2))
        for d in rows:
            d.outcome = outcome
    a,b = compute_targets(records,lv,la)
    c,d = old_targets(records,lv,la)
    np.testing.assert_allclose(a,c,atol=1e-6)
    np.testing.assert_allclose(b,d,atol=1e-6)


def test_growth_and_compaction():
    buf = ColumnBuffer()
    rows = buf.append(dict(a=np.arange(200), b=np.ones((200,3), np.float16)))
    assert len(rows) == 200
    capacity = buf.capacity
    buf.append(dict(a=np.arange(200,400), b=np.zeros((200,3), np.float16)))
    assert buf.capacity == 2*capacity
    np.testing.assert_array_equal(buf.data['a'][:400], np.arange(400))
    buf.compact({k:v[100:200].clone() for k,v in buf.data.items()})
    assert buf.size == 100
    np.testing.assert_array_equal(buf.data['a'][:100], np.arange(100,200))


def test_columnar_update_matches_reference():
    from legacy.learner import Learner as OldLearner, LearnerConfig as OldConfig
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(12)
        model = tiny()
        rollout = Collector(model, num_envs=16, threads=2, seed=3).collect(2)
        batch = rollout.decisions[:24]
        options = dict(microbatch_size=4, minibatch_size=12, magnet=True, filtering=True)
        old = OldLearner(deepcopy(model), OldConfig(**options))
        new = Learner(deepcopy(model), LearnerConfig(**options))
        np.random.seed(99)
        records = list(batch)
        # The reference stores fp32 log probabilities. Match that arithmetic
        # after applying the new storage precision to both inputs.
        for decision in records:
            for stage in decision.stages:
                stage.old_log_probs = stage.old_log_probs.astype(np.float32)
        expected = old.update(records, 1)
        np.random.seed(99)
        actual = new.update(batch, 1)
        for key,value in expected.items():
            np.testing.assert_allclose(actual[key], value, atol=1e-6)
        for a,b in zip(old.model.parameters(), new.model.parameters()):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        assert old.optimizer_steps == new.optimizer_steps
    finally:
        torch.set_num_threads(previous)


def test_seeded_pool_reference():
    # One environment also makes legacy's unordered model batches immaterial.
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(92)
        model = tiny()
        pool = [Learner(model).pool_entry(1)]
        old = OldCollector(deepcopy(model), num_envs=1, threads=1, seed=11, pool_fraction=1.)
        new = Collector(deepcopy(model), num_envs=1, threads=1, seed=11, pool_fraction=1.)
        torch.manual_seed(93)
        expected = old.collect(3, pool)
        torch.manual_seed(93)
        actual = new.collect(3, pool)
        compare(expected.decisions, actual.decisions)
    finally:
        torch.set_num_threads(previous)


def test_previous_columnar_matches_device_across_updates():
    """Compare the actual previous column layout, including retained batches."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(27)
        model = tiny()
        old = ColumnarCollector(deepcopy(model), num_envs=32, threads=2, seed=12)
        new = Collector(deepcopy(model), num_envs=32, threads=2, seed=12,
                        decision_capacity=16, stage_capacity=32)
        pool = [Learner(model).pool_entry(4)]
        retained = []
        for iteration in range(3):
            torch.manual_seed(70+iteration)
            expected = old.collect(8, pool)
            state = torch.get_rng_state()
            torch.manual_seed(70+iteration)
            actual = new.collect(8, pool)
            torch.testing.assert_close(torch.get_rng_state(), state)
            compare(expected, actual)
            # Ordering matters to target generation and minibatch permutations.
            for key, value in expected.columns.items():
                np.testing.assert_allclose(value, actual.columns[key], atol=1e-6)
            for key, value in expected.stages.items():
                np.testing.assert_array_equal(value, actual.stages[key])
            retained.append((expected, actual))
            with torch.no_grad():
                for a, b in zip(old.model.parameters(), new.model.parameters()):
                    a.add_(.001)
                    b.add_(.001)
        for expected, actual in retained:
            compare(expected, actual)
    finally:
        torch.set_num_threads(previous)
