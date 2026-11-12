import math

import numpy as np
import pytest

from gfrl import VecEnv, _core
from gfrl.evaluation import (wilson_interval, pair_bootstrap, summarize,
                             evaluate_policy, evaluate_bots)


def records(scores):
    return [dict(g=g, pair=g//2, agent_seat=g%2, result=r,
                 truncated=False, decisions=10, hp=(20, 10))
            for g, r in enumerate(scores)]


def test_wilson():
    assert wilson_interval(0, 0) == (0, 1)
    assert wilson_interval(50, 100) == pytest.approx((0.4038298286, 0.5961701714))
    assert wilson_interval(0, 10) == pytest.approx((0, 1.96**2/(10+1.96**2)))
    assert wilson_interval(10, 10) == pytest.approx((10/(10+1.96**2), 1))
    # Five draws are 2.5 effective wins.
    assert wilson_interval(2.5, 5)[0] + wilson_interval(2.5, 5)[1] == pytest.approx(1)
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_bootstrap():
    assert pair_bootstrap([]) == (0, 1)
    assert pair_bootstrap(records([1, -1]*4)) == (0.5, 0.5)
    assert pair_bootstrap(records([0, 0]*4)) == (0.5, 0.5)
    assert pair_bootstrap(records([1, 1]*4)) == (1, 1)
    assert pair_bootstrap(records([-1, -1]*4)) == (0, 0)
    # Two pairs with scores 0,1: bootstrap means 0,0.5,1; exact 95% bounds 0,1.
    r = records([-1, -1, 1, 1])
    assert pair_bootstrap(r, seed=7) == (0, 1)
    assert pair_bootstrap(r*1, seed=7) == pair_bootstrap(list(reversed(r)), seed=7)
    with pytest.raises(ValueError, match='complete'):
        pair_bootstrap(records([1]))


def test_summary():
    s = summarize(records([1, -1, 0, 0]))
    assert s == dict(games=4, win_rate=.5, wilson_interval=wilson_interval(2, 4),
                     bootstrap_interval=(.5, .5), wins=1, losses=1, draws=2,
                     truncation_rate=0, mean_decisions=10, truncations=0,
                     mean_game_length=10, mean_win_decisions=10, mean_loss_decisions=10,
                     first_win_rate=.75, second_win_rate=.25)
    assert summarize([])['win_rate'] is None


def test_partial_summary_uses_only_complete_pairs_for_bootstrap():
    rows = records([1, 1, -1, -1, 0])
    partial = summarize(rows[:3], complete_pairs_only=True)
    assert partial['games'] == 3 and partial['win_rate'] == 2 / 3
    assert partial['wilson_interval'] == wilson_interval(2, 3)
    assert partial['bootstrap_interval'] == (1, 1)
    assert partial['first_win_rate'] == .5 and partial['second_win_rate'] == 1
    assert partial['mean_decisions'] == 10 and partial['truncation_rate'] == 0
    assert summarize(rows[2:3], complete_pairs_only=True)['bootstrap_interval'] == (0, 1)
    assert summarize(rows[:4], complete_pairs_only=True) == summarize(rows[:4])
    assert summarize([], complete_pairs_only=True) == summarize([])
    with pytest.raises(ValueError, match='complete'):
        summarize(rows[:3])


def test_progress_deadline_and_completion(monkeypatch):
    import gfrl.evaluation as evaluation
    clock = [0.0]
    monkeypatch.setattr(evaluation, 'monotonic', lambda: clock[0])
    rows, snapshots = [], []

    class Env:
        def eval_results(self):
            return list(rows)

    reporter = evaluation._EvaluationProgress(snapshots.append, 5)
    rows.extend(records([1]))
    clock[0] = 4.9
    reporter(Env())
    assert snapshots == []
    clock[0] = 5
    reporter(Env())
    assert snapshots == [summarize(rows, complete_pairs_only=True)]
    rows.extend(records([1, -1])[1:])
    clock[0] = 6
    reporter(Env(), final=True)
    assert snapshots[-1] == summarize(rows)
    reporter(Env(), final=True)
    assert [s['games'] for s in snapshots] == [1, 2]
    for interval in (0, -1, float('inf'), float('nan')):
        with pytest.raises(ValueError, match='progress_interval_s'):
            evaluation._EvaluationProgress(snapshots.append, interval)


def test_policy_and_bots():
    rng = np.random.default_rng(4)
    seen = []
    def policy(env):
        active = env.stage >= 0
        assert np.all(env.mask[active].sum(axis=1) >= 2)
        assert not env.mask[~active].any()
        seen.extend(env.actor_seat[active].tolist())
        return np.array([rng.choice(np.flatnonzero(mask)) if mask.any() else -999
                         for mask in env.mask], dtype=np.int32)
    s = evaluate_policy(policy, 'greedy', 4, 11, 2, 123)
    assert seen and s['games'] == 8 and s['wins'] + s['losses'] + s['draws'] == 8
    assert s['win_rate'] == (s['wins'] + s['draws']/2)/8
    assert s['win_rate'] == pytest.approx((s['first_win_rate']+s['second_win_rate'])/2)
    assert 0 <= s['truncation_rate'] <= 1 and 0 < s['mean_decisions'] <= 512
    raw = _core.evaluate_bots('random', 'greedy', 4, 2, 123)
    assert summarize(raw) == evaluate_bots('random', 'greedy', 4, 1, 123)
    assert len(raw) == 8 and all(set(r) == {'g','pair','agent_seat','result','truncated','decisions','hp'} for r in raw)


def test_training_bot_seat_and_errors():
    env = VecEnv(3, 2, 5, controllers=('external', 'greedy'))
    for _ in range(50):
        assert np.all(env.actor_seat == 0)
        env.step(np.argmax(env.mask, axis=1).astype(np.int32))
    assert env.stats()['automatic_decisions'] > 0
    assert not env.eval_done() and env.eval_results() == []
    env = VecEnv(3, 1, eval_pairs=0)
    assert env.eval_done() and np.all(env.stage == -1) and not env.mask.any()
    env.step(np.full(3, -999, dtype=np.int32))
    bots = VecEnv(1, 1, controllers=('random', 'greedy'))
    assert bots.stage[0] == -1 and not bots.mask.any()
    episodes = bots.stats()['episodes']
    bots.step(np.array([-999], dtype=np.int32))
    assert bots.stats()['episodes'] == episodes + 1
    with pytest.raises(ValueError):
        evaluate_bots('bad', 'greedy', 1, 1, 0)
