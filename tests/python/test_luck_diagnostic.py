"""Oracle RNG interventions, crossed estimators and terminal GAE equivalence."""
import argparse
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from luck_diagnostic import (TailSampler, aggregate, analyze_root, candidates, collect, copy_exact,
                             crossed_cov, panel, policy_baseline, regression, summarize)
from gfrl import VecEnv, SearchEnv
from gfrl.actor import Actor
from gfrl.model import Model, ModelConfig
from gfrl.targets import sequence_targets


def args(**kw):
    return argparse.Namespace(**(dict(games=4, roots_per_game=2, target_range=5, num_envs=4,
        threads=1, device='cpu', seed=321, root_selection='reservoir', chance_repeats=4, policy_repeats=4,
        max_candidates=6, min_candidates=2, candidate_nodes=256, prefix_batch=8, mass=.9,
        max_decisions=10000, lambda_v=.8, lambda_a=.5, gamma=1.,
        inference_batch=64, cache_mb=16, world_batch=48) | kw))


def test_crossed_cov_unbiased_exhaustive():
    estimates = []
    means = []
    for signs in itertools.product((-1., 1.), repeat=4):
        c, p = np.array(signs[:2])[:, None], np.array(signs[2:])[None, :]
        x = np.stack([c+p+c*p, c-p, c*p])
        estimates.append(crossed_cov(x))
        means.append(x.mean((1, 2)))
    # E[F|C], E[F|P], interaction are orthogonal in population.
    for key, expected in [('chance', [[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
                          ('policy', [[1, -1, 0], [-1, 1, 0], [0, 0, 0]]),
                          ('interaction', [[1, 0, 1], [0, 0, 0], [1, 0, 1]])]:
        np.testing.assert_allclose(np.mean([s[key] for s in estimates], axis=0), expected, atol=1e-12)
    np.testing.assert_allclose(np.mean([s['mc'] for s in estimates], axis=0),
                               np.cov(means, rowvar=False, ddof=0), atol=1e-12)
    with pytest.raises(ValueError):
        crossed_cov(np.zeros((2, 1, 4)))


def test_mc_corrected_signal_and_crn():
    signal = []
    for signs in itertools.product((-1., 1.), repeat=4):
        c, p = np.array(signs[:2])[:, None], np.array(signs[2:])[None, :]
        x = np.stack([1+c+p, -1+c+p])
        cov = crossed_cov(x)
        M = np.diag([.25, .75])-np.outer([.25, .75], [.25, .75])
        signal.append(cov['mean']@M@cov['mean']-np.trace(M@cov['mc']))
        d = np.array([1., -1.])
        assert d@cov['mc']@d == pytest.approx(0.)
    assert np.mean(signal) == pytest.approx(.75)


def test_online_gae_matches_learner_recursion():
    ps = np.array([[.4, .3, .3], [.6, .1, .3], [.2, .7, .1]])
    for lam, gamma, z in itertools.product((0., .5, 1.), (.9, 1.), (-1, 0, 1)):
        v = ps[:, 0]-ps[:, 1]
        accum, weight = -v[0], 1.
        for value in v[1:]:
            accum += gamma*(1-lam)*weight*value
            weight *= gamma*lam
        accum += gamma*weight*z
        _, advantage = sequence_targets(ps, z, .8, lam, gamma)
        assert accum == pytest.approx(advantage[0], abs=2e-7)


def test_tail_conditioning_and_exact_full_policy_mean():
    rec = dict(candidate_paths=[[0, 0], [1]], candidate_pi=[.2, .4], tail_mass=.4,
               root_policy_prefixes=[dict(path=[], probabilities=[.6, .4]),
                                     dict(path=[0], probabilities=[1/3, 2/3])])
    sampler = TailSampler(rec, 10)
    for i, u in enumerate(np.linspace(.01, .99, 10)):
        # Even a substantially differently rounded live head must not change
        # the frozen retained-prefix law used by conditional subtraction.
        assert sampler.choose(i, np.log([.5, .5]), u) == 0
        assert sampler.choose(i, np.log([.5, .5]), 1-u) == 1
    outcomes = np.array([np.ones((4, 4)), -np.ones((4, 4)), np.zeros((4, 4))])
    baseline = policy_baseline(rec, dict(outcome=outcomes))
    np.testing.assert_allclose(baseline, -.2)


def test_gae_variance_identity_includes_bias_cross_term():
    rows = []
    for i, a in enumerate((-1., -.5, .5, 1.)):
        rows.append(dict(game=i, real_truncated=False, gae=2*a+.2, v=0., candidate_mass=1.,
                         coverage_met=True, top_certified=True,
                         metrics=dict(actual_advantage=a, actual_advantage_mc_variance=0.,
                           gae_conditional_mean=2*a+.2, gae_conditional_mean_mc_variance=0., gae_conditional_variance=0.,
                           gae_advantage_mc_covariance=0., q_policy=0., q_actual=a, post_v=a,
                           next_own_v=a, q_policy_mc_variance=0., q_actual_mc_variance=0.,
                           post_v_mc_variance=0., next_own_v_mc_variance=0., post_q_mc_covariance=0.,
                           next_own_q_mc_covariance=0., signal=.5, outcome_noise=1.)))
    m = aggregate(rows)
    assert m['gae_variance_identity_residual'] == pytest.approx(0.)
    assert m['gae_cross_term'] > 0
    assert m['gae_advantage_correlation'] == pytest.approx(1.)
    assert regression([1, 2, 3], [1, 2, 3])['r2'] == pytest.approx(1.)
    assert summarize([], [], 3, 1)['overall']['estimate']['roots'] == 0


def test_oracle_end_to_end_chunk_invariance_and_censoring():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(6)
        model = Model(ModelConfig(d=16, heads=2, depth=1, ff=32)).eval()
        config = args()
        archive, records, games, targets = collect(config, model)
        assert records and len(games) == config.games
        assert targets is None
        for game in games:
            indices = [r['decision_index'] for r in records if r['game'] == game['g']]
            assert len(set(indices)) == len(indices) == min(config.roots_per_game, game['external_decisions'])
            assert all(0 <= i < game['external_decisions'] for i in indices)
        # Collection sampling is untouched by diagnostic copying or NumPy seeds.
        other_archive, other_records, other_games, _ = collect(config, model)
        assert games == other_games and records == other_records
        _, window_records, window_games, window_targets = collect(args(root_selection='window'), model)
        assert window_games == games
        assert len(window_targets) == config.games
        assert all(r['decision_index'] in window_targets[r['game']] for r in window_records)
        rec = records[0]
        found = candidates(config, model, archive, rec)
        assert found and 0 < rec['candidate_mass'] <= 1.00001
        assert rec['actual_path'] in rec['candidate_paths']
        a = panel(config, model, archive, rec)
        b = panel(args(world_batch=19, inference_batch=7), model, archive, rec)
        for key in ('outcome', 'gae', 'post_v', 'next_own_v', 'status'):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-6)
        assert not a['status'].any()
        if rec['tail_mass'] > 1e-7:
            assert np.all(a['tail_retained_index'] == -1)
        analyze_root(rec, a)
        assert 'metrics' in rec
        assert len(rec['q']) == len(found)+1
        summary = summarize([rec], games, 30, 1)['overall']
        assert summary['bootstrap_valid_fraction']['signal'] < .975
        assert summary['bootstrap_deal_pair_95']['signal'] is None
        bad = dict(a, status=a['status'].copy())
        bad['status'][0, 0, 0] = 2
        censored = {k: v for k, v in rec.items() if k != 'metrics'}
        analyze_root(censored, bad)
        assert censored['metrics_excluded'] == 'incomplete_crossed_grid'
        # No finite floor claim when depth is too short.
        shallow = panel(args(max_decisions=1), model, archive, rec)
        assert np.any(shallow['status'] == 2)
    finally:
        torch.set_num_threads(old)


def test_diagnostic_copy_shapes_and_source_immutability():
    src, dst = VecEnv(2, 1, 3), SearchEnv(2, 1)
    saved = {k: getattr(src, k).copy() for k in ('obs_cards', 'obs_cat', 'mask', 'stage')}
    copy_exact(dst, src, [0, 1])
    for k, value in saved.items():
        np.testing.assert_array_equal(getattr(src, k), value)
        np.testing.assert_array_equal(getattr(dst, k), value)
    with pytest.raises(ValueError):
        copy_exact(dst, src, [0, 1], destinations=[0, 0])
    with pytest.raises((ValueError, IndexError)):
        copy_exact(dst, src, [-1])
