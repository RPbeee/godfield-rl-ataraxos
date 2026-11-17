"""Oracle RNG interventions, crossed estimators and terminal GAE equivalence."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from luck_diagnostic import (TailSampler, aggregate, analyze_root, candidates, collect, copy_exact,
                             crossed_cov, panel, policy_baseline, regression, summarize,
                             wide_candidates, analyze_wide_root, summarize_wide, save_report,
                             parser, main, _json_default)
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
        # Extra arms preserve every original outcome, tail and metric. Their
        # proposal stream must not consume the collection/panel RNG streams.
        baseline_rec = copy.deepcopy(rec)
        config_wide = args(wide_candidates=3, wide_draws=128, wide_method='prefix-uniform',
                           wide_temperature=2., wide_max_pi=1.)
        wide_candidates(config_wide, model, archive, rec)
        assert rec['wide']['actions']
        again = copy.deepcopy(baseline_rec)
        wide_candidates(config_wide, model, archive, again)
        assert rec['wide'] == again['wide']
        for action in rec['wide']['actions']:
            assert action['path'] not in rec['candidate_paths']
            replay = SearchEnv(1, 1)
            copy_exact(replay, archive, [rec['archive_row']])
            for choice in action['path']:
                assert replay.mask[0, choice]
                replay.step(np.array([choice], np.int32))
            assert replay.decisions[0] == 1
        wdata = panel(config_wide, model, archive, rec)
        K = len(found)
        for key in ('outcome', 'gae', 'post_v', 'next_own_v', 'status'):
            np.testing.assert_array_equal(wdata[key][:K+1], a[key])
        np.testing.assert_array_equal(wdata['tail_retained_index'], a['tail_retained_index'])
        np.testing.assert_array_equal(policy_baseline(rec, wdata), policy_baseline(rec, a))
        analyze_root(rec, wdata)
        assert rec['metrics'] == baseline_rec['metrics']
        assert rec['q'] == baseline_rec['q']
        assert 'summary' in rec['wide']
        extra_failed = dict(wdata, status=wdata['status'].copy())
        extra_failed['status'][-1, 0, 0] = 2
        analyze_root(rec, extra_failed)
        assert rec['metrics'] == baseline_rec['metrics']
        assert rec['wide']['metrics_excluded'] == 'incomplete_crossed_grid'
        analyze_root(rec, wdata)
        tempered = copy.deepcopy(baseline_rec)
        wide_candidates(args(wide_candidates=2, wide_draws=128, wide_method='tempered',
                             wide_temperature=2., wide_max_pi=1.), model, archive, tempered)
        assert tempered['wide']['actions']
        capped = copy.deepcopy(baseline_rec)
        wide_candidates(args(wide_candidates=2, wide_draws=16, wide_method='prefix-uniform',
                             wide_temperature=2., wide_max_pi=1e-30), model, archive, capped)
        assert not capped['wide']['actions']
        assert capped['wide']['shortfall'] == 2 and capped['wide']['draws'] == 16
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


def wide_fixture():
    rec = dict(game=0, decision_index=3, decisions_to_end=12, real_truncated=False,
               phase='main', progress=.2, candidate_paths=[[0], [1]],
               candidate_pi=[.6, .3], tail_mass=.1,
               wide=dict(actions=[dict(path=[2], pi=.001, log_pi=np.log(.001)),
                                  dict(path=[3], pi=1e-8, log_pi=np.log(1e-8))],
                         requested=2, shortfall=0, draws=5))
    # Two top arms, original tail, two wide actions. Wide 0 is the genuinely
    # best action; wide 1 is bad. Post V reverses them to detect a bad head.
    outcomes = np.broadcast_to(np.array([0., .1, -.1, .8, -.8])[:, None, None], (5, 4, 4)).copy()
    post = np.broadcast_to(np.array([0., .1, 0., -.8, .8])[:, None, None], (5, 4, 4)).copy()
    return rec, dict(outcome=outcomes, post_v=post, status=np.zeros((5, 4, 4), np.int8))


def test_wide_estimands_and_value_discrimination():
    rec, data = wide_fixture()
    analyze_wide_root(rec, data)
    summary = rec['wide']['summary']
    assert policy_baseline(rec, data).mean() == pytest.approx(.02)
    assert rec['wide']['actions'][0]['advantage'] == pytest.approx(.78)
    assert summary['distribution/all/mean_advantage'] == pytest.approx(-.02)
    assert summary['distribution/lt_1e-6/actions'] == 1
    assert summary['discovery/2/heldout_gap'] == pytest.approx(.7)
    assert summary['discovery/2/evidence_fraction'] == 1
    assert summary['discovery/2/value_gain_vs_wide_mean'] == pytest.approx(-.8)
    assert summary['value/q_v_covariance_corrected'] < 0
    # Tiny-tail control variate also ignores appended extra arms.
    tiny = dict(rec, candidate_pi=[.6, .4], tail_mass=0.)
    tiny_data = dict(data, tail_retained_index=np.zeros((4, 4), np.int32))
    np.testing.assert_allclose(policy_baseline(tiny, tiny_data), -.06)
    # Shared chance/policy noise cancels in advantages through paired MC
    # covariance, rather than being counted as independent arm noise.
    common = (np.array([-1., -1., 1., 1.])[:, None]+
              np.array([-1., 1., -1., 1.])[None, :])*.1
    data['outcome'] += common[None]
    analyze_wide_root(rec, data)
    assert rec['wide']['actions'][0]['advantage_mc_variance'] == pytest.approx(0., abs=1e-14)


def test_wide_crossfit_does_not_evaluate_training_winner_in_sample():
    rec, data = wide_fixture()
    # Apparent wide winner reverses in the independent quadrant.
    data['outcome'][3:, :2, :2] = np.array([1., -1.])[:, None, None]
    data['outcome'][3:, 2:, 2:] = np.array([-1., 1.])[:, None, None]
    analyze_wide_root(rec, data)
    summary = rec['wide']['summary']
    assert summary['discovery/2/train_positive_fraction'] == 1
    assert summary['discovery/2/heldout_gap'] == pytest.approx(-1.1)
    assert summary['discovery/2/heldout_win_fraction'] == 0
    assert summary['discovery/2/evidence_fraction'] == 0


def test_wide_shortfall_and_bootstrap_clusters():
    rec, data = wide_fixture()
    analyze_wide_root(rec, data)
    # Two roots from the same deal move together in deal-pair bootstrap; root
    # bootstrap is allowed to break that dependence and must also be reported.
    second = copy.deepcopy(rec)
    second['game'] = 1
    second['wide']['summary']['discovery/2/heldout_gap'] = -.7
    report = summarize_wide([rec, second], [{'g': 0}, {'g': 1}], 100, 42)['overall']
    assert report['estimate']['discovery/2/heldout_gap'] == pytest.approx(0.)
    assert report['bootstrap_deal_pair_95']['discovery/2/heldout_gap'] == pytest.approx([0., 0.])
    assert report['bootstrap_root_95']['discovery/2/heldout_gap'] == pytest.approx([-.7, .7])
    assert report['estimate']['value/within_root_correlation'] == pytest.approx(-1.)
    empty = copy.deepcopy(rec)
    empty['wide'].update(actions=[], shortfall=2)
    analyze_wide_root(empty, {k: v[:3] for k, v in data.items()})
    assert empty['wide']['summary']['discovery/2/evidence_fraction'] == 0
    assert 'discovery/2/heldout_gap' not in empty['wide']['summary']
    combined = summarize_wide([rec, empty], [{'g': 0}, {'g': 1}], 10, 1)['overall']
    assert combined['estimate']['discovery/2/evidence_fraction'] == .5
    assert combined['metric_roots']['discovery/2/heldout_gap'] == 1


def test_all_json_report_paths_numpy_scalars_and_cuda_metadata(tmp_path, monkeypatch, capsys):
    from time import perf_counter
    rec, data = wide_fixture()
    analyze_wide_root(rec, data)
    rec.update(world_counts=dict(requested=80, completed=80, load_failed=0, truncated=0),
               continuation_decisions=100, coverage_met=np.bool_(True),
               top_certified=np.bool_(False), candidate_mass=np.float64(.9))
    games = [dict(g=0, truncated=np.bool_(False))]
    config = parser().parse_args(['--device', 'cuda', '--wide-candidates', '2', '--bootstrap', '10',
                                  '--out', str(tmp_path/'report.json')])
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'mock CUDA (serialization only)')
    start = perf_counter()
    # Initial, checkpoint and final reports all exercise the same encoder.
    for records, final in [([], False), ([rec], False), ([rec], True)]:
        save_report(config, records, games, None, start, start, final=final)
        loaded = json.loads(config.out.read_text())
        assert loaded['runtime']['device_name'].startswith('mock CUDA')
    assert loaded['roots'][0]['coverage_met'] is True
    assert loaded['wide_summary']['overall']['estimate']['discovery/2/heldout_gap'] == pytest.approx(.7)
    monkeypatch.setattr(sys, 'argv', ['luck_diagnostic.py', '--summarize', str(config.out),
                                     '--out', str(tmp_path/'resummary.json'), '--bootstrap', '10'])
    main()
    json.loads(capsys.readouterr().out)
    result = json.loads((tmp_path/'resummary.json').read_text())
    assert result['wide_summary']['overall']['roots'] == 1
    assert not list(tmp_path.glob('*.tmp'))
    assert json.loads(json.dumps(dict(b=np.bool_(True), i=np.int64(2), a=np.array([3])),
                                allow_nan=False, default=_json_default)) == dict(b=True, i=2, a=[3])
    config.wide_candidates = 0
    legacy = save_report(config, [], games, None, start, start)
    assert legacy['schema_version'] == 1 and 'wide_summary' not in legacy
    assert not any(k.startswith('wide_') for k in legacy['config'])


@pytest.mark.parametrize('option,value', [('--wide-candidates', '-1'), ('--wide-draws', '0'),
    ('--wide-temperature', '0'), ('--wide-temperature', 'inf'),
    ('--wide-max-pi', 'nan'), ('--wide-max-pi', '1.1')])
def test_wide_invalid_cli_before_checkpoint_load(monkeypatch, option, value):
    monkeypatch.setattr(sys, 'argv', ['luck_diagnostic.py', option, value])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize('method,temperature,count', [('prefix-uniform', 2., 1),
    ('tempered', 1e308, 1), ('tempered', 1e-308, 0)])
def test_wide_underflowed_legal_action_and_extreme_temperature(monkeypatch, method, temperature, count):
    import luck_diagnostic as diagnostic
    class TinyEnv:
        def __init__(self):
            self.decisions = np.zeros(1, int)
            self.mask = np.array([[True, True, False]])

        def __getattr__(self, name):
            return np.zeros((1, 1), np.int32)

        def step(self, choices):
            assert self.mask[0, choices[0]]
            self.decisions[:] = 1

    class TinyModel:
        def encode(self, batch):
            return torch.zeros(1, 1, 1)

        def heads(self, encoded, **kwargs):
            return torch.tensor([[0., -1000., -1e9]]), None

    monkeypatch.setattr(diagnostic, 'world', lambda *a: TinyEnv())
    monkeypatch.setattr(diagnostic, 'copy_exact', lambda env, *a: env.decisions.fill(0))
    monkeypatch.setattr(diagnostic, 'tensor_batch', lambda *a: {})
    archive = argparse.Namespace(**{k: np.zeros((1, 1)) for k in diagnostic.OBS_FIELDS})
    rec = dict(game=0, decision_index=0, archive_row=0, candidate_paths=[[0]],
               root_policy_prefixes=[dict(path=[], probabilities=[1., 0., 0.])])
    wide_candidates(args(wide_candidates=1, wide_draws=128, wide_method=method,
                         wide_temperature=temperature, wide_max_pi=.01), TinyModel(), archive, rec)
    assert len(rec['wide']['actions']) == count
    if not count:
        assert rec['wide']['shortfall'] == 1
        return
    action, = rec['wide']['actions']
    assert action['path'] == [1] and action['pi'] == 0.
    assert action['log_pi'] == -1000. and action['proposal_index'] <= 128
