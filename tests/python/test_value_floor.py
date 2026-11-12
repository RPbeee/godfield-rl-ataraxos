"""Terminal-world measurement math and real-policy RNG isolation."""
import argparse
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from value_floor import (Continuations, METRICS, brier_counts, bucket, real_pass,
                         root_metrics, summarize)
from gfrl.model import Model, ModelConfig
from gfrl.search import load_search_belief


def test_brier_and_draw_correction():
    p, raw, corrected = brier_counts(2, 2, 0, 1)
    assert (p, raw, corrected) == pytest.approx((.5, .25, .25-.25/3))
    # All draws have zero Monte Carlo variance, unlike a Bernoulli(.5).
    assert brier_counts(0, 0, 4, 0) == pytest.approx((.5, .25, .25))
    # Exhaust all n=3 samples to verify unbiasedness with nonzero draw mass.
    probs = {0.: .2, .5: .3, 1.: .5}
    expectation = 0.
    for sample in itertools.product(probs, repeat=3):
        weight = np.prod([probs[x] for x in sample])
        expectation += weight*brier_counts(sample.count(1.), sample.count(0.), sample.count(.5), .5)[2]
    assert expectation == pytest.approx((.65-.5)**2)
    with pytest.raises(ValueError):
        brier_counts(1, 0, 0, 1)
    with pytest.raises(ValueError):
        brier_counts(-1, 4, 0, 1)


@pytest.mark.parametrize('d,expected', [(0, '[0,5)'), (4, '[0,5)'), (5, '[5,10)'),
    (9, '[5,10)'), (10, '[10,20)'), (20, '[20,30)'), (30, '[30,45)'),
    (45, '[45,60)'), (59, '[45,60)'), (60, '[60,inf)'), (10000, '[60,inf)')])
def test_bucket(d, expected):
    assert bucket(d) == expected


def test_summary_and_cluster_bootstrap():
    c = dict(wins=2, losses=1, draws=1, completed=4)
    row = dict(game=0, y=1., v=0., real_truncated=False, decisions_to_end=5, state=c, info=c)
    m = root_metrics(row)
    assert m[0] == .25
    assert m[-1] == pytest.approx(m[0]-m[2])
    result = summarize([row, dict(row), dict(row, real_truncated=True)], bootstrap=30)
    assert result['overall']['n'] == 2
    assert result['excluded_roots'] == 1
    assert result['buckets']['[5,10)']['n'] == 2
    for key, value in zip(METRICS, m):
        assert result['overall_bootstrap_game_95'][key] == pytest.approx([value, value])
    assert summarize([])['overall']['brier_value'] is None
    assert root_metrics(dict(row, state=dict(c, completed=1))) is None


def test_terminal_continuations_preserve_real_trajectory():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(33)
        model = Model(ModelConfig(d=16, heads=2, depth=1, ff=32)).eval()
        args = argparse.Namespace(games=3, num_envs=3, threads=1, seed=91,
                                  device='cpu', k=2, inference_batch=8, root_batch=8,
                                  cache_mb=256, roots_per_game=2, max_decisions=10000)
        targets = [set(range(10)), {0}, {0}]
        selected, games, digest, _ = real_pass(args, model, targets=targets)
        mc = Continuations(args, model, load_search_belief('uniform'))
        replay_selected, replay_games, replay_digest, _ = real_pass(args, model, mc, targets=targets)
        # Identical freeze schedule, with/without continuation RNG consumption.
        assert digest == replay_digest and games == replay_games
        assert selected == replay_selected
        assert len(mc.rows) == sum(map(len, selected)) >= 3
        assert len(selected[0]) > 1
        assert max(b['roots'] for b in mc.batches) == 3
        assert all(b['worlds'] == b['roots']*2*args.k for b in mc.batches)
        assert mc.decisions > 0
        for row in mc.rows:
            assert -1 <= row['v'] <= 1
            for variant in ('state', 'info'):
                c = row[variant]
                assert c['requested'] == 2
                assert c['completed'] + c['truncations'] + c['invalid_loads'] + c['exhausted_samples'] == 2
                assert c['truncations'] == 0
                assert c['completed'] == 2
        # Greedy real trajectories also match when freezing changes batch timing.
        _, frozen_games, _, _ = real_pass(args, model, targets=targets, sample=False)
        args.root_batch = 1
        _, unfrozen_games, _, _ = real_pass(args, model, targets=targets, sample=False)
        assert frozen_games == unfrozen_games
    finally:
        torch.set_num_threads(old_threads)


def test_target_sampling():
    from value_floor import draw_targets
    args = argparse.Namespace(seed=12, games=5, roots_per_game=4)
    targets = draw_targets(args)
    assert targets == draw_targets(args)
    assert len(targets) == 5
    assert all(len(ds) == 4 and min(ds) >= 0 and max(ds) < 100 for ds in targets)


def test_no_roots_and_limited_concurrency():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(3)
        model = Model(ModelConfig(d=16, heads=2, depth=1, ff=32)).eval()
        args = argparse.Namespace(games=3, num_envs=1, threads=1, seed=91,
                                  device='cpu', root_batch=64)
        selected, games, _, _ = real_pass(args, model, targets=[{10000}]*3)
        assert selected == [[], [], []]
        assert len(games) == 3
    finally:
        torch.set_num_threads(old_threads)
