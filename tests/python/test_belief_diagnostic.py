"""Offline unused-hand diagnostic contracts, metric references and CPU CLI."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from gfrl import VecEnv
from gfrl.belief import Belief, BeliefConfig
from gfrl.belief_data import collect_positions
from gfrl.belief_diagnostic import public_used, true_unused, sample_metrics, summarize
from gfrl.model import Model, ModelConfig


def test_unused_extraction_live_env():
    env = VecEnv(4, 1, 17)
    checked_used = 0
    for _ in range(500):
        for row in np.flatnonzero(env.substep == 0):
            used = public_used(env, row)
            truth = true_unused(env, row)
            np.testing.assert_array_equal(truth+used, env.label_opp_hand[row])
            np.testing.assert_array_equal(truth, env.unused_assignment(int(row), env.label_opp_hand[row].astype(np.int32)))
            checked_used += bool(used.sum())
        env.step(env.mask.argmax(1).astype(np.int32))
    assert checked_used > 0


def test_metric_references_and_padding():
    # code 1 present in half the samples; code 2 always, with mean 1.5 copies.
    samples = np.array([[0, 1, 1, 0], [0, 0, 2, 0]])
    truth = np.array([0, 1, 1, 0])
    v = sample_metrics(samples, truth)
    assert v[0] == 1 and v[1] == 0
    assert v[2] == pytest.approx(.25)
    assert v[3] == pytest.approx(-np.log(.5)-np.log(5/6)-np.log(5/6))
    assert v[4] == pytest.approx(.25)
    assert v[5] == pytest.approx(-np.log(.5)-np.log(5/6))
    assert v[6] == pytest.approx(.5)
    summary = summarize([v], [10], bootstrap=20)
    assert summary['metrics']['presence_ece']['mean'] == pytest.approx(.5/3)
    assert summary['metrics']['copies_mse']['ci95'] == [.5, .5]
    assert sample_metrics(samples, truth, [3])[2] == 0
    with pytest.raises(ValueError):
        sample_metrics(samples[:0], truth)
    assert summarize([], [], 10)['metrics'] is None


def test_mode_tie_and_game_bootstrap():
    small = sample_metrics([[0, 0], [0, 1]], [0, 0])
    large = sample_metrics([[0, 1], [0, 1]], [0, 0])
    assert small[0] == 1 and small[1] == .5
    # One cluster: every bootstrap draw must keep all three positions together.
    s = summarize([small, large, large], [5, 5, 5], bootstrap=40)
    metric = s['metrics']['size_accuracy']
    assert metric['mean'] == pytest.approx(1/3)
    assert metric['ci95'] == pytest.approx([1/3, 1/3])


def test_finish_games_matches_collection():
    class FirstLegal:
        def __call__(self, env, rows):
            return env.mask.argmax(1).astype(np.int32)
    kwargs = dict(positions=40, num_envs=4, pool_fraction=0, sample_prob=.1, seed=8)
    normal = collect_positions([FirstLegal()], **kwargs)
    finished = collect_positions([FirstLegal()], finish_games=True, **kwargs)
    for key in normal:
        if key != 'config':
            assert torch.equal(normal[key], finished[key])
    assert (finished['decisions_to_end'] > 0).all()
    terminal = finished['decision']+finished['decisions_to_end']
    for game in finished['game_id'].unique():
        assert terminal[finished['game_id'] == game].unique().numel() == 1
    assert terminal.max() <= 512


def test_cpu_cli(tmp_path):
    config = ModelConfig(d=16, depth=1, heads=4, ff=32)
    torch.save(dict(ema=Model(config).state_dict(), model_config=asdict(config)), tmp_path/'policy.pt')
    belief_config = BeliefConfig(heads=4)
    model = Belief(config, belief_config)
    # Use the same checkpoint schema as training.
    from gfrl.belief import belief_from_checkpoint
    checkpoint = dict(model=model.state_dict(), model_config=asdict(config), belief_config=asdict(belief_config))
    belief_from_checkpoint(checkpoint)
    torch.save(checkpoint, tmp_path/'ar.pt')
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root/'scripts/belief_diagnostic.py'),
                    '--policy', str(tmp_path/'policy.pt'), '--belief', str(tmp_path/'ar.pt'),
                    '--positions', '40', '--samples', '8', '--fit-positions', '60',
                    '--num-envs', '4', '--bootstrap', '20', '--out', str(tmp_path/'report.json')],
                   check=True, capture_output=True, text=True)
    report = json.loads((tmp_path/'report.json').read_text())
    assert set(report['results']) == {'ar', 'hybrid', 'b1', 'uninformed'}
    for result in report['results'].values():
        assert result['accepted']+result['exhausted_slots'] == 320
        assert result['overall']['positions'] <= 40
        assert len(result['decisions_to_end']) == 6
    assert report['results']['uninformed']['overall']['metrics']['size_mae']['mean'] == 0
    data = torch.load(tmp_path/'report.positions.pt', weights_only=False)
    assert len(data['obs_cards']) == len(data['true_unused']) == 40
    assert torch.equal(data['true_unused'].sum(1), data['true_unused_size'].long())
    assert '| all | ar |' in (tmp_path/'report.md').read_text()
