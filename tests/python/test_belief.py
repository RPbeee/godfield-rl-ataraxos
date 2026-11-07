"""CPU belief data isolation, reproducibility and likelihood reference tests."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from gfrl import VecEnv
from gfrl.actor import Actor
from gfrl.belief_data import collect_positions, make_belief_data
from gfrl.belief_metrics import (Baseline, calibration_metrics, evaluate_baseline,
                               fit_baselines, gift_probabilities, likelihood_metrics,
                               multiset_log_prob)
from gfrl.model import Model, ModelConfig, OBS_FIELDS


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(71)
    config = ModelConfig(d=16, depth=1, heads=4, ff=32)
    weights = Model(config).state_dict()
    path = tmp_path / 'checkpoint.pt'
    torch.save(dict(ema=weights, model_config=asdict(config),
                    pool=[dict(iteration=3, weights=weights)]), path)
    return path


@pytest.mark.parametrize('opponent', ['self', 'greedy'])
def test_recording(checkpoint, opponent):
    seen = []
    def check(env, rows, saved):
        assert np.all(env.substep[rows] == 0)
        for key in OBS_FIELDS:
            target = getattr(env, key)[rows]
            if key == 'obs_num':
                target = target.astype(np.float16)
            np.testing.assert_array_equal(saved[key], target)
        np.testing.assert_array_equal(saved['label_opp_hand'], env.label_opp_hand[rows])
        np.testing.assert_array_equal(saved['label_opp_hand'].sum(1), saved['label_opp_hand_count'])
        seen.extend(zip(saved['game_id'], saved['decision'], saved['seat']))
    kwargs = dict(positions=120, seed=53, num_envs=4, threads=2,
                  sample_prob=.5, opponent=opponent)
    a = make_belief_data(checkpoint, on_record=check, **kwargs)
    # Loading models initializes weights before loading; collection itself isolates RNG.
    b = make_belief_data(checkpoint, **kwargs)
    c = make_belief_data(checkpoint, labels_only=True, **kwargs)
    for key in a:
        if key != 'config':
            assert torch.equal(a[key], b[key])
            if key not in OBS_FIELDS:
                assert torch.equal(a[key], c[key])
    assert not any(k in c for k in OBS_FIELDS)
    assert len(seen) == len(set(seen)) == 120
    if opponent == 'greedy':
        assert (a['seat'] == 0).all() and (a['opponent_type'] == 2).all()
    else:
        assert set(a['seat'].tolist()) == {0, 1}
        assert (a['opponent_type'] == 1).any()
        assert (a['opponent_type'][a['seat'] == 1] == 0).all()
        for game in a['game_id'].unique():
            rows = (a['game_id'] == game) & (a['seat'] == 0)
            assert a['opponent_type'][rows].unique().numel() <= 1
            assert a['pool_index'][rows].unique().numel() <= 1
    assert a['obs_num'].dtype == torch.float16


def test_actor_receives_exact_observation_and_ignores_labels():
    env = VecEnv(3, 1, 31)
    actor = Actor(Model(ModelConfig(d=16, depth=1, heads=4, ff=32)), env, 'cpu', sample=False)
    encode = actor.model.encode
    def spy(obs):
        assert set(obs) == set(OBS_FIELDS)
        for key in OBS_FIELDS:
            np.testing.assert_array_equal(obs[key].numpy(), getattr(env, key))
        return encode(obs)
    actor.model.encode = spy
    first = actor.act()
    # Label-only mutations cannot affect either the encoder or action heads.
    # NumPy buffers are read-only to users; a public-only proxy makes access fail.
    class Public:
        num_envs = env.num_envs
        def __getattr__(self, key):
            assert not key.startswith('label_')
            return getattr(env, key)
    actor._bind(Public())
    second = actor.act()
    np.testing.assert_array_equal(first.choices, second.choices)
    np.testing.assert_array_equal(first.log_probs, second.log_probs)


def test_multinomial_reference_and_normalization():
    pn, pk = torch.tensor([.2, .3, .5]), torch.tensor([.25, .75])
    value = multiset_log_prob([[1, 1]], pn, pk)
    assert value.item() == pytest.approx(math.log(.5 * 2 * .25 * .75))
    mass = 0
    for n in range(3):
        for a in range(n+1):
            mass += multiset_log_prob([[a, n-a]], pn, pk).exp().item()
    assert mass == pytest.approx(1)
    assert multiset_log_prob([[0, 0]], pn, pk).item() == pytest.approx(math.log(.2))


def test_calibration():
    y = torch.zeros(100, 297)
    y[:20] = 1
    perfect = calibration_metrics(torch.full_like(y, .2), y)
    shifted = calibration_metrics(torch.full_like(y, .8), y)
    assert perfect['ece'] < 1e-7
    assert shifted['ece'] > .59
    assert sum(b['count'] for b in perfect['reliability']) == y.numel()
    edge = calibration_metrics(y, y)
    assert edge['ece'] == 0
    assert edge['reliability'][-1]['mean_prediction'] == 1


def test_fit_and_evaluate():
    labels = torch.zeros(10, 297, dtype=torch.uint8)
    labels[:, 1] = 2
    data = dict(label_opp_hand=labels, label_opp_hand_count=labels.sum(1).byte(), decision=torch.arange(10))
    baselines = fit_baselines(data)
    assert baselines['B1'].count_probs[2] == pytest.approx(11/29)
    assert baselines['B1'].card_probs[1] == pytest.approx(21/316)
    b = baselines['B1']
    result = evaluate_baseline(b, data, batch_size=3)
    direct = calibration_metrics(b.presence_probs().expand(10, -1), labels)
    assert result['ece'] == pytest.approx(direct['ece'])
    assert result['count_accuracy'] == 1
    assert result['phase_boundaries'] == [3, 6]
    pk = b.card_probs.clone(); pk[1] = 0; pk /= pk.sum()
    zero = evaluate_baseline(Baseline(b.count_probs, pk), data)
    assert zero['zero_probability_positions'] == 10
    assert zero['zero_probability_card_copies'] == 20
    assert math.isfinite(zero['multiset_nll'])
    assert gift_probabilities()[0] == 0
    arbitrary = likelihood_metrics([-1, -2, -3], [1, 2, 3])
    assert arbitrary['multiset_nll'] == 2


def test_cli(checkpoint, tmp_path):
    root = Path(__file__).resolve().parents[2]
    for name, extra in [('self', []), ('greedy', ['--opponent', 'greedy']), ('fit', ['--labels-only'])]:
        subprocess.run([sys.executable, str(root/'scripts/make_belief_data.py'),
                        '--checkpoint', str(checkpoint), '--positions', '20', '--sample-prob', '1',
                        '--num-envs', '4', '--threads', '1', '--out', str(tmp_path/f'{name}.pt'), *extra], check=True, capture_output=True)
    result = subprocess.run([sys.executable, str(root/'scripts/belief_baselines.py'), '--fit', str(tmp_path/'fit.pt'),
                             '--validation', str(tmp_path/'self.pt'), str(tmp_path/'greedy.pt'),
                             '--out', str(tmp_path/'result.json')], check=True, capture_output=True, text=True)
    assert '| B0 |' in result.stdout and '| B1 |' in result.stdout
    assert len(json.loads((tmp_path/'result.json').read_text())['results']) == 4
