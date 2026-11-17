"""The same stage distribution on CPU evaluation and routed GPU opponents."""
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gfrl import VecEnv
from gfrl.actor import Actor
from gfrl.noise import epsilon_log_probs
from gfrl.gpu.actor import RoutedGpuActor
from gfrl.rollout import Collector
from test_training import tiny, threads


@pytest.mark.parametrize('epsilon', [0., .15, 1.])
def test_stage_mixture_legal_and_uniform(epsilon):
    torch.manual_seed(42)
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=torch.bool)
    lp = torch.tensor([[.7, 0, .2, .1], [0, 1, 0, 0]]).log()
    actual = epsilon_log_probs(lp, mask, epsilon)
    expected = (1-epsilon)*lp.exp()+epsilon*mask/mask.sum(-1, keepdim=True)
    torch.testing.assert_close(actual.exp(), expected)
    if epsilon == 0:
        assert actual is lp
    assert not actual.exp()[~mask].any()
    samples = torch.multinomial(actual.exp().repeat(20000, 1), 1).view(20000, 2)
    observed = torch.bincount(samples[:, 0], minlength=4)/20000
    torch.testing.assert_close(observed, expected[0], atol=.015, rtol=0)


def test_routed_noise_is_opponent_only_and_zero_preserves_rng():
    # Exercise the production RoutedGpuActor.act without CUDA hardware; the
    # model/heads stand-ins return fixed public stage distributions.
    actor = object.__new__(RoutedGpuActor)
    mask = torch.tensor([[True, False, True]]*8)
    lp = torch.tensor([[.9, 0., .1]]*8).log()
    value = torch.zeros(8, 3)
    model = SimpleNamespace(heads=lambda *a, **k: (lp, value))
    actor.model, actor.sample, actor.pool_epsilon = model, True, .15
    actor.env = SimpleNamespace(device='cpu', num_envs=8, mask=mask)
    from gfrl.actor import PREFIX_FIELDS
    for key in PREFIX_FIELDS:
        if key != 'mask':
            setattr(actor.env, key, torch.zeros(8))
    actor.sparse_heads, actor._autocast = None, nullcontext
    opponent = SimpleNamespace(sparse_heads=None, model=model, _autocast=nullcontext)
    rows = torch.arange(8)%2 == 0
    encoded = [(actor, ~rows, None, None), (opponent, rows, None, None)]
    # The real environment has 21 options; no inference kernels are invoked here.
    mask = torch.nn.functional.pad(mask, (0, 18))
    actor.env.mask = mask
    lp = torch.nn.functional.pad(lp, (0, 18), value=-float('inf'))
    choice, actual, _ = actor.act(encoded)
    torch.testing.assert_close(actual[~rows], lp[~rows], atol=0, rtol=0)
    torch.testing.assert_close(actual[rows].exp(), epsilon_log_probs(lp[rows], mask[rows], .15).exp())
    assert mask[torch.arange(8), choice.long()].all()
    actor.pool_epsilon = 0.
    torch.manual_seed(70)
    expected = torch.multinomial(lp.exp(), 1).squeeze(1)
    rng = torch.get_rng_state()
    torch.manual_seed(70)
    choice, actual, _ = actor.act(encoded)
    assert torch.equal(actual, lp)
    assert torch.equal(choice.long(), expected)
    assert torch.equal(torch.get_rng_state(), rng)


def test_cpu_actor_evaluation_noise_and_no_label_access():
    from test_model_actor import PublicOnly
    model = tiny().eval()
    env = VecEnv(16, 2, 37)
    actor = Actor(model, PublicOnly(env), 'cpu', epsilon=1.)
    for _ in range(30):
        result = actor.act()
        mask = torch.tensor(env.mask.copy())
        expected = mask/mask.sum(-1, keepdim=True)
        torch.testing.assert_close(torch.tensor(result.log_probs).exp(), expected)
        assert env.mask[np.arange(16), result.choices].all()
        env.step(result.choices)


def test_default_collector_and_zero_noise_bit_identical():
    torch.manual_seed(3)
    model = tiny()
    outputs = []
    pool = [dict(iteration=1, weights=model.state_dict())]
    for kwargs in ({}, dict(pool_epsilon=0.)):
        torch.manual_seed(10)
        batch = Collector(model, num_envs=8, threads=2, seed=51, pool_fraction=1., **kwargs).collect(8, pool)
        outputs.append((batch, torch.get_rng_state()))
    a, b = outputs
    for part in ('columns', 'stages'):
        for key, value in getattr(a[0], part).items():
            torch.testing.assert_close(value, getattr(b[0], part)[key], rtol=0, atol=0)
    assert torch.equal(a[1], b[1])


@pytest.mark.parametrize('epsilon', [-.1, 1.1, float('nan')])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match='epsilon'):
        Collector(tiny(), pool_epsilon=epsilon)
    with pytest.raises(ValueError, match='epsilon'):
        Actor(tiny(), VecEnv(1, 1), 'cpu', epsilon=epsilon)


def test_eval_cli_uses_noise_for_b_and_never_opens_critic(tmp_path):
    model = tiny()
    checkpoint = tmp_path/'policy.pt'
    # Deliberately unavailable companion proves the CLI takes only public
    # weights, as it does for real privileged training checkpoints.
    torch.save(dict(ema=model.state_dict(), model_config=asdict(model.config),
                    critic_checkpoint=dict(file='unavailable.pt')), checkpoint)
    script = Path(__file__).resolve().parents[2]/'scripts/eval_policies.py'
    output = tmp_path/'evaluation.json'
    command = [sys.executable, str(script), '--a', str(checkpoint), '--b', str(checkpoint),
               '--pool-epsilon', '.15', '--pairs', '2', '--num-envs', '4', '--threads', '1',
               '--out', str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result['pool_epsilon'] == .15 and result['results'][0]['games'] == 4
    bad = subprocess.run(command+['--pool-epsilon', 'nan'], capture_output=True, text=True)
    assert bad.returncode != 0 and 'pool-epsilon must be in [0,1]' in bad.stderr
