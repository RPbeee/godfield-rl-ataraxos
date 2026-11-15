"""Moving regularization policy, legacy arithmetic, and checkpoint lifecycle."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from gfrl.learner import Learner, LearnerConfig, decision_terms, schedules
from gfrl.rollout import Collector
from test_training import tiny, train_module, threads


def assert_weights(a, b):
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_default_uniform_arithmetic_and_rng():
    from legacy.learner import decision_terms as original_terms, schedules as original_schedules
    cfg = LearnerConfig(magnet=True)
    assert cfg.magnet_source == 'uniform' and cfg.alpha_schedule == 'decay'
    for i in (1, 2, 2500, 9999):
        assert schedules(cfg, i) == original_schedules(cfg, i)
    torch.manual_seed(7)
    mask = torch.rand(9, 21) > .4
    mask[:, 0] = True
    logits = torch.randn(9, 21, requires_grad=True)
    lp = logits.masked_fill(~mask, -float('inf')).log_softmax(-1)
    old = torch.randn(9, 21).masked_fill(~mask, -float('inf')).log_softmax(-1)
    args = (old, mask, torch.zeros(9, dtype=torch.long), torch.arange(9)//3, 3)
    actual, expected = decision_terms(lp, *args), original_terms(lp, *args)
    assert_weights(actual, expected)
    ga = torch.autograd.grad(sum(x.sum() for x in actual.values()), logits, retain_graph=True)[0]
    ge = torch.autograd.grad(sum(x.sum() for x in expected.values()), logits)[0]
    torch.testing.assert_close(ga, ge, atol=0, rtol=0)
    model = tiny()
    rng = torch.get_rng_state()
    learner = Learner(model)
    assert learner.reg is None
    assert torch.equal(rng, torch.get_rng_state())
    assert schedules(LearnerConfig(magnet=True, alpha_schedule='constant'), 2500)[1] == .05
    assert schedules(LearnerConfig(alpha_schedule='constant'), 2500)[1] == 0


def test_masked_reg_kl_value_gradient_and_stage_prefix_sum():
    mask = torch.tensor([[True, False, True], [False, True, True], [True, True, True]])
    logits = torch.tensor([[1., 8., 3.], [5., 2., 4.], [3., 1., 2.]], requires_grad=True)
    reg = torch.tensor([[2., 9., 1.], [4., 3., 2.], [1., 2., 3.]], requires_grad=True)
    lp = logits.masked_fill(~mask, -float('inf')).log_softmax(-1)
    rp = reg.masked_fill(~mask, -float('inf')).log_softmax(-1)
    terms = decision_terms(lp, lp.detach(), mask, torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1]), 2, rp)
    stage = torch.stack([(lp[i, mask[i]].exp()*(lp[i, mask[i]]-rp[i, mask[i]].detach())).sum() for i in range(3)])
    expected = torch.stack((stage[:2].sum(), stage[2]))
    torch.testing.assert_close(terms['magnet'], expected)
    ga = torch.autograd.grad(terms['magnet'].sum(), logits, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(ga, ge)
    assert reg.grad is None


@pytest.mark.parametrize('source', ['online', 'ema'])
@pytest.mark.parametrize('storage', ['device', 'cpu'])
def test_reg_zero_freeze_and_update_cadence(source, storage):
    torch.manual_seed(31)
    learner = Learner(tiny().eval(), LearnerConfig(magnet=True, magnet_source='reg',
                      reg_source=source, reg_update_every=2, alpha_schedule='constant',
                      minibatch_size=64, microbatch_size=4))
    batch = Collector(learner.model, num_envs=8, seed=1, pool_fraction=0,
                      rollout_storage=storage).collect(1)[:8]
    initial = deepcopy(learner.reg.state_dict())
    torch.testing.assert_close(learner.forward(batch)[0]['magnet'], torch.zeros(len(batch)), atol=1e-6, rtol=0)
    for i in range(1, 5):
        metrics = learner.update(batch, i)
        assert metrics['reg_updated'] == (i % 2 == 0)
        assert metrics['alpha'] == .05
        assert metrics['reg_last_update'] == i//2*2
        assert np.isfinite(metrics['reg_kl'])
        if i % 2:
            assert_weights(initial, learner.reg.state_dict())
        else:
            selected = learner.model if source == 'online' else learner.ema
            assert_weights(selected.state_dict(), learner.reg.state_dict())
            initial = deepcopy(learner.reg.state_dict())
            if source == 'online':
                torch.testing.assert_close(learner.forward(batch)[0]['magnet'], torch.zeros(len(batch)), atol=1e-6, rtol=0)
    assert not learner.reg.training
    assert all(not p.requires_grad and p.grad is None for p in learner.reg.parameters())


def test_reg_checkpoint_exact_update_continuation(tmp_path):
    train = train_module()
    torch.manual_seed(12)
    cfg = LearnerConfig(magnet=True, magnet_source='reg', reg_update_every=2,
                        minibatch_size=8, microbatch_size=4)
    learner = Learner(tiny(), cfg)
    batch = Collector(learner.model, num_envs=8, seed=2, pool_fraction=0).collect(1)[:8]
    learner.update(batch, 1)
    path = tmp_path/'checkpoint.pt'
    train.save_checkpoint(path, learner, 1, [], {}, vars(cfg), 0.)
    state = torch.load(path, weights_only=False)
    other = Learner(tiny(), deepcopy(cfg))
    for name in ('model', 'ema'):
        getattr(other, name).load_state_dict(state[name])
    other.optimizer.load_state_dict(state['optimizer'])
    other.optimizer_steps = state['optimizer_steps']
    other.restore_reg(state, resume=True)
    assert other.reg_last_update == learner.reg_last_update == 0
    assert_weights(other.reg.state_dict(), learner.reg.state_dict())
    np.random.seed(99)
    a = learner.update(batch, 2)
    np.random.seed(99)
    b = other.update(batch, 2)
    assert {k:v for k,v in a.items() if not k.endswith('_s')} == {k:v for k,v in b.items() if not k.endswith('_s')}
    for name in ('model', 'ema', 'reg'):
        assert_weights(getattr(learner, name).state_dict(), getattr(other, name).state_dict())


@pytest.mark.parametrize('has_reg', [False, True])
def test_reg_init_from_and_resume_auto(tmp_path, has_reg):
    train = train_module()
    base = ['--d','32','--depth','1','--num-envs','8','--threads','2',
            '--games-per-iter','1','--iterations','1','--eval-every','0', '--pool-fraction','0']
    origin = tmp_path/'origin'
    path = train.run(train.parse_args(base+['--run-dir',str(origin)] +
                (['--magnet-source','reg'] if has_reg else [])))
    state = torch.load(path, weights_only=False)
    # A saved magnet must be discarded by init-from, even when present.
    if has_reg:
        state['reg'] = {k: v+1 if v.is_floating_point() else v for k,v in state['model'].items()}
        state['reg_last_update'] = -100
        torch.save(state, path)
    new_dir = tmp_path/'new'
    flags = ['--magnet','--magnet-source','reg','--alpha-schedule','constant',
             '--reg-update-every','2','--reg-source','ema']
    argv = base+flags+['--run-dir',str(new_dir)]
    new_path = train.run(train.parse_args(argv+['--init-from',str(path),'--max-hours','1e-12']))
    initialized = torch.load(new_path, weights_only=False)
    assert initialized['iteration'] == initialized['reg_last_update'] == state['iteration']
    assert_weights(initialized['reg'], state['model'])
    # Conflicting flags on resume must not replace the saved experiment.
    for expected in (2, 3):
        train.run(train.parse_args(base+['--run-dir',str(new_dir),'--resume','auto',
                                        '--reg-update-every','99','--magnet-source','uniform']))
        saved = torch.load(new_path, weights_only=False)
        assert saved['config']['magnet_source'] == 'reg'
        assert saved['config']['reg_update_every'] == 2
        assert saved['iteration'] == expected
        assert saved['reg_last_update'] == (1 if expected == 2 else 3)
        if expected == 2:
            assert_weights(saved['reg'], initialized['reg'])
        else:
            assert_weights(saved['reg'], saved['ema'])
    metrics = [json.loads(x) for x in (new_dir/'metrics.jsonl').read_text().splitlines()]
    assert [m['reg_updated'] for m in metrics] == [0, 1]


@pytest.mark.parametrize('flags', [['--reg-update-every','0'], ['--magnet-source','bad'],
                                  ['--alpha-schedule','bad'], ['--reg-source','bad']])
def test_magnet_cli_validation(flags):
    with pytest.raises(SystemExit):
        train_module().parse_args(flags)
