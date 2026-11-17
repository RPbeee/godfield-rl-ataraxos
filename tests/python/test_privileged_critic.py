"""Training-only value inputs, gradient isolation, targets and restart lifecycle."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from gfrl.critic import PRIVILEGED_FIELDS, prediction_metrics
from gfrl.learner import Learner, LearnerConfig
from gfrl.model import OBS_FIELDS
from gfrl.rollout import Collector
from gfrl.targets import compute_targets
from gfrl.training_checkpoint import load_training_checkpoint
from test_training import tiny, train_module, threads
from test_magnet import assert_weights


def collected(model, storage='device'):
    return Collector(model, num_envs=8, threads=2, seed=72, pool_fraction=0,
                     privileged_labels=True, rollout_storage=storage).collect(2)[:16]


@pytest.mark.parametrize('storage', ['device', 'cpu'])
def test_inputs_targets_and_disjoint_gradients(storage, monkeypatch):
    torch.manual_seed(44)
    learner = Learner(tiny(), LearnerConfig(critic='privileged', minibatch_size=8, microbatch_size=3))
    batch = collected(learner.model, storage)
    assert not (set(map(id, learner.model.parameters())) & set(map(id, learner.critic.parameters())))
    assert not any('critic' in k for k in learner.model.state_dict())
    old = batch.old_value.clone()
    prepared = learner._prepare(batch, torch.arange(len(batch)))
    assert set(PRIVILEGED_FIELDS).isdisjoint(prepared)
    original = learner.model.encode

    def public(obs, **kwargs):
        assert set(obs) == set(OBS_FIELDS)
        return original(obs, **kwargs)

    monkeypatch.setattr(learner.model, 'encode', public)
    _, values = learner.forward(prepared)
    values.square().mean().backward()
    assert learner.model.value_head.weight.grad is not None
    assert all(p.grad is None for n, p in learner.model.named_parameters() if not n.startswith('value_head'))
    assert all(p.grad is None for p in learner.critic.parameters())
    learner.optimizer.zero_grad(set_to_none=True)
    # Verify both privileged fields affect predictions once the correction learns.
    torch.nn.init.normal_(learner.critic.condition[-1].weight, std=.1)
    baseline = learner._critic_baseline(batch)
    inputs = learner._critic_inputs(batch, torch.arange(len(batch)))
    public_obs, hand, count = inputs
    a = learner.critic(public_obs, hand, count)
    b = learner.critic(public_obs, hand*0, count)
    c = learner.critic(public_obs, hand, count*0)
    assert not torch.equal(a, b) and not torch.equal(a, c)
    assert torch.equal(old, batch.old_value)
    expected_target, expected_advantage = compute_targets(baseline)
    captured = []
    import gfrl.learner as module
    original_targets = module.compute_targets

    def targets(data, *args):
        result = original_targets(data, *args)
        captured.append(result)
        return result

    monkeypatch.setattr(module, 'compute_targets', targets)
    before = deepcopy(learner.critic.state_dict())
    metrics = learner.update(batch, 1)
    for a, b in zip(captured[0], (expected_target, expected_advantage)):
        np.testing.assert_array_equal(a, b)
    assert any(not torch.equal(v, before[k]) for k, v in learner.critic.state_dict().items())
    assert torch.equal(batch.old_value, old)
    for key in ('critic_loss', 'critic_explained_variance', 'shared_value_explained_variance',
                'critic_brier', 'shared_value_brier', 'advantage_std'):
        assert np.isfinite(metrics[key])
    np.testing.assert_allclose(metrics['advantage_std'], expected_advantage.std(), rtol=1e-6)


def test_missing_labels_rejected_and_size_config():
    learner = Learner(tiny(), LearnerConfig(critic='privileged', critic_d=64, critic_depth=2))
    assert learner.critic.config.d == 64 and learner.critic.config.depth == 2
    batch = Collector(learner.model, num_envs=4, pool_fraction=0).collect(1)
    with pytest.raises(ValueError, match='unused hand labels'):
        learner.update(batch, 1)


def test_critic_resume_exact_same_batch_and_rng(tmp_path):
    train = train_module()
    torch.manual_seed(45)
    cfg = LearnerConfig(critic='privileged', minibatch_size=8, microbatch_size=4)
    learner = Learner(tiny(), cfg)
    batch = collected(learner.model)
    learner.update(batch, 1)
    path = tmp_path/'checkpoint.pt'
    train.save_checkpoint(path, learner, 1, [learner.pool_entry(1)], {}, vars(cfg), 0.)
    saved = load_training_checkpoint(path)
    other = Learner(tiny(), deepcopy(cfg))
    for name in ('model', 'ema'):
        getattr(other, name).load_state_dict(saved[name])
    other.optimizer.load_state_dict(saved['optimizer'])
    other.optimizer_steps = saved['optimizer_steps']
    other.restore_critic(saved, resume=True)
    assert other.critic_optimizer_steps == learner.critic_optimizer_steps
    results = []
    for instance in (learner, other):
        torch.set_rng_state(saved['torch_rng'])
        np.random.set_state(saved['numpy_rng'])
        results.append(instance.update(batch, 2))
    assert {k: v for k, v in results[0].items() if not k.endswith('_s')} == {
        k: v for k, v in results[1].items() if not k.endswith('_s')}
    for name in ('model', 'ema', 'critic'):
        assert_weights(getattr(learner, name).state_dict(), getattr(other, name).state_dict())
    for name in ('optimizer', 'critic_optimizer'):
        a, b = getattr(learner, name).state_dict(), getattr(other, name).state_dict()
        assert a['param_groups'] == b['param_groups']
        for key in a['state']:
            assert_weights(a['state'][key], b['state'][key])


def test_cli_init_legacy_resume_auto_and_inference_isolation(tmp_path, monkeypatch):
    train = train_module()
    common = ['--d','32','--depth','1','--num-envs','8','--threads','2',
              '--games-per-iter','1','--iterations','1','--eval-every','0','--pool-every','1']
    source = train.run(train.parse_args(common+['--run-dir', str(tmp_path/'old')]))
    old = torch.load(source, weights_only=False)
    assert 'critic' not in old
    path = train.run(train.parse_args(common+['--run-dir', str(tmp_path/'new'),
        '--init-from', str(source), '--critic','privileged','--pool-epsilon','.15', '--max-hours','1e-12']))
    initialized = load_training_checkpoint(path)
    for key, value in old['model'].items():
        if key in initialized['critic']:
            torch.testing.assert_close(value, initialized['critic'][key], atol=0, rtol=0)
    train.run(train.parse_args(common+['--run-dir',str(tmp_path/'new'),'--resume','auto']))
    saved = load_training_checkpoint(path)
    assert saved['config']['critic'] == 'privileged' and saved['config']['pool_epsilon'] == .15
    assert saved['critic_optimizer_steps'] > 0
    # init-from with an existing critic retains its weights and optimizer.
    copy_path = train.run(train.parse_args(common+['--run-dir', str(tmp_path/'copy'),
        '--init-from', str(path), '--critic','privileged', '--max-hours','1e-12']))
    copied = load_training_checkpoint(copy_path)
    assert_weights(saved['critic'], copied['critic'])
    assert saved['critic_optimizer_steps'] == copied['critic_optimizer_steps']
    # Inference entry points load only the public Model, including pool weights.
    from gfrl import critic
    from gfrl.evaluation import load_policy_model
    from gfrl.actor import Actor
    from test_model_actor import PublicOnly
    from gfrl import VecEnv
    monkeypatch.setattr(critic.PrivilegedCritic, '__init__', lambda *a, **k: pytest.fail('critic loaded by inference'))
    payload = torch.load(path, weights_only=False)
    assert not any(k.startswith('critic') and k != 'critic_checkpoint' for k in payload)
    # Make the training companion unavailable: every inference entry point
    # must still work, without even reading privileged weights/Adam tensors.
    (path.parent/payload['critic_checkpoint']['file']).unlink()
    for spec in (str(path), f"{path}@pool:{saved['pool'][-1]['iteration']}"):
        model = load_policy_model(spec)
        assert set(model.state_dict()) == set(old['model'])
        encode = model.encode
        def public_only(obs, **kwargs):
            assert set(obs) == set(OBS_FIELDS)
            return encode(obs, **kwargs)
        monkeypatch.setattr(model, 'encode', public_only)
        actor = Actor(model, PublicOnly(VecEnv(4, 1, 32)), 'cpu')
        actor.act()
        assert not hasattr(model, 'critic')
        # The official bot and advisor use this same loader. Exercise offline
        # advice and real search/recording with the companion still absent.
        from gfrl.official_advisor import advise
        from test_official_search import result_for, games
        advice = advise(result_for(games()), actor)
        assert advice['status'] in ('advice', 'unsupported')
        from gfrl.search import SearchAgent, SearchConfig, load_search_belief
        agent = SearchAgent(model, load_search_belief('uniform'),
            config=SearchConfig(worlds=1, candidates=2, candidate_samples=3, depth=1), record=True)
        env = VecEnv(2, 1, 81, track_projection=True)
        choices = agent(env, np.arange(2))
        assert env.mask[np.arange(2), choices].all()


def test_critic_companion_missing_mismatched_and_atomic_snapshots(tmp_path):
    train = train_module()
    learner = Learner(tiny(), LearnerConfig(critic='privileged'))
    a, b = tmp_path/'a.pt', tmp_path/'b.pt'
    for path in (a, b):
        train.save_checkpoint(path, learner, 1, [], {}, {}, 0.)
    assert load_training_checkpoint(a)['critic_optimizer_steps'] == 0
    assert load_training_checkpoint(b)['critic_optimizer_steps'] == 0
    old_ref = torch.load(b, weights_only=False)['critic_checkpoint']['file']
    train.save_checkpoint(b, learner, 2, [], {}, {}, 0.)
    assert not (tmp_path/old_ref).exists()
    assert load_training_checkpoint(a)['iteration'] == 1
    reference = torch.load(b, weights_only=False)['critic_checkpoint']['file']
    companion = torch.load(tmp_path/reference, weights_only=False)
    companion['iteration'] = 0
    torch.save(companion, tmp_path/reference)
    with pytest.raises(ValueError, match='iteration differs'):
        load_training_checkpoint(b)
    (tmp_path/reference).unlink()
    with pytest.raises(FileNotFoundError):
        load_training_checkpoint(b)
    assert 'ema' in load_training_checkpoint(b, load_critic=False)


def test_metrics_use_identical_terminal_targets():
    outcomes = torch.tensor([1, -1, 0])
    perfect = torch.eye(3)
    uniform = torch.full((3, 3), 1/3)
    result = prediction_metrics(perfect, uniform, outcomes, torch.tensor([1., 2., 3.]))
    assert result['critic_explained_variance'] == 1
    assert result['shared_value_explained_variance'] == 0
    assert result['critic_brier'] == 0
    assert result['shared_value_brier'] > .66


@pytest.mark.parametrize('magnet_source,filtering,gamma', [('uniform', False, 1.), ('reg', True, .98)])
def test_shared_default_bit_identical_to_legacy_update(magnet_source, filtering, gamma):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('gfrl._shared_reference',
        Path(__file__).parent/'legacy/shared_learner.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.manual_seed(73)
    model = tiny()
    # Frozen pre-privileged source: compare actual columnar training arithmetic,
    # optimizer and EMA bits, rather than comparing two new configurations.
    batch = Collector(model, num_envs=8, threads=2, seed=7, pool_fraction=0).collect(8)
    cfg = LearnerConfig(minibatch_size=8, microbatch_size=4, magnet=True,
                        magnet_source=magnet_source, filtering=filtering, gamma=gamma)
    old, new = original.Learner(deepcopy(model), deepcopy(cfg)), Learner(deepcopy(model), cfg)
    rng = torch.get_rng_state()
    assert new.critic is None and torch.equal(rng, torch.get_rng_state())
    for iteration in (1, 2):
        np.random.seed(17)
        expected = old.update(batch, iteration)
        np.random.seed(17)
        actual = new.update(batch, iteration)
        for key, value in expected.items():
            if not key.endswith('_s'):
                assert value == actual[key]
        for name in ('model', 'ema', *(('reg',) if old.reg is not None else ())):
            assert_weights(getattr(old, name).state_dict(), getattr(new, name).state_dict())
        for key, state in old.optimizer.state_dict()['state'].items():
            assert_weights(state, new.optimizer.state_dict()['state'][key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('backend', ['packed-eager', 'packed-grouped'])
def test_packed_critic_forward_gradient_targets_update_and_restore(backend, tmp_path):
    from gfrl.model import Model, ModelConfig
    from gfrl.rollout import RolloutBatch
    from gfrl.runtime import configure_precision
    configure_precision('cuda', False)
    torch.manual_seed(37)
    model = Model(ModelConfig(d=128, heads=4, depth=2, ff=512)).eval()
    cpu_batch = collected(model)
    batch = RolloutBatch({k:v.cuda() for k,v in cpu_batch.columns.items()},
                         {k:v.cuda() for k,v in cpu_batch.stages.items()})
    cfg = LearnerConfig(critic='privileged', minibatch_size=16, microbatch_size=16)
    eager = Learner(deepcopy(model), deepcopy(cfg), 'cuda', 'bf16')
    cfg.learner_backend = backend
    fast = Learner(deepcopy(model), cfg, 'cuda', 'bf16')
    torch.nn.init.normal_(eager.critic.condition[-1].weight, std=.02)
    fast.critic.load_state_dict(eager.critic.state_dict())
    outputs, gradients = [], []
    for learner in (eager, fast):
        inputs = learner._critic_inputs(batch, torch.arange(len(batch), device='cuda'))
        with learner.autocast():
            logits = learner._critic_forward(*inputs)
        logits.square().mean().backward()
        outputs.append(logits.detach())
        gradients.append(torch.cat([p.grad.flatten() for p in learner.critic.parameters()]))
        assert all(p.grad is None for p in learner.model.parameters())
        learner.critic_optimizer.zero_grad(set_to_none=True)
    torch.testing.assert_close(outputs[0], outputs[1], atol=.035, rtol=.03)
    assert float((gradients[0]-gradients[1]).norm()/gradients[0].norm()) < .08
    baseline = fast._critic_baseline(batch)
    from gfrl.gpu.targets import compute_targets as gpu_targets
    for actual, expected in zip(gpu_targets(baseline), compute_targets(baseline)):
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)
    metrics = fast.update(batch, 1)
    assert np.isfinite(metrics['critic_loss']) and metrics['advantage_std'] > 0
    path = tmp_path/'checkpoint.pt'
    train_module().save_checkpoint(path, fast, 1, [], {}, vars(cfg), 0.)
    state = load_training_checkpoint(path)
    other = Learner(deepcopy(model), deepcopy(cfg), 'cuda', 'bf16')
    other.restore_critic(state, resume=True)
    assert_weights(fast.critic.state_dict(), other.critic.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('streams', [1, 2])
def test_cuda_collection_privileged_labels_and_noisy_pool(streams):
    from gfrl.gpu.collector import GpuCollector
    from gfrl.model import Model, ModelConfig
    model = Model(ModelConfig(d=128, heads=4, depth=1, ff=512)).cuda().eval()
    critic = Learner(model, LearnerConfig(critic='privileged'), 'cuda', 'bf16')
    pool = [critic.pool_entry(1)]
    collector = GpuCollector(model, num_envs=8, seed=35, pool_fraction=1.,
        pool_epsilon=.15, streams=streams, privileged_labels=True)
    try:
        for _ in range(2):
            batch = collector.collect(4, pool)
            assert len(batch) and batch.games >= 4
            torch.testing.assert_close(batch.label_opp_unused_hand.long().sum(-1),
                                       batch.label_opp_unused_hand_count.long(), atol=0, rtol=0)
            assert bool((batch.label_opp_unused_hand <= batch.label_opp_hand).all())
            assert all(not k.startswith('label_') for k in critic._prepare(batch, torch.arange(len(batch),device='cuda')))
            assert np.isfinite(critic.update(batch, 1)['critic_loss'])
    finally:
        collector.close()


@pytest.mark.parametrize('flags', [['--critic','bad'], ['--critic-lr','0'], ['--critic-d','-1'],
    ['--critic-depth','-1'], ['--pool-epsilon','1.01'], ['--pool-epsilon','nan']])
def test_invalid_flags(flags):
    with pytest.raises(SystemExit):
        train_module().parse_args(flags)
