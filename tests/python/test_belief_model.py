"""Belief normalization, autoregression, replay isolation and checkpoint tests."""
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from gfrl import VecEnv
from gfrl.model import Model, ModelConfig, OBS_FIELDS
from gfrl.belief import Belief, IndependentBelief, BeliefConfig, belief_from_checkpoint
from gfrl.belief_training import ReplayBuffer, PositionStream
from gfrl.belief_metrics import fit_baselines


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def model():
    return Belief(ModelConfig(d=16, depth=1, heads=4, ff=32), BeliefConfig(depth=1, heads=4)).eval()


def observation(n=1):
    env = VecEnv(n, 1, 431)
    return {k: torch.from_numpy(getattr(env, k).copy()) for k in OBS_FIELDS}


def test_normalization_all_multisets(model):
    obs = observation()
    with torch.no_grad():
        _, memory, padding = model.memory(obs)
        one = model.decode(memory, padding, torch.zeros(1, 1, dtype=torch.long), torch.ones(1, dtype=torch.long)).exp()
        assert one.sum().item() == pytest.approx(1., abs=1e-4)
        # All 43,956 sorted two-card multisets, including repeated codes.
        a, b = torch.triu_indices(296, 296)+1
        total = 0.
        first = model.decode(memory, padding, torch.zeros(1, 1, dtype=torch.long), torch.tensor([2]))[0, 0]
        second = model.decode(memory.expand(296, -1, -1), padding.expand(296, -1),
                              torch.stack((torch.zeros(296, dtype=torch.long), torch.arange(1, 297)), 1),
                              torch.full((296,), 2))[:, 1]
        total = (first[a-1]+second[a-1, b-1]).double().exp().sum()
        assert total.item() == pytest.approx(1., abs=1e-4)
        assert torch.isneginf(second[20, :20]).all()


def test_teacher_forcing_and_empty(model):
    obs = observation(3)
    labels = torch.zeros(3, 297, dtype=torch.uint8)
    labels[0, 23] = 2; labels[0, 178] = 1
    labels[1, 296] = 1
    total, count, card = model.log_prob(obs, labels)
    cls, mem, pad = model.memory(obs)
    n = labels.sum(1).long()
    expected = []
    for row in range(3):
        seq = torch.repeat_interleave(torch.arange(297), labels[row].long())
        prefix = torch.zeros(1, 1, dtype=torch.long)
        lp = torch.tensor(0.)
        for code in seq:
            lp = lp + model.decode(mem[row:row+1], pad[row:row+1], prefix, n[row:row+1])[0, -1, code-1]
            prefix = torch.cat((prefix, code.reshape(1, 1)), 1)
        expected.append(lp)
    torch.testing.assert_close(card, torch.stack(expected), atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(total, count+card)
    assert card[2] == 0
    all_empty = model.log_prob(obs, labels*0)
    assert torch.isfinite(all_empty[0]).all()
    (-total.mean()).backward()
    assert model.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0


def test_samples_distribution_and_monotonicity(model):
    obs = observation()
    with torch.no_grad():
        model.count_head.weight.zero_(); model.count_head.bias.fill_(-100)
        model.count_head.bias[2] = 100
        model.card_head.weight.zero_(); model.card_head.bias.fill_(-100)
        model.card_head.bias[294:] = 0
    # Three possible hands: (295,295), (295,296), (296,296).
    g = torch.Generator().manual_seed(71)
    prefixes = []
    original = model.decode
    def spy(memory, padding, inputs, counts):
        assert (inputs[:, 1:] >= inputs[:, :-1]).all()
        prefixes.append(inputs.shape[1])
        return original(memory, padding, inputs, counts)
    model.decode = spy
    samples = model.sample(obs, 4000, generator=g)[0]
    assert prefixes == [1, 2]
    assert (samples.sum(1) == 2).all() and not samples[:, 0].any()
    for c in (0, 1, 2):
        labels = torch.zeros(1, 297, dtype=torch.uint8)
        labels[0, 295] = c; labels[0, 296] = 2-c
        p = model.log_prob(obs, labels)[0].exp().item()
        empirical = (samples[:, 295] == c).float().mean().item()
        assert abs(p-empirical) < .035
    with torch.no_grad():
        model.count_head.bias.zero_()
    variable = model.sample(obs, 100)
    assert (variable.sum(-1) <= 18).all()
    assert variable.sum(-1).unique().numel() > 10
    assert model.presence_probs(obs, 8).shape == (1, 297)


@pytest.mark.parametrize("belief_class", [Belief, IndependentBelief])
def test_warmstart_and_label_isolation(tmp_path, belief_class):
    config = ModelConfig(d=16, depth=1, heads=4, ff=32)
    policy = Model(config)
    path = tmp_path/'policy.pt'
    torch.save(dict(ema=policy.state_dict(), model_config=asdict(config),
                    pool=[dict(iteration=1, weights=policy.state_dict())]), path)
    before = hashlib.sha256(path.read_bytes()).digest()
    model = belief_class.from_policy(path, BeliefConfig(heads=4)).eval()
    for k, v in model.state_dict().items():
        if k in policy.state_dict():
            torch.testing.assert_close(v, policy.state_dict()[k], rtol=0, atol=0)
            assert v.data_ptr() != policy.state_dict()[k].data_ptr()
    obs = observation()
    torch.testing.assert_close(model.encode(obs), policy.eval().encode(obs), rtol=0, atol=0)
    public = model.count_probs(obs)
    obs['label_opp_hand'] = torch.ones(1, 297)
    torch.testing.assert_close(public, model.count_probs(obs))
    stream = PositionStream(path, num_envs=4, sample_prob=1, seed=7)
    public_before = {k: getattr(stream.env, k).copy() for k in OBS_FIELDS}
    labels_before = stream.env.label_opp_hand.copy()
    batch = stream.next()
    for k in OBS_FIELDS:
        target = public_before[k].astype(np.float16) if k == 'obs_num' else public_before[k]
        np.testing.assert_array_equal(batch[k].numpy(), target)
    np.testing.assert_array_equal(batch['label_opp_hand'].numpy(), labels_before)
    assert batch['obs_num'].dtype == torch.float16
    # Stream records public observation fields only; labels remain separate replay columns.
    for k in OBS_FIELDS:
        assert batch[k].shape[1:] == obs[k].shape[1:]
    buffer = ReplayBuffer(5, batch, 'cpu')
    buffer.add(batch); buffer.add(batch)
    assert buffer.size == 5
    sampled = buffer.sample(9)
    assert set(sampled) == set(batch)
    assert torch.equal(sampled['label_opp_hand'].sum(1), sampled['label_opp_hand_count'].long())
    assert hashlib.sha256(path.read_bytes()).digest() == before


@pytest.mark.parametrize("belief_class", [Belief, IndependentBelief])
def test_learns_below_b0(belief_class):
    model = belief_class(ModelConfig(d=16, depth=1, heads=4, ff=32), BeliefConfig(depth=1, heads=4)).eval()
    obs = observation(8)
    labels = torch.zeros(8, 297, dtype=torch.uint8)
    labels[:, 20] = 2
    data = dict(label_opp_hand=labels, label_opp_hand_count=labels.sum(1).byte())
    b0 = -fit_baselines(data)['B0'].log_prob(labels).mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = -model.log_prob(obs, labels)[0].mean()
        loss.backward(); optimizer.step()
    model.eval()
    # Held-out observations (same learnable target law, independent environment seed).
    env = VecEnv(8, 1, 191)
    validation = {k: torch.from_numpy(getattr(env, k).copy()) for k in OBS_FIELDS}
    assert -model.log_prob(validation, labels)[0].mean() < b0


def test_resume_optimizer(model, tmp_path):
    spec = importlib.util.spec_from_file_location('train_belief', Path(__file__).resolve().parents[2]/'scripts/train_belief.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    labels = torch.zeros(1, 297, dtype=torch.uint8); labels[:, 13] = 1
    obs = observation()
    (-model.log_prob(obs, labels)[0].mean()).backward(); opt.step()
    path = tmp_path/'checkpoint.pt'
    module.save_checkpoint(path, model, opt, 19, dict(checkpoint='fixed.pt'), 3)
    state = torch.load(path, weights_only=False)
    other = Belief(model.config, model.belief_config)
    other.load_state_dict(state['model'])
    opt2 = torch.optim.AdamW(other.parameters()); opt2.load_state_dict(state['optimizer'])
    assert state['step'] == 19 and state['generation'] == 3
    assert not path.with_suffix('.tmp').exists()
    for a, b in zip(opt.state.values(), opt2.state.values()):
        for k in a:
            torch.testing.assert_close(a[k], b[k])
    for m, o in ((model, opt), (other, opt2)):
        o.zero_grad(); (-m.log_prob(obs, labels)[0].mean()).backward(); o.step()
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("kind", ["autoregressive", "independent"])
def test_streaming_cli_resume_and_metrics(tmp_path, kind, capsys):
    import json
    from gfrl.belief_data import make_belief_data
    config = ModelConfig(d=16, depth=1, heads=4, ff=32)
    weights = Model(config).state_dict()
    policy = tmp_path/'policy.pt'
    torch.save(dict(ema=weights, model_config=asdict(config),
                    pool=[dict(iteration=2, weights=weights)]), policy)
    data = make_belief_data(policy, positions=8, num_envs=4, sample_prob=1, seed=92)
    validation = tmp_path/'validation.pt'
    torch.save(data, validation)
    spec = importlib.util.spec_from_file_location('online_belief', Path(__file__).resolve().parents[2]/'scripts/train_belief.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    argv = ['--model', kind, '--checkpoint', str(policy), '--run-dir', str(tmp_path/'run'), '--steps', '3',
            '--buffer-size', '32', '--warmup-positions', '8', '--batch-size', '4',
            '--microbatch', '2', '--num-envs', '4', '--threads', '1', '--sample-prob', '1',
            '--warmup-steps', '2', '--save-every', '1', '--eval-every', '3',
            '--presence-samples', '2', '--decoder-depth', '1', '--decoder-heads', '4',
            '--fit', str(validation), '--validation', str(validation)]
    checkpoint = module.run(module.parse_args(argv))
    first = torch.load(checkpoint, weights_only=False)
    assert first['step'] == 3
    argv[argv.index('--steps')+1] = '5'
    module.run(module.parse_args([*argv, '--resume']))
    second = torch.load(checkpoint, weights_only=False)
    assert second['step'] == 5 and second['generation'] == 2
    assert {int(s['step']) for s in second['optimizer']['state'].values()} == {5}
    report = json.loads((tmp_path/'run/metrics.jsonl').read_text().splitlines()[-1])
    assert set(report['evaluation'][str(validation)]) == {'belief', 'B0', 'B1'}
    assert report['buffer_positions'] >= 8
    assert report['learn_positions_per_s'] > 0 and report['collect_positions_per_s'] > 0

    assert first['model_type'] == kind
    restored = belief_from_checkpoint(second).eval()
    assert isinstance(restored, IndependentBelief) == (kind == 'independent')
    if kind == 'autoregressive':
        del second['model_type']
        torch.save(second, checkpoint)
        assert type(belief_from_checkpoint(second)) is Belief
        # Exercise actual legacy resume as well as evaluation.
        argv[argv.index('--steps')+1] = '6'
        module.run(module.parse_args([*argv, '--resume']))
        second = torch.load(checkpoint, weights_only=False)
        del second['model_type']
        torch.save(second, checkpoint)
    spec = importlib.util.spec_from_file_location('eval_belief', Path(__file__).resolve().parents[2]/'scripts/eval_belief.py')
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    out = tmp_path/'eval.json'
    result = evaluator.run(evaluator.parse_args(['--checkpoint', str(checkpoint),
        '--validation', str(validation), '--fit', str(validation), '--presence-samples', '3',
        '--presence-sample-batch-size', '2', '--out', str(out)]))
    assert {r['model_type'] for r in result['results'] if 'model_type' in r} == {kind}
    assert len(json.loads(out.read_text())['results']) == 3
    assert '| Data | Model | Multiset NLL |' in capsys.readouterr().out
