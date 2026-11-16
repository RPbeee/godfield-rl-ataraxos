"""Search recording parity, public boundary, compact format and distillation smoke."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
from gfrl import VecEnv, _core
from gfrl.belief_metrics import Baseline, gift_probabilities
from gfrl.distill import (outcome_target, pack_records, save_shard, load_shard,
                         prepare_batch, forward_paths, sharpen_probabilities)
from gfrl.evaluation import _policy_rng, load_policy_model
from gfrl.model import Model, ModelConfig, OBS_FIELDS
from gfrl.search import SearchAgent, SearchConfig


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


def recorded(skip=False):
    torch.manual_seed(17)
    model = Model(ModelConfig(d=16, depth=1, heads=2, ff=32)).eval()
    baseline = Baseline(torch.full((19,), 1/19, dtype=torch.float64), gift_probabilities())
    config = SearchConfig(worlds=2, depth=1, candidates=4, candidate_samples=12,
                          search_response=True, skip_single_candidate=skip, root_batch_size=3)
    agents, envs, choices = [], [], []
    for record in (False, True):
        env = VecEnv(8, 1, 741, track_projection=True)
        agent = SearchAgent(deepcopy(model), baseline, config=config, seed=77, record=record)
        with _policy_rng(43, (agent,)):
            choices.append(agent(env, np.arange(8)))
        agents.append(agent)
        envs.append(env)
    np.testing.assert_array_equal(*choices)
    assert agents[0].last_search == agents[1].last_search
    assert agents[0].counts == agents[1].counts
    assert agents[0].records == []
    assert len(agents[1].records) == 8
    for record in agents[1].records:
        record['outcome'] = outcome_target(1, record['seat'])
    return model, agents[1].records


@pytest.mark.parametrize('skip', [False, True])
def test_record_teacher_forcing_and_roundtrip(tmp_path, skip):
    model, records = recorded(skip)
    data = pack_records(records, dict(seed=900000))
    path = tmp_path/'shard.pt.gz'
    save_shard(path, data)
    restored = load_shard(path)
    for k in ('candidate_offsets', 'stage_offsets', 'log_pi', 'probabilities', 'outcome'):
        torch.testing.assert_close(data[k], restored[k])
    assert set(restored['obs']) == set(OBS_FIELDS)
    batch = prepare_batch(restored, np.arange(8))
    with torch.no_grad():
        _, lp, _ = forward_paths(model, batch)
    torch.testing.assert_close(lp.double(), restored['log_pi'], atol=1e-4, rtol=1e-5)
    for group in ('obs', 'prefix'):
        for k in data[group]:
            torch.testing.assert_close(data[group][k], restored[group][k])


def test_outcome_view_and_truncation():
    np.testing.assert_array_equal(outcome_target(1, 0, 0), [1, 0, 0])
    np.testing.assert_array_equal(outcome_target(1, 1, 0), [0, 1, 0])
    np.testing.assert_array_equal(outcome_target(-1, 0, 1), [1, 0, 0])
    np.testing.assert_array_equal(outcome_target(-1, 1, 1), [0, 1, 0])
    np.testing.assert_array_equal(outcome_target(1, 0, 0, True), [0, 0, 1])
    np.testing.assert_array_equal(outcome_target(0, 1, 0), [0, 0, 1])


@pytest.mark.parametrize('power', [.5, 1., 2., 1000., 1e308, float('inf')])
def test_target_power_normalization_zeros_and_ties(power):
    # Ragged, interleaved roots, a singleton, and probabilities too small for fp32.
    p = torch.tensor([0., .2, .4, .8, .4, 1., .2, 0., 1e-300, 2e-300], dtype=torch.float64)
    owners = torch.tensor([0, 1, 0, 1, 0, 2, 0, 1, 3, 3])
    actual = sharpen_probabilities(p, owners, 4, power)
    if power == 1:
        assert actual is p
    expected = torch.empty_like(p)
    for root in range(4):
        group = p[owners == root]
        if power == 1:
            expected[owners == root] = group
            continue
        elif power == float('inf'):
            weights = (group == group.max()).double()
        else:
            weights = (group/group.max()).pow(power)
        expected[owners == root] = weights/weights.sum()
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    assert (actual[p == 0] == 0).all()
    if power >= 1000:
        hard = torch.tensor([0., 0., .5, 1., .5, 1., 0., 0., 0., 1.], dtype=torch.float64)
        torch.testing.assert_close(actual, hard, atol=1e-100, rtol=0)


@pytest.mark.parametrize('power', [2., float('inf')])
def test_prepared_target_drives_all_losses_and_diagnostics(power):
    from gfrl.distill import objective
    model, records = recorded()
    reference = deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        model.kind_head.bias.add_(torch.linspace(-.3, .3, 10))
    data = pack_records(records, dict(seed=1))
    # Deliberately distinguish maxima in float64 that would tie in float32.
    co = data['candidate_offsets']
    for root in range(len(records)):
        count = int(co[root+1]-co[root])
        p = torch.arange(1, count+1, dtype=torch.float64)
        if count > 1:
            p[-2:] = torch.tensor([count, count+1e-8], dtype=torch.float64)
        data['probabilities'][co[root]:co[root+1]] = p/p.sum()
    rows = np.array([7, 1, 4])  # Rebase and reorder multiple roots.
    original = data['probabilities'].clone()
    expected = prepare_batch(data, rows)
    for root, row in enumerate(rows):
        p = data['probabilities'][co[row]:co[row+1]]
        weights = (p == p.max()).double() if power == float('inf') else p.pow(power)
        expected['probabilities'][expected['candidate_owners'] == root] = (weights/weights.sum()).float()
    actual = prepare_batch(data, rows, target_power=power)
    torch.testing.assert_close(actual['probabilities'], expected['probabilities'])
    torch.testing.assert_close(data['probabilities'], original, atol=0, rtol=0)
    for batch in (actual, expected):
        batch['q'].copy_(torch.linspace(-.8, .8, len(batch['q'])))
    loss, metrics = objective(model, reference, actual, value_target_mix=.5)
    manual_loss, manual_metrics = objective(model, reference, expected, value_target_mix=.5)
    torch.testing.assert_close(loss, manual_loss)
    for key in metrics:
        assert metrics[key] == pytest.approx(manual_metrics[key])
    baseline = prepare_batch(data, rows)
    baseline['q'].copy_(actual['q'])
    _, original_metrics = objective(model, reference, baseline, value_target_mix=.5)
    assert metrics['q_search'] != original_metrics['q_search']
    assert metrics['policy'] != original_metrics['policy']


def test_no_hidden_inputs_both_seats():
    torch.manual_seed(5)
    model = Model(ModelConfig(d=16, depth=1, heads=2, ff=32))
    class PublicBelief:
        def __init__(self):
            self.calls = 0
        def sample(self, obs, n):
            assert set(obs) == set(OBS_FIELDS)
            self.calls += 1
            # Valid empty hypothetical opponent hand; never use actual hands.
            return torch.zeros(len(obs['obs_type']), n, 297, dtype=torch.long)
    envs = [VecEnv(4, 1, 42, track_projection=True) for _ in range(2)]
    _core._test_replace_hidden_hand(envs[1], 0, 912)
    outputs, records = [], []
    for env in envs:
        belief = PublicBelief()
        agent = SearchAgent(deepcopy(model), belief, config=SearchConfig(worlds=1, depth=0,
                            candidates=2, candidate_samples=4), seed=20, record=True)
        with _policy_rng(17, (agent,)):
            outputs.append(agent(env, np.arange(4)))
        records.append(agent.records)
        assert belief.calls
    np.testing.assert_array_equal(*outputs)
    for a,b in zip(*records):
        for k in OBS_FIELDS:
            np.testing.assert_array_equal(a['obs'][k], b['obs'][k])
        assert a['probabilities'] == b['probabilities']


@pytest.mark.parametrize('power', ['1', '3', 'inf'])
def test_training_cli_reduces_policy_and_loads(tmp_path, power):
    model, records = recorded()
    # Independent synthetic games, same public observations, strong teacher.
    expanded = []
    for g in range(40):
        for record in records:
            r = deepcopy(record)
            r['game'] = g
            n = len(r['candidates'])
            r['probabilities'] = [1. if i == n-1 else 0. for i in range(n)]
            expanded.append(r)
    data_dir = tmp_path/'data'
    data_dir.mkdir()
    save_shard(data_dir/'shard-000000.pt.gz', pack_records(expanded, dict(seed=900000)))
    checkpoint = tmp_path/'initial.pt'
    torch.save(dict(model=model.state_dict(), ema=model.state_dict(), model_config=asdict(model.config),
                    iteration=0, config={}, pool=[]), checkpoint)
    run = tmp_path/'train'
    command = [sys.executable, 'scripts/train_distill.py', '--checkpoint', str(checkpoint),
               '--data-dir', str(data_dir), '--run-dir', str(run), '--epochs', '2',
               '--minibatch', '32', '--microbatch', '8', '--validation-fraction', '.3',
               '--lr', '.002', '--value-coef', '0', '--warmup-steps', '0', '--log-every', '1000']
    if power != '1':
        command += ['--target-power', power]
    subprocess.run(command, check=True, capture_output=True, text=True)
    metrics = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    for report in metrics:
        assert {'outcome_ce', 'outcome_brier', 'search_value_mse'} <= report['val'].keys()
        assert report['val']['value_ce'] == report['val']['outcome_ce']
    assert metrics[0]['step'] == 0
    assert metrics[-1]['val']['policy'] < metrics[0]['val']['policy']
    trained = load_policy_model(run/'best.pt')
    for k, v in model.value_head.state_dict().items():
        torch.testing.assert_close(v, trained.value_head.state_dict()[k], rtol=0, atol=0)
    assert (run/'last.pt').exists()
    state = torch.load(run/'last.pt', weights_only=False)
    assert state['config']['target_power'] == float(power)
    first = state['iteration']
    if power == '1':
        # Check backward-compatible resume of a pre-target-power checkpoint.
        del state['config']['target_power']
        torch.save(state, run/'last.pt')
    subprocess.run(command+['--resume', '--epochs', '3'], check=True, capture_output=True, text=True)
    assert torch.load(run/'last.pt', weights_only=False)['iteration'] > first
    changed = subprocess.run(command+['--resume', '--policy-coef', '0'], capture_output=True, text=True)
    assert changed.returncode != 0 and 'resume setting mismatch: policy_coef' in changed.stderr
    changed = subprocess.run(command+['--resume', '--value-target-mix', '.5'], capture_output=True, text=True)
    assert changed.returncode != 0 and 'resume setting mismatch: value_target_mix' in changed.stderr
    changed = subprocess.run(command+['--resume', '--target-power', '2'], capture_output=True, text=True)
    assert changed.returncode != 0 and 'resume setting mismatch: target_power' in changed.stderr


def test_belief_conditions_on_current_actor_and_response_paths():
    torch.manual_seed(92)
    model = Model(ModelConfig(d=16, depth=1, heads=2, ff=32)).eval()
    env = VecEnv(4, 1, eval_pairs=2, opponent='external', eval_seed=900001, track_projection=True)
    class ActingViewBelief:
        def __init__(self):
            self.seats = set()
        def sample(self, obs, n):
            assert set(obs) == set(OBS_FIELDS)
            for j in range(len(obs['obs_type'])):
                matching = [r for r in range(env.num_envs) if env.stage[r] >= 0 and
                            all(np.array_equal(obs[k][j].numpy(), getattr(env, k)[r]) for k in OBS_FIELDS)]
                assert matching  # Exactly a current root's actor-relative view.
                self.seats.update(int(env.actor_seat[r]) for r in matching)
            p = env.obs_schema()['positions']
            samples = torch.zeros(len(obs['obs_type']), n, 297, dtype=torch.long)
            # Hypothetical empty unused hand plus publicly known used cards.
            for j in range(len(samples)):
                codes = obs['obs_cards'][j, p['opponent_used_start']:p['history_start'], 0].long()
                counts = torch.bincount(codes[codes > 0], minlength=297)
                samples[j] = counts
            return samples
    belief = ActingViewBelief()
    agent = SearchAgent(model, belief, config=SearchConfig(worlds=1, depth=0, candidates=1,
                        candidate_samples=6, search_response=True, skip_single_candidate=True),
                        seed=19, record=True)
    with _policy_rng(28, (agent,)):
        while not env.eval_done():
            rows = np.flatnonzero(env.stage >= 0)
            env.step(agent(env, rows))
    assert belief.seats == {0, 1}
    assert any(r['phase'] == 'response' for r in agent.records)
    assert any(c['q'] is None for r in agent.records for c in r['candidates'])
    results = {r['g']: r for r in env.eval_results()}
    for r in agent.records:
        result = results[r['game']]
        r['outcome'] = outcome_target(result['result'], r['seat'], result['agent_seat'], result['truncated'])
    data = pack_records(agent.records, dict(seed=900001))
    with torch.no_grad():
        _, log_pi, _ = forward_paths(model, prepare_batch(data, np.arange(len(agent.records))))
    torch.testing.assert_close(log_pi.double(), data['log_pi'], atol=1e-4, rtol=1e-5)
    for game in results:
        indices = [r['decision'] for r in agent.records if r['game'] == game]
        assert indices == sorted(set(indices))


def test_generator_resume_and_outcomes(tmp_path):
    model, _ = recorded()
    checkpoint = tmp_path/'initial.pt'
    torch.save(dict(model=model.state_dict(), ema=model.state_dict(), model_config=asdict(model.config),
                    iteration=0, config={}, pool=[]), checkpoint)
    out = tmp_path/'data'
    command = [sys.executable, 'scripts/make_distill_data.py', '--checkpoint', str(checkpoint),
               '--belief', 'uniform', '--out-dir', str(out), '--games', '4', '--games-per-shard', '2',
               '--num-envs', '2', '--worlds', '1', '--depth', '0', '--candidates', '2', '--candidate-samples', '4',
               '--skip-single-candidate', '--max-hours', '0']
    subprocess.run(command, check=True, capture_output=True, text=True)
    paths = sorted(out.glob('shard-*.pt.gz'))
    assert len(paths) == 1  # finish one in-flight batch, even with a zero budget
    original = paths[0].read_bytes()
    subprocess.run(command+['--resume', '--max-hours', '.1'], check=True, capture_output=True, text=True)
    paths = sorted(out.glob('shard-*.pt.gz'))
    assert len(paths) == 2
    assert paths[0].read_bytes() == original
    seeds = [load_shard(path)['metadata']['seed'] for path in paths]
    assert seeds == [900000, 900001]
    summary = json.loads((out/'summary.json').read_text())
    assert summary['games'] == 4 and summary['bytes_per_example'] > 0
    assert set(summary['seat_counts']) == {'0', '1'}
    for path in paths:
        d = load_shard(path)
        assert (d['outcome'].sum(1) == 1).all()


def legacy_terms(model, reference, batch):
    """Original objective, retained independently for exact default parity."""
    from torch.nn import functional as F
    lp, paths, value = forward_paths(model, batch)
    with torch.no_grad():
        ref_lp, _, _ = forward_paths(reference, batch)
    mask = batch['prefix']['mask'].bool()
    stage = (lp.exp().masked_fill(~mask, 0) *
             (lp.masked_fill(~mask, 0)-ref_lp.masked_fill(~mask, 0))).sum(-1)
    candidate = paths.new_zeros(len(paths)).index_add(0, batch['stage_owners'], stage)
    p, n = batch['probabilities'], len(value)
    return (-(p*paths).sum()/n, (p*candidate).sum()/n,
            -(batch['outcome']*F.log_softmax(value, -1)).sum(-1).mean())


def assert_gradients_equal(a, b):
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert (pa.grad is None) == (pb.grad is None)
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, atol=0, rtol=0)


@pytest.mark.parametrize('power', [None, 1.])
def test_zero_mix_exact_legacy_loss_and_gradients(power):
    from gfrl.distill import objective
    model, records = recorded()
    reference = deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        model.value_head.bias.add_(.03)
        model.kind_head.bias.add_(torch.linspace(-.1, .1, 10))
    old = deepcopy(model)
    data = pack_records(records, dict(seed=1))
    batch = prepare_batch(data, np.arange(len(records)), **({} if power is None else dict(target_power=power)))
    torch.testing.assert_close(batch['probabilities'], data['probabilities'].float(), atol=0, rtol=0)
    policy, anchor, value = legacy_terms(old, reference, batch)
    expected = policy+.1*anchor+.5*value
    actual, metrics = objective(model, reference, batch, value_target_mix=0.)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert metrics['value_ce'] == metrics['outcome_ce']
    expected.backward()
    actual.backward()
    assert_gradients_equal(model, old)


@pytest.mark.parametrize('mix', [0., .5, 1.])
def test_value_target_expectation_and_missing_search(mix):
    from gfrl.distill import search_values, value_targets
    # Includes a root with only one invalid candidate among several, and Inf.
    batch = dict(outcome=torch.eye(3)[torch.tensor([0, 1, 2, 0, 1, 2])],
                 candidate_owners=torch.tensor([0, 0, 1, 2, 3, 3, 4, 5]),
                 probabilities=torch.tensor([.25, .75, 1., 1., .5, .5, 1., 1.]),
                 q=torch.tensor([-.8, .4, -2., 2., .5, float('nan'), float('nan'), float('inf')]))
    search, valid = search_values(batch)
    torch.testing.assert_close(valid, torch.tensor([True, True, True, False, False, False]))
    target = value_targets(batch['outcome'], search, valid, mix)
    assert torch.isfinite(target).all()
    torch.testing.assert_close(target.sum(-1), torch.ones(6))
    assert (target >= 0).all()
    outcome = batch['outcome'][:, 0]-batch['outcome'][:, 1]
    expectation = torch.where(valid, (1-mix)*outcome+mix*search.clamp(-1, 1), outcome)
    torch.testing.assert_close(target[:, 0]-target[:, 1], expectation)
    torch.testing.assert_close(target[~valid], batch['outcome'][~valid], atol=0, rtol=0)
    if mix == 0:
        assert target is batch['outcome']
    if mix == 1:
        torch.testing.assert_close(target[:3], torch.tensor([[.55, .45, 0.], [0., 1., 0.], [1., 0., 0.]]))


def test_policy_zero_keeps_anchor_loss_and_gradients():
    from gfrl.distill import objective
    model, records = recorded()
    reference = deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        model.kind_head.bias.add_(torch.linspace(-.5, .5, 10))
    manual = deepcopy(model)
    batch = prepare_batch(pack_records(records, dict(seed=1)), np.arange(len(records)))
    _, anchor, ce = legacy_terms(manual, reference, batch)
    expected = .1*anchor+ce
    actual, metrics = objective(model, reference, batch, value_coef=1., policy_coef=0.)
    assert metrics['policy'] > 0 and metrics['anchor_kl'] > 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.backward()
    actual.backward()
    assert_gradients_equal(model, manual)
    assert model.kind_head.bias.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    loss, _ = objective(model, reference, batch, kl_coef=0., value_coef=1., policy_coef=0.)
    loss.backward()
    assert model.kind_head.bias.grad.abs().sum() == 0
    assert model.value_head.weight.grad.abs().sum() > 0
    assert model.type_embedding.weight.grad.abs().sum() > 0


def train_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('train_distill', 'scripts/train_distill.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('power', [3., float('inf')])
def test_training_checkpoint_uses_ema_and_sharpened_validation(tmp_path, power):
    train = train_module()
    model, records = recorded()
    raw_model = deepcopy(model)
    with torch.no_grad():
        raw_model.kind_head.bias.add_(torch.linspace(-1., 1., 10))
        raw_model.value_head.bias.add_(torch.tensor([1., -1., 0.]))
    expanded = []
    for game in range(20):
        for record in records:
            record = deepcopy(record)
            record['game'] = game
            expanded.append(record)
    data = pack_records(expanded, dict(seed=900000))
    data_dir = tmp_path/'data'
    data_dir.mkdir()
    save_shard(data_dir/'shard-000000.pt.gz', data)
    checkpoint = tmp_path/'ckpt-4500.pt'
    torch.save(dict(model=raw_model.state_dict(), ema=model.state_dict(),
                    model_config=asdict(model.config), iteration=4500, config={}, pool=[]), checkpoint)
    run = tmp_path/'train'
    subprocess.run([sys.executable, 'scripts/train_distill.py', '--checkpoint', str(checkpoint),
                    '--data-dir', str(data_dir), '--run-dir', str(run), '--epochs', '1',
                    '--max-hours', '0', '--validation-fraction', '.4',
                    '--target-power', str(power), '--value-target-mix', '.5'],
                   check=True, capture_output=True, text=True)
    exported = torch.load(run/'last.pt', weights_only=False)
    assert exported['optimizer_steps'] == 0
    loaded = load_policy_model(checkpoint).state_dict()
    for key, value in model.state_dict().items():
        for weights in (exported['model'], exported['ema'], loaded):
            torch.testing.assert_close(weights[key], value, atol=0, rtol=0)
    subset = train.select_rows(data, np.flatnonzero(train.validation_mask(data, .4, 17)))
    expected = train.average_metrics(train.validation_parts(
        model, model, [subset], 8, 'cpu', .1, .5, value_target_mix=.5, target_power=power))
    reports = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    for report in reports:
        for key in expected:
            assert report['val'][key] == pytest.approx(expected[key])


@pytest.mark.parametrize('overrides', [{}, {'TARGET_POWER': 'inf', 'VALUE_TARGET_MIX': '.5'}])
def test_session_target_flags_and_evaluation_checkpoint(tmp_path, overrides):
    # Capture commands without running expensive data generation or evaluations.
    fake = tmp_path/'python'
    calls = tmp_path/'calls.jsonl'
    fake.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
                    'with open(os.environ["DISTILL_TEST_CALLS"], "a") as f:\n'
                    '    f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON=str(fake), OUT_DIR=str(tmp_path/'out'),
               DATA_DIR=str(tmp_path/'data'), CHECKPOINT=str(tmp_path/'ckpt-4500.pt'),
               BELIEF='unused-ar.pt', SKIP_REFERENCE='0', DISTILL_TEST_CALLS=str(calls))
    for key in ('TARGET_POWER', 'VALUE_TARGET_MIX'):
        env.pop(key, None)
    env.update(overrides)
    subprocess.run(['bash', 'scripts/session_distill.sh'], env=env, check=True, capture_output=True, text=True)
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    def option(command, flag):
        return command[command.index(flag)+1]
    training = next(c for c in commands if c[0] == 'scripts/train_distill.py')
    assert option(training, '--target-power') == overrides.get('TARGET_POWER', '1')
    assert option(training, '--value-target-mix') == overrides.get('VALUE_TARGET_MIX', '0')
    assert option(training, '--checkpoint') == env['CHECKPOINT']
    distilled = str(tmp_path/'out/train/best.pt')
    plain = next(c for c in commands if c[0] == 'scripts/eval_policies.py')
    assert option(plain, '--a') == distilled
    assert option(plain, '--b') == env['CHECKPOINT']
    search, reference = [c for c in commands if c[0] == 'scripts/eval_search.py']
    assert option(search, '--checkpoint') == distilled
    assert option(search, '--opponent-checkpoint') == env['CHECKPOINT']
    assert option(reference, '--checkpoint') == env['CHECKPOINT']
    assert option(reference, '--opponent') == 'policy'


def test_validation_cache_metrics_match_uncached(tmp_path, monkeypatch):
    from gfrl.distill import objective, select_rows
    train = train_module()
    model, records = recorded()
    reference = deepcopy(model).requires_grad_(False)
    expanded = []
    for g in range(12):
        for record in records:
            record = deepcopy(record)
            record['game'] = g
            if g % 4 == 0:
                record['candidates'][0]['q'] = None
            expanded.append(record)
    paths, cached = [], []
    for seed in (1, 2):
        path = tmp_path/f'shard-{seed}.pt.gz'
        save_shard(path, pack_records(expanded, dict(seed=seed)))
        paths.append(path)
        data = load_shard(path)
        rows = np.flatnonzero(train.validation_mask(data, .4, 17))
        cached.append(select_rows(data, rows))
        original = prepare_batch(data, rows)
        subset = prepare_batch(cached[-1], np.arange(len(rows)))
        for k, value in original.items():
            if isinstance(value, dict):
                for field in value:
                    torch.testing.assert_close(value[field], subset[k][field], atol=0, rtol=0)
            else:
                torch.testing.assert_close(value, subset[k], atol=0, rtol=0, equal_nan=True)
    with torch.no_grad():
        model.value_head.bias.add_(torch.tensor([.2, -.1, 0.]))
        uncached = []
        for path in paths:
            data = load_shard(path)
            rows = np.flatnonzero(train.validation_mask(data, .4, 17))
            for lo in range(0, len(rows), 7):
                rr = rows[lo:lo+7]
                _, metrics = objective(model, reference, prepare_batch(data, rr), .1, 1., .5, 0.)
                uncached.append((len(rr), metrics))
    def no_reload(*args, **kwargs):
        pytest.fail('validation must use cached rows')
    monkeypatch.setattr(train, 'load_shard', no_reload)
    for _ in range(2):
        actual = train.average_metrics(train.validation_parts(model, reference, cached, 7, 'cpu', .1, 1., .5, 0.))
        assert actual == train.average_metrics(uncached)
    # Search MSE uses the finite-q root count rather than all roots as denominator.
    assert actual['search_value_mse'] == pytest.approx(sum(n['search_value_mse']*n['q_roots'] for _, n in uncached)/actual['q_roots'])
    all_missing = deepcopy(cached[0])
    all_missing['q'].fill_(float('nan'))
    missing = train.average_metrics(train.validation_parts(model, reference, [all_missing], 7, 'cpu', .1, 1., 1., 0.))
    assert missing['search_value_mse'] is None
    assert missing['value_ce'] == missing['outcome_ce']
    json.dumps(missing, allow_nan=False)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Metal device unavailable')
def test_distill_mps_compact_columns_and_backward():
    from gfrl.distill import objective
    model, records = recorded()
    data = pack_records(records, dict(seed=1))
    assert data['probabilities'].dtype == torch.float64
    assert any(v.dtype in (torch.int8, torch.uint8) for v in data['prefix'].values())
    batch = prepare_batch(data, np.arange(len(records)), 'mps')
    assert batch['probabilities'].dtype == batch['q'].dtype == torch.float32
    model = model.to('mps')
    reference = deepcopy(model).requires_grad_(False)
    loss, metrics = objective(model, reference, batch, value_target_mix=.5, policy_coef=0.)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.value_head.weight.grad).all()
    assert all(np.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize('mix', [.5, 1.])
def test_objective_soft_targets_and_validation_metrics(mix):
    from gfrl.distill import objective
    model, records = recorded()
    reference = deepcopy(model).requires_grad_(False)
    data = pack_records(records, dict(seed=1))
    batch = prepare_batch(data, np.arange(len(records)))
    # Ensure both searched and fallback roots occur, independently of search behavior.
    batch['q'].copy_(torch.linspace(-.8, .8, len(batch['q'])))
    batch['q'][0] = float('nan')
    _, paths, logits = forward_paths(model, batch)
    targets, valid_search = [], []
    for root in range(len(records)):
        candidates = batch['candidate_owners'] == root
        q, p = batch['q'][candidates], batch['probabilities'][candidates]
        outcome = batch['outcome'][root]
        if torch.isfinite(q).all():
            search = (p*q).sum()
            valid_search.append((root, search))
            v = search.clamp(-1, 1)
            targets.append((1-mix)*outcome+mix*torch.stack(((1+v)/2, (1-v)/2, v.new_zeros(()))))
        else:
            targets.append(outcome)
    expected_ce = -(torch.stack(targets)*logits.log_softmax(-1)).sum(-1).mean()
    expected_policy = -(batch['probabilities']*paths).sum()/len(records)
    loss, metrics = objective(model, reference, batch, kl_coef=0., value_coef=1.,
                              value_target_mix=mix, policy_coef=.3)
    torch.testing.assert_close(loss, .3*expected_policy+expected_ce)
    prediction = logits.softmax(-1)[:, 0]-logits.softmax(-1)[:, 1]
    scalar_outcome = batch['outcome'][:, 0]-batch['outcome'][:, 1]
    assert metrics['value_ce'] == pytest.approx(float(expected_ce.detach()))
    assert metrics['outcome_ce'] == pytest.approx(float(-(batch['outcome']*logits.log_softmax(-1)).sum(-1).mean().detach()))
    assert metrics['outcome_brier'] == pytest.approx(float((prediction-scalar_outcome).square().mean().detach()))
    expected_mse = torch.stack([(prediction[i]-v).square() for i, v in valid_search]).mean()
    assert metrics['search_value_mse'] == pytest.approx(float(expected_mse.detach()))


def test_value_focused_training_cli(tmp_path):
    model, records = recorded()
    expanded = []
    for game in range(20):
        for record in records:
            record = deepcopy(record)
            record['game'] = game
            expanded.append(record)
    data_dir = tmp_path/'data'
    data_dir.mkdir()
    save_shard(data_dir/'shard-000000.pt.gz', pack_records(expanded, dict(seed=900000)))
    checkpoint = tmp_path/'initial.pt'
    torch.save(dict(model=model.state_dict(), ema=model.state_dict(), model_config=asdict(model.config),
                    iteration=0, config={}, pool=[]), checkpoint)
    run = tmp_path/'train'
    subprocess.run([sys.executable, 'scripts/train_distill.py', '--checkpoint', str(checkpoint),
                    '--data-dir', str(data_dir), '--run-dir', str(run), '--epochs', '1',
                    '--minibatch', '32', '--validation-fraction', '.4', '--lr', '.001',
                    '--value-coef', '1', '--policy-coef', '0', '--value-target-mix', '.5',
                    '--log-every', '1000'], check=True, capture_output=True, text=True)
    reports = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert reports[0]['step'] == 0 and reports[-1]['epoch_end']
    assert {'outcome_ce', 'outcome_brier', 'search_value_mse'} <= reports[0]['val'].keys()
    assert reports[-1]['val']['value_ce'] != reports[-1]['val']['outcome_ce']
    trained = load_policy_model(run/'last.pt')
    assert not torch.equal(trained.value_head.weight, model.value_head.weight)
    state = torch.load(run/'last.pt', weights_only=False)
    assert state['config']['policy_coef'] == 0
    assert state['config']['value_target_mix'] == .5


@pytest.mark.parametrize('option,value', [('--target-power', '0'), ('--target-power', '-1'),
                                         ('--target-power', 'nan'), ('--target-power', '-inf'),
                                         ('--value-target-mix', '-.1'), ('--value-target-mix', '1.1'),
                                         ('--value-target-mix', 'nan'), ('--policy-coef', '-1'),
                                         ('--policy-coef', 'nan'), ('--policy-coef', 'inf')])
def test_invalid_loss_options(option, value, monkeypatch, capsys):
    train = train_module()
    monkeypatch.setattr(sys, 'argv', ['train_distill.py', '--checkpoint', 'unused.pt',
                                     '--data-dir', 'unused', '--run-dir', 'unused', option, value])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
    assert 'error:' in capsys.readouterr().err
