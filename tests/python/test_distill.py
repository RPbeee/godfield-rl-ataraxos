"""Search recording parity, public boundary, compact format and distillation smoke."""
from copy import deepcopy
from dataclasses import asdict
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from gfrl import VecEnv, _core
from gfrl.belief_metrics import Baseline, gift_probabilities
from gfrl.distill import (outcome_target, pack_records, save_shard, load_shard,
                         prepare_batch, forward_paths)
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


def test_training_cli_reduces_policy_and_loads(tmp_path):
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
    subprocess.run(command, check=True, capture_output=True, text=True)
    metrics = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['val']['policy'] < metrics[0]['val']['policy']
    trained = load_policy_model(run/'best.pt')
    for k, v in model.value_head.state_dict().items():
        torch.testing.assert_close(v, trained.value_head.state_dict()[k], rtol=0, atol=0)
    assert (run/'last.pt').exists()
    first = torch.load(run/'last.pt', weights_only=False)['iteration']
    subprocess.run(command+['--resume', '--epochs', '3'], check=True, capture_output=True, text=True)
    assert torch.load(run/'last.pt', weights_only=False)['iteration'] > first


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
