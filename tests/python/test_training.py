"""Hand-calculated returns, masked losses, rollout isolation and CPU lifecycle."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gfrl import VecEnv
from gfrl.actor import Actor
from gfrl.learner import Learner, LearnerConfig, decision_terms
from gfrl.model import Model, ModelConfig
from gfrl.rollout import Collector, Decision
from gfrl.targets import compute_targets, sequence_targets


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def tiny():
    return Model(ModelConfig(d=32, depth=1, heads=4, ff=128))


@pytest.mark.parametrize('steps, decay', [(0, .1), (10000, .999)])
def test_ema_decay_uses_optimizer_steps(steps, decay, monkeypatch):
    learner = Learner(torch.nn.Linear(1, 3))
    assert learner.optimizer_steps == 0
    learner.optimizer_steps = steps
    decision = Decision({}, 0, 0, 1, np.array([.4, .2, .4]), outcome=1)

    def forward(decisions):
        value = learner.model(torch.ones(len(decisions), 1))
        zero = value[:, 0] * 0
        return dict(ratio=zero + 1, kl=zero, magnet=zero, entropy=zero), value

    monkeypatch.setattr(learner, 'forward', forward)
    for n in range(2):
        previous = [p.clone() for p in learner.ema.parameters()]
        learner.update([decision], iteration=1)
        assert learner.optimizer_steps == steps + n + 1
        expected_decay = decay if n == 0 else min(.999, (2 + steps) / (11 + steps))
        changed = False
        for before, ema, current in zip(previous, learner.ema.parameters(), learner.model.parameters()):
            changed |= not torch.equal(before, current)
            torch.testing.assert_close(ema, before * expected_decay + current * (1 - expected_decay))
        assert changed


def test_hand_returns_and_gae():
    p = np.array([[.4, .2, .4], [.6, .1, .3], [.3, .4, .3]])
    xi, a = sequence_targets(p, 1)
    np.testing.assert_allclose(xi, [[.808, .084, .108], [.86, .08, .06], [1, 0, 0]], atol=1e-7)
    # E[v]=[.2,.5,-.1], TD errors=[.3,-.6,1.1].
    np.testing.assert_allclose(a, [.275, -.05, 1.1], atol=1e-7)
    xi, a = sequence_targets(p[-1:], 0)
    np.testing.assert_array_equal(xi, [[0, 0, 1]])
    np.testing.assert_allclose(a, [.1], atol=1e-7)


def test_seats_games_and_opponent_skipped():
    records = [Decision({}, 0, 0, 1, np.array([.4, .2, .4]), outcome=1),
               Decision({}, 0, 1, 1, np.array([.1, .6, .3]), outcome=-1),
               Decision({}, 1, 0, 1, np.array([.4, .2, .4]), outcome=0),
               Decision({}, 0, 0, 1, np.array([.6, .1, .3]), outcome=1),
               Decision({}, 0, 0, 2, np.array([.4, .2, .4]), outcome=0)]
    xi, a = compute_targets(records)
    np.testing.assert_allclose(a, [.55, -.5, -.2, .5, -.2], atol=1e-7)
    np.testing.assert_allclose(xi[0], [.92, .02, .06], atol=1e-7)
    np.testing.assert_array_equal(xi[[1, 2, 4]], [[0, 1, 0], [0, 0, 1], [0, 0, 1]])


def test_exact_kl_uniform_mask_and_stage_sum():
    mask = torch.zeros(3, 21, dtype=torch.bool)
    mask[0, :2] = True
    mask[1, [1, 4, 8]] = True
    mask[2, :4] = True
    lp = torch.log_softmax(torch.zeros(3, 21).masked_fill(~mask, -1e9), -1)
    assert (lp.exp()[~mask] == 0).all()
    choice, owners = torch.tensor([0, 4, 2]), torch.tensor([0, 0, 1])
    terms = decision_terms(lp, lp.clone(), mask, choice, owners, 2)
    torch.testing.assert_close(terms['ratio'], torch.ones(2))
    torch.testing.assert_close(terms['kl'], torch.zeros(2))
    torch.testing.assert_close(terms['magnet'], torch.zeros(2))
    torch.testing.assert_close(terms['log_prob'], torch.tensor([-np.log(6), -np.log(4)]).float())
    new = torch.log_softmax(torch.arange(63).reshape(3,21).float().masked_fill(~mask, -1e9), -1)
    changed = decision_terms(new, lp, mask, choice, owners, 2)
    assert changed['kl'].min() > 0
    expected = (new.exp() * (new-lp)).sum(-1)
    torch.testing.assert_close(changed['kl'], torch.stack([expected[:2].sum(), expected[2]]))


class PublicOnly:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('label_'):
            raise AssertionError(name)
        return getattr(self.env, name)


def test_collection_pool_carry_and_encoder_once():
    torch.manual_seed(7)
    model = tiny()
    collector = Collector(model, env=PublicOnly(VecEnv(16, 2, 123)), seed=8, pool_fraction=1.)
    entry = Learner(model).pool_entry(1)
    first = collector.collect(1, [entry])
    assert first.games >= 1 and first.decisions
    for d in first.decisions:
        # Each training decision belongs to the seat opposite its game's fixed pool seat.
        assert d.stages and d.old_value.shape == (3,)
        assert set(d.obs) == {'obs_type','obs_cards','obs_aux_card','obs_cat','obs_num','obs_valid'}
    seats = {}
    for d in first.decisions:
        seats.setdefault((d.env, d.game), set()).add(d.seat)
    assert all(len(s) == 1 for s in seats.values())
    pending = [(d, [s.old_log_probs.copy() for s in d.stages])
               for game in collector.pending for d in game]
    assert pending
    with torch.no_grad():
        next(model.parameters()).add_(.01)
    second = collector.collect(1, [])
    for d, previous in pending:
        for s, old in zip(d.stages, previous):
            np.testing.assert_array_equal(s.old_log_probs, old)
    learner = Learner(model, LearnerConfig(microbatch_size=4, minibatch_size=8))
    batch = first.decisions[:4]
    calls = []
    original = model.encode
    def encode(obs):
        calls.append(obs['obs_type'].shape[0])
        return original(obs)
    model.encode = encode
    terms, value = learner.forward(batch)
    assert calls == [len(batch)]
    expected = [sum(float(s.old_log_probs[s.choice]) for s in d.stages) for d in batch]
    np.testing.assert_allclose(terms['old_log_prob'].detach(), expected, atol=1e-6)
    assert value.shape == (len(batch), 3)
    assert second.decisions


def test_truncation_is_draw_and_no_labels():
    class ShortEnv(PublicOnly):
        def step(self, actions):
            self.env.step(actions)
        def __getattr__(self, name):
            if name == 'truncated':
                return self.env.episode_done
            if name == 'outcome':
                return np.ones((self.env.num_envs, 2), dtype=np.int8)
            return super().__getattr__(name)
    model = tiny()
    env = ShortEnv(VecEnv(16, 2, 6))
    actor = Actor(model, env, 'cpu')
    assert actor.act().log_probs.shape == (16,21)
    rollout = Collector(model, env=env).collect(1)
    assert rollout.truncated == rollout.games
    assert all(d.outcome == 0 for d in rollout.decisions)


def train_module():
    spec = importlib.util.spec_from_file_location('train', Path(__file__).parents[2]/'scripts/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_resume_schedule_pool_and_config(tmp_path):
    train = train_module()
    base = ['--device','cpu','--num-envs','16','--threads','2','--games-per-iter','2',
            '--d','32','--depth','1','--eval-every','0','--pool-every','1','--pool-size','2',
            '--magnet','--lr-decay','--lr-c','0.0001','--filtering']
    def run(folder, count, resume=False):
        args = train.parse_args(base+['--run-dir',str(folder),'--iterations',str(count)]
                                + (['--resume'] if resume else []))
        path = train.run(args)
        return torch.load(path, weights_only=False)
    separate, continuous = tmp_path/'resume', tmp_path/'full'
    initial = run(separate, 2)
    resumed = run(separate, 1, True)
    full = run(continuous, 3)
    rows = [json.loads(s) for s in (separate/'metrics.jsonl').read_text().splitlines()]
    other = [json.loads(s) for s in (continuous/'metrics.jsonl').read_text().splitlines()]
    assert len(rows) == len(other) == 3
    assert resumed['iteration'] == full['iteration'] == 3
    assert [p['iteration'] for p in resumed['pool']] == [2, 3]
    assert [p['iteration'] for p in full['pool']] == [2, 3]
    assert resumed['seed_positions']['train'] == 2
    assert full['seed_positions']['train'] == 1
    assert rows[-1]['lr'] == other[-1]['lr']
    assert rows[-1]['alpha'] == other[-1]['alpha']
    assert rows[-1]['lr'] < rows[0]['lr'] and rows[-1]['alpha'] < rows[0]['alpha']
    assert resumed['optimizer']['state']
    expected_steps = [sum((row['decisions'] + 4095) // 4096 for row in history)
                      for history in (rows[:2], rows, other)]
    assert [state['optimizer_steps'] for state in (initial, resumed, full)] == expected_steps
    config = tmp_path/'overrides.json'
    config.write_text(json.dumps({'iterations': 7, 'magnet': True}))
    assert train.parse_args(['--iterations','2','--config',str(config)]).iterations == 7


def test_default_ppo_three_iterations_with_evaluation(tmp_path):
    train = train_module()
    args = train.parse_args(['--device','cpu','--num-envs','16','--threads','2',
                             '--games-per-iter','2','--iterations','3','--d','32',
                             '--depth','1','--eval-every','3','--eval-pairs','2',
                             '--run-dir',str(tmp_path)])
    train.run(args)
    rows = [json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    assert all(row['alpha'] == 0 and row['filtering_fraction'] == 1 for row in rows)
    assert all(row['lr'] == 1e-4 for row in rows)
    assert set(rows[-1]['evaluation']) == {'random', 'greedy'}
    assert rows[-1]['evaluation']['random']['games'] == 4


def test_evaluation_rng_does_not_advance_training_sampler():
    train = train_module()
    torch.manual_seed(13)
    expected = torch.rand(4)
    torch.manual_seed(13)
    with train.evaluation_rng(42, 'cpu'):
        evaluated = torch.rand(4)
    torch.testing.assert_close(torch.rand(4), expected)
    with train.evaluation_rng(42, 'cpu'):
        torch.testing.assert_close(torch.rand(4), evaluated)


def test_time_limit_saves_at_boundary(tmp_path):
    train = train_module()
    args = train.parse_args(['--d','32','--depth','1','--num-envs','16','--threads','2',
                             '--max-hours','1e-12','--run-dir',str(tmp_path)])
    path = train.run(args)
    state = torch.load(path, weights_only=False)
    assert state['iteration'] == 0
    assert state['seed_positions']['train'] == 1
    assert not (tmp_path/'metrics.jsonl').exists()


@pytest.mark.parametrize('saved_steps', [37, None])
def test_resume_optimizer_steps_without_updates(tmp_path, saved_steps):
    train = train_module()
    argv = ['--d','32','--depth','1','--num-envs','16','--threads','2',
            '--max-hours','1e-12','--run-dir',str(tmp_path)]
    path = train.run(train.parse_args(argv))
    state = torch.load(path, weights_only=False)
    assert state['optimizer_steps'] == 0
    if saved_steps is None:
        del state['optimizer_steps']
    else:
        state['optimizer_steps'] = saved_steps
    torch.save(state, path)
    train.run(train.parse_args(argv + ['--resume']))
    resumed = torch.load(path, weights_only=False)
    assert resumed['optimizer_steps'] == (saved_steps or 0)
