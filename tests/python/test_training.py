"""Hand-calculated returns, masked losses, rollout isolation and CPU lifecycle."""
import importlib.util
import json
from pathlib import Path
import weakref

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


def test_resume_overrides_microbatch_size(tmp_path):
    train = train_module()
    argv = ['--device','cpu','--num-envs','16','--threads','2','--games-per-iter','2',
            '--iterations','1','--d','32','--depth','1','--eval-every','0',
            '--run-dir',str(tmp_path)]
    path = train.run(train.parse_args(argv + ['--microbatch-size','64']))
    initial = torch.load(path, weights_only=False)
    assert initial['config']['microbatch_size'] == 64
    train.run(train.parse_args(argv + ['--resume','--microbatch-size','16']))
    resumed = torch.load(path, weights_only=False)
    assert resumed['iteration'] == 2
    assert resumed['config']['microbatch_size'] == 16
    assert json.loads((tmp_path/'config.json').read_text())['microbatch_size'] == 16


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


@pytest.mark.parametrize('eval_every', [0, 1])
def test_training_releases_rollout_before_next_collection(tmp_path, monkeypatch, eval_every):
    train = train_module()
    collectors, batch_refs, column_refs = [], [], []

    class TrackingCollector(Collector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            collectors.append(self)

        def collect(self, *args, **kwargs):
            # Check before collection: the preceding update, evaluation, metrics
            # and checkpoint must retain neither the batch nor its tensor views.
            assert all(ref() is None for ref in batch_refs + column_refs)
            batch = super().collect(*args, **kwargs)
            assert len(self._batch_buffers) == 1
            batch_refs.append(weakref.ref(batch))
            column_refs.extend(weakref.ref(tensor)
                               for columns in (batch.columns, batch.stages)
                               for tensor in columns.values())
            return batch

    monkeypatch.setattr(train, 'Collector', TrackingCollector)
    args = train.parse_args(['--device','cpu','--num-envs','16','--threads','2',
                             '--games-per-iter','2','--iterations','3','--d','32',
                             '--depth','1','--pool-every','1','--pool-size','2',
                             '--eval-every',str(eval_every),'--eval-pairs','1',
                             '--decision-capacity','32','--stage-capacity','64',
                             '--run-dir',str(tmp_path)])
    train.run(args)
    assert len(batch_refs) == 3
    assert len(collectors[0]._batch_buffers) == 1
    assert all(ref() is None for ref in batch_refs + column_refs)


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


@pytest.mark.parametrize('gamma', [.5, .98, .995])
@pytest.mark.parametrize('outcome', [-1, 0, 1])
@pytest.mark.parametrize('lv,la', [(0., 0.), (.8, .5), (1., 1.)])
def test_discounted_targets_reference(gamma, outcome, lv, la):
    # Independent forward-view n-step mixture, no backwards target recursion.
    p = np.array([[.4, .2, .4], [.6, .1, .3], [.3, .4, .3]], dtype=np.float32)
    terminal = np.eye(3)[{1: 0, -1: 1, 0: 2}[outcome]]
    bootstraps = np.vstack([p, terminal])
    values = bootstraps[:, 0]-bootstraps[:, 1]
    expected, advantages, scalar = [], [], []
    draw = np.array([0., 0., 1.])
    for t in range(len(p)):
        distribution = np.zeros(3)
        value_return = 0.
        for n in range(1, len(p)-t+1):
            weight = lv**(n-1) * (1-lv if t+n < len(p) else 1)
            distribution += weight*(gamma**n*bootstraps[t+n]+(1-gamma**n)*draw)
            value_return += weight*gamma**n*values[t+n]
        expected.append(distribution)
        scalar.append(value_return)
        advantages.append(sum((gamma*la)**k*(gamma*values[t+k+1]-values[t+k])
                              for k in range(len(p)-t)))
    records = [Decision({}, 0, 0, 1, row, outcome=outcome) for row in p]
    for target, advantage in (sequence_targets(p, outcome, lv, la, gamma),
                              compute_targets(records, lv, la, gamma)):
        np.testing.assert_allclose(target, expected, atol=1e-7)
        np.testing.assert_allclose(advantage, advantages, atol=1e-7)
        np.testing.assert_allclose(target[:, 0]-target[:, 1], scalar, atol=1e-7)
        np.testing.assert_allclose(target.sum(1), 1, atol=1e-7)


def test_discount_half_hand_computed():
    p = [[.4, .2, .4], [.6, .1, .3]]
    for target, advantage in (sequence_targets(p, 1, gamma=.5),
                              compute_targets([Decision({}, 0, 0, 0, row, outcome=1)
                                               for row in p], gamma=.5)):
        np.testing.assert_allclose(target, [[.26, .01, .73], [.5, 0, .5]], atol=1e-7)
        np.testing.assert_allclose(advantage, [.05, 0], atol=1e-7)


def test_gamma_one_bit_identical_to_original():
    spec = importlib.util.spec_from_file_location('original_targets', Path(__file__).parent/'legacy/targets.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    rng = np.random.default_rng(817)
    records = []
    for i in range(200):
        p = rng.dirichlet([1, 1, 1]).astype(np.float32)
        records.append(Decision({}, i % 3, i % 2, i % 5, p, outcome=(i % 3)-1))
    # Compare each implementation to its own original arithmetic (scalar GAE
    # and batched GAE historically differ in scalar promotion on some NumPy).
    for outcome in (-1, 0, 1):
        p = np.array([r.old_value for r in records])
        for actual, expected in zip(sequence_targets(p, outcome, gamma=1), legacy.sequence_targets(p, outcome)):
            np.testing.assert_array_equal(actual, expected)
    # Load the pre-change batched implementation from an independent frozen fixture.
    spec = importlib.util.spec_from_file_location('original_columnar_targets', Path(__file__).parent/'legacy/columnar_targets.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    for actual, expected in zip(compute_targets(records, gamma=1), original.compute_targets(records)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('gamma', [0, -.1, 1.01, float('nan'), float('inf')])
def test_invalid_gamma(gamma):
    for fn, args in ((sequence_targets, (np.zeros((0, 3)), 0)), (compute_targets, ([],))):
        with pytest.raises(ValueError, match='gamma'):
            fn(*args, gamma=gamma)
    with pytest.raises(ValueError, match='gamma'):
        Learner(tiny(), LearnerConfig(gamma=gamma))
    with pytest.raises(SystemExit):
        train_module().parse_args(['--gamma', str(gamma)])


def test_init_from_full_state_and_config_resume(tmp_path):
    train = train_module()
    source_dir, new_dir = tmp_path/'source', tmp_path/'new'
    common = ['--device', 'cpu', '--threads', '2', '--num-envs', '8', '--games-per-iter', '2',
              '--iterations', '1', '--eval-every', '0', '--pool-every', '1']
    source = train.run(train.parse_args(common+['--d', '32', '--depth', '1', '--run-dir', str(source_dir)]))
    before = torch.load(source, weights_only=False)
    argv = common+['--run-dir', str(new_dir), '--init-from', str(source), '--gamma', '.98',
                   '--lr', '.00003', '--games-per-iter', '3', '--max-hours', '1e-12']
    path = train.run(train.parse_args(argv))
    after = torch.load(path, weights_only=False)
    def equal(a, b):
        if torch.is_tensor(a):
            assert torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b
    for key in ('model', 'ema', 'optimizer', 'optimizer_steps', 'pool', 'iteration',
                'torch_rng', 'numpy_rng', 'model_config'):
        equal(before[key], after[key])
    assert after['seed_positions'] == dict(train=before['seed_positions']['train']+1,
                                         evaluation=before['seed_positions']['evaluation'])
    assert after['config']['gamma'] == .98 and after['config']['lr'] == 3e-5
    assert after['config']['games_per_iter'] == 3
    assert after['elapsed_s'] < before['elapsed_s']
    import hashlib
    with source.open('rb') as f:
        expected = dict(path=str(source.resolve()), md5=hashlib.file_digest(f, 'md5').hexdigest())
    assert after['init_from'] == after['config']['init_from'] == expected
    assert json.loads((new_dir/'config.json').read_text())['init_from'] == expected
    with pytest.raises(ValueError, match='use --resume'):
        train.run(train.parse_args(argv))
    with pytest.raises(ValueError, match='use --resume'):
        train.run(train.parse_args(common+['--run-dir', str(new_dir)]))
    train.run(train.parse_args(common+['--run-dir', str(new_dir), '--resume', 'auto']))
    resumed = torch.load(path, weights_only=False)
    assert resumed['iteration'] == before['iteration']+1
    assert resumed['config']['gamma'] == .98
    assert resumed['config']['games_per_iter'] == 3
    assert resumed['init_from'] == expected
    metrics = json.loads((new_dir/'metrics.jsonl').read_text().splitlines()[-1])
    assert metrics['lr'] == 3e-5
    assert metrics['mean_own_decisions'] > 0


def test_rollout_own_lengths_group_by_seat():
    from gfrl.rollout import RolloutBatch
    columns = {k: torch.tensor(v) for k, v in dict(env=[0]*6, game=[0, 0, 0, 1, 1, 2],
                seat=[0, 0, 1, 0, 0, 1], outcome=[1, 1, -1, -1, -1, 0]).items()}
    assert train_module().rollout_length_metrics(RolloutBatch(columns, {})) == dict(
        mean_own_decisions=1.5, mean_win_own_decisions=2., mean_loss_own_decisions=1.5)


@pytest.mark.parametrize('gamma', [.5, .98])
def test_discounted_interleaved_columnar(gamma):
    from gfrl.rollout import RolloutBatch
    records = [Decision({}, 0, 0, 1, np.array([.4, .2, .4]), outcome=1),
               Decision({}, 1, 1, 1, np.array([.1, .6, .3]), outcome=-1),
               Decision({}, 0, 1, 1, np.array([.3, .3, .4]), outcome=-1),
               Decision({}, 0, 0, 2, np.array([.2, .6, .2]), outcome=0),
               Decision({}, 0, 0, 1, np.array([.6, .1, .3]), outcome=1)]
    expected_target = np.empty((5, 3), dtype=np.float32)
    expected_advantage = np.empty(5, dtype=np.float32)
    for rows in ([0, 4], [1], [2], [3]):
        t, a = sequence_targets([records[i].old_value for i in rows], records[rows[-1]].outcome, gamma=gamma)
        expected_target[rows], expected_advantage[rows] = t, a
    columns = {k: torch.as_tensor(np.array([getattr(r, k) for r in records]))
               for k in ('env', 'game', 'seat', 'old_value', 'outcome')}
    for batch in (records, RolloutBatch(columns, {})):
        target, advantage = compute_targets(batch, gamma=gamma)
        np.testing.assert_allclose(target, expected_target, atol=1e-7)
        np.testing.assert_allclose(advantage, expected_advantage, atol=1e-7)


def test_cuda_backend_accepts_main_training_pool_fraction():
    args = train_module().parse_args(['--device', 'cuda', '--env-backend', 'cuda',
                                      '--dtype', 'bf16', '--size', 'L',
                                      '--learner-backend', 'packed-grouped'])
    assert args.pool_fraction == .2
    assert args.size == 'L'
