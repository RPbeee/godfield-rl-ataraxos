"""Two-policy routing, paired deals, checkpoint loading, and CPU CLI smoke."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from gfrl import VecEnv
from gfrl.actor import Actor
from gfrl.evaluation import evaluate_policies, list_pool_iterations, load_policy_model, restriction_categories
from gfrl.model import Model, ModelConfig, OBS_FIELDS


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def extreme_policy(last=False, seen=None):
    def policy(env, rows):
        assert len(rows) and np.all(env.stage[rows] >= 0)
        if seen is not None:
            seen.extend((int(env.episode_id[r]), int(env.actor_seat[r])) for r in rows)
        choices = np.full(env.num_envs, -999, dtype=np.int32)
        mask = env.mask[rows]
        choices[rows] = 20 - mask[:, ::-1].argmax(1) if last else mask.argmax(1)
        return choices
    return policy


def test_routing_and_swap():
    a, b = [], []
    ab = evaluate_policies(extreme_policy(seen=a), extreme_policy(True, b), 12, 7, 2, 321)
    assert a and b
    assert all(seat == g % 2 for g, seat in a)
    assert all(seat != g % 2 for g, seat in b)
    ba = evaluate_policies(extreme_policy(True), extreme_policy(), 12, 7, 2, 321)
    assert ab['games'] == ba['games'] == 24
    assert ab['wins'] == ba['losses'] and ab['losses'] == ba['wins']
    assert ab['draws'] == ba['draws']
    assert ab['wins'] + ab['losses'] > 0
    assert ab['win_rate'] == pytest.approx(1 - ba['win_rate'])
    assert ab['first_win_rate'] == pytest.approx(1 - ba['second_win_rate'])
    same = evaluate_policies(extreme_policy(), extreme_policy(), 4, 11, 2, 9)
    assert same['win_rate'] == .5
    # Deterministic callbacks do not depend on batching.
    assert ab == evaluate_policies(extreme_policy(), extreme_policy(True), 12, 1, 1, 321)


def test_paired_initial_observations():
    env = VecEnv(10, 2, eval_pairs=5, opponent='external', eval_seed=42)
    bot = VecEnv(10, 2, eval_pairs=5, opponent='greedy', eval_seed=42)
    for key in OBS_FIELDS:
        obs = getattr(env, key)
        np.testing.assert_array_equal(obs[::2], obs[1::2])
        np.testing.assert_array_equal(obs[::2], getattr(bot, key)[::2])


@pytest.fixture
def checkpoints(tmp_path):
    paths = []
    for d in (16, 24):
        config = ModelConfig(d=d, depth=1, heads=4, ff=32)
        torch.manual_seed(d)
        ema = Model(config).state_dict()
        torch.manual_seed(d + 1)
        pool = Model(config).state_dict()
        path = tmp_path / f'model-{d}.pt'
        torch.save(dict(model=pool, ema=ema, model_config=asdict(config),
                        pool=[dict(iteration=30, weights=pool)]), path)
        paths.append(path)
    return paths


def test_load_model_and_pool(checkpoints):
    for path, d in zip(checkpoints, (16, 24)):
        state = torch.load(path, weights_only=False)
        for spec, key in ((str(path), 'ema'), (f'{path}@pool:30', 'model')):
            model = load_policy_model(spec)
            assert model.config.d == d and not model.training
            assert all(not p.requires_grad for p in model.parameters())
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key][name])
        assert list_pool_iterations(path) == [30]
        with pytest.raises(ValueError, match='iteration 99 not found.*available: \\[30\\]'):
            load_policy_model(f'{path}@pool:99')
        with pytest.raises(ValueError, match='integer'):
            load_policy_model(f'{path}@pool:bad')
    empty = checkpoints[0].parent / 'empty.pt'
    torch.save(dict(pool=[]), empty)
    assert list_pool_iterations(empty) == []
    with pytest.raises(ValueError, match='not found'):
        load_policy_model(f'{empty}@pool:30')


def test_actor_sampling_reproducible_and_rng_isolated(checkpoints, monkeypatch):
    env = VecEnv(1, 1, eval_pairs=0)
    a = Actor(load_policy_model(checkpoints[0]), env, 'cpu', sample=True)
    b = Actor(load_policy_model(checkpoints[0]), env, 'cpu', sample=True)
    before = torch.get_rng_state().clone()
    result = evaluate_policies(a, b, 8, 16, 2, 123)
    assert torch.equal(before, torch.get_rng_state())
    assert result == evaluate_policies(a, b, 8, 16, 2, 123)
    assert json.dumps(result) == json.dumps(evaluate_policies(a, b, 8, 16, 2, 123, a_restrict='none'))
    # A simulated clock makes periodic reporting deterministic and fast.
    import gfrl.evaluation as evaluation
    ticks = iter(range(1_000_000))
    monkeypatch.setattr(evaluation, 'monotonic', lambda: next(ticks))
    snapshots = []
    numpy_before = np.random.get_state()
    with_progress = evaluate_policies(a, b, 8, 16, 2, 123,
                                      progress=snapshots.append, progress_interval_s=5)
    assert json.dumps(result) == json.dumps(with_progress)
    assert torch.equal(before, torch.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    counts = [s['games'] for s in snapshots]
    assert len(counts) > 1 and counts == sorted(counts)
    assert any(0 < n < 16 for n in counts)
    assert snapshots[-1] == result
    assert result['games'] == 16
    assert .15 <= result['win_rate'] <= .85
    # New binding resets caches, including when a different-width B is used.
    b = Actor(load_policy_model(checkpoints[1]), env, 'cpu', sample=True)
    assert evaluate_policies(a, b, 1, 4, 1, 123)['games'] == 2


def test_no_restriction_matches_pre_restriction_trace():
    # Recorded with the pre-change native module and evaluator. Include every
    # mask and sampled substep, so matching only the final scores cannot pass.
    rng = np.random.default_rng(991)
    trace = hashlib.sha256()
    def policy(env, rows):
        choices = np.zeros(env.num_envs, np.int32)
        for row in rows:
            choices[row] = rng.choice(np.flatnonzero(env.mask[row]))
        for name in ('episode_id', 'substep', 'stage', 'mask'):
            trace.update(np.asarray(getattr(env, name)[rows]).tobytes())
        trace.update(choices[rows].tobytes())
        return choices
    result = evaluate_policies(policy, policy, 4, 8, 1, 321, a_restrict='none')
    assert trace.hexdigest() == '63549656e7a254e72452ddef76876476d99e5d86810fc8188dfb950fa1351c62'
    assert result == dict(games=8, wins=4, losses=4, draws=0, win_rate=.5,
                          first_win_rate=.5, second_win_rate=.5,
                          mean_decisions=18.625, mean_game_length=15.875,
                          mean_loss_decisions=15.5, mean_win_decisions=16.25,
                          truncation_rate=0., truncations=0,
                          bootstrap_interval=(.5, .5),
                          wilson_interval=(0.21521252682444186, 0.7847874731755582))


def test_official_restrictions_only_a_and_decision_counts():
    seen_a, seen_b = set(), set()
    def policy(seen, restricted):
        def choose(env, rows):
            for r in rows:
                board = env.play_view(int(r), int(env.actor_seat[r]))
                seen.add((int(env.episode_id[r]), board['decisions']))
                if env.stage[r] == 0:
                    if restricted:
                        assert not env.mask[r, 2] and not env.mask[r, 9]
                    else:
                        seen_b_kinds.update(np.flatnonzero(env.mask[r]))
                if restricted and env.stage[r] == 2 and env.chosen_count[r]:
                    hand = env.play_hand(int(r), int(env.actor_seat[r]))
                    base = hand[env.chosen_cards[r, 0]]['face']
                    if env.chosen_kind[r] in (0, 5, 6) and catalog[base]['targets_enemy_by_default']:
                        assert not env.mask[r, 1]
            return extreme_policy(last=True)(env, rows)
        return choose
    from gfrl import _core
    catalog = _core.play_catalog()
    seen_b_kinds = set()
    result = evaluate_policies(policy(seen_a, True), policy(seen_b, False),
                               4, 8, 2, 321, a_restrict='official')
    counts = result['a_restrictions']
    assert seen_a and seen_b and 2 in seen_b_kinds
    assert counts['decisions'] >= len(seen_a)
    assert counts['restricted_decisions'] >= counts['kind'] > 0
    assert counts['restricted_decisions'] >= counts['self_target'] > 0
    assert all(0 <= n <= counts['decisions'] for n in counts.values())
    assert result == evaluate_policies(policy(set(), True), policy(set(), False),
                                      4, 8, 2, 321, a_restrict='official')
    with pytest.raises(ValueError, match='a_restrict'):
        evaluate_policies(None, None, 1, 1, 1, 0, a_restrict='invalid')


@pytest.mark.parametrize('active', [
    'kind', 'dream', 'self_target', 'kind,dream', 'kind,self_target',
    'dream,self_target', 'kind,dream,self_target', 'self_target,dream,kind',
])
def test_restriction_subsets(active):
    categories = restriction_categories(active)
    seen_kinds = set()
    def choose(env, rows):
        for r in rows:
            if env.stage[r] == 0:
                seen_kinds.update(np.flatnonzero(env.mask[r]))
                if 'kind' in categories:
                    assert not env.mask[r, 2] and not env.mask[r, 9]
        return extreme_policy(last=True)(env, rows)
    result = evaluate_policies(choose, extreme_policy(True), 4, 8, 2, 321, a_restrict=active)
    assert result['a_restrict_categories'] == list(categories)
    counts = result['a_restrictions']
    assert counts['decisions'] > 0
    for category in ('kind', 'dream', 'self_target'):
        if category not in categories:
            assert counts[category] == 0
    if 'dream' not in categories:
        assert counts['ambiguous'] == counts['dream_last_resort'] == 0
    if 'kind' not in categories:
        assert 2 in seen_kinds
    if len(categories) == 3:
        assert result == evaluate_policies(choose, extreme_policy(True), 4, 8, 2, 321, a_restrict='official')


@pytest.mark.parametrize('value', ['', 'kind,', ',dream', 'kind,kind', 'official,dream', 'none,kind', 'self', 'Dream'])
def test_invalid_restriction_subsets(value):
    with pytest.raises(ValueError, match='a_restrict'):
        evaluate_policies(None, None, 1, 1, 1, 0, a_restrict=value)


def test_cli(checkpoints, tmp_path):
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'eval_policies.py'
    out = tmp_path / 'nested' / 'results.json'
    command = [sys.executable, str(script), '--a', str(checkpoints[0]),
               '--b', f'{checkpoints[0]}@pool:30', str(checkpoints[1]),
               '--pairs', '1', '--device', 'cpu', '--num-envs', '4',
               '--threads', '1', '--eval-seed', '12345', '--no-tf32',
               '--out', str(out)]
    run = subprocess.run(command, capture_output=True, text=True, check=True)
    assert 'Wilson 95%' in run.stdout and 'Pair bootstrap 95%' in run.stdout
    data = json.loads(out.read_text())
    assert data['eval_seed'] == 12345 and data['tf32'] is False
    assert [r['games'] for r in data['results']] == [2, 2]
    assert 'a_restrict' not in data
    original_json = out.read_bytes()
    explicit_none = subprocess.run(command + ['--a-restrict', 'none'],
                                   capture_output=True, text=True, check=True)
    assert explicit_none.stdout == run.stdout and out.read_bytes() == original_json
    subprocess.run([sys.executable, str(script), '--a', str(checkpoints[0]),
                    '--b', str(checkpoints[0]), '--a-restrict', 'official',
                    '--pairs', '1', '--num-envs', '2', '--threads', '1',
                    '--out', str(out)], capture_output=True, text=True, check=True)
    restricted = json.loads(out.read_text())
    assert restricted['a_restrict'] == 'official'
    assert restricted['a_restrict_categories'] == ['kind', 'dream', 'self_target']
    assert restricted['results'][0]['a_restrictions']['decisions'] > 0
    subprocess.run(command + ['--a-restrict', 'dream,self_target'],
                   capture_output=True, text=True, check=True)
    subset = json.loads(out.read_text())
    assert subset['a_restrict_categories'] == ['dream', 'self_target']
    assert all(r['a_restrictions']['kind'] == 0 for r in subset['results'])
    invalid = subprocess.run(command + ['--a-restrict', 'dream,unknown'],
                             capture_output=True, text=True)
    assert invalid.returncode == 2 and 'a_restrict must be' in invalid.stderr
    run = subprocess.run([sys.executable, str(script), '--list', str(checkpoints[0])],
                         capture_output=True, text=True, check=True)
    assert json.loads(run.stdout)['iterations'] == [30]


def test_external_game_lengths_independent_count():
    lengths = {}
    def policy(env, rows):
        for row in rows:
            if env.substep[row] == 0:
                game = int(env.episode_id[row])
                lengths[game] = lengths.get(game, 0)+1
        return extreme_policy()(env, rows)
    result = evaluate_policies(policy, policy, 4, 5, 1, 321)
    assert result['mean_game_length'] == sum(lengths.values())/8
    assert result['mean_game_length'] <= result['mean_decisions']


def test_summary_win_loss_lengths_and_truncations():
    from gfrl.evaluation import summarize
    rows = [dict(pair=i//2, agent_seat=i%2, result=outcome, truncated=i==3,
                 decisions=100, external_decisions=length)
            for i, (outcome, length) in enumerate([(1, 10), (-1, 20), (1, 30), (0, 40)])]
    result = summarize(rows)
    assert result['mean_game_length'] == 25
    assert result['mean_decisions'] == 100
    assert result['mean_win_decisions'] == 20
    assert result['mean_loss_decisions'] == 20
    assert result['truncations'] == 1
    empty = summarize([])
    assert empty['mean_game_length'] is empty['mean_win_decisions'] is empty['mean_loss_decisions'] is None
    assert empty['truncations'] == 0
