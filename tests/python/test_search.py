"""CPU search formula, information boundary, legal playback and repeatability."""
import json
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from gfrl import VecEnv, _core
from gfrl.actor import Actor
from gfrl.belief import Belief, BeliefConfig, IndependentBelief
from gfrl.belief_metrics import Baseline, gift_probabilities
from gfrl.evaluation import evaluate_policies, _policy_rng
from gfrl.model import Model, ModelConfig, OBS_FIELDS
from gfrl.search import SearchAgent, SearchConfig, search_distribution, load_search_belief


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


def model():
    torch.manual_seed(15)
    return Model(ModelConfig(d=16, heads=2, depth=1, ff=32))


def baseline():
    return Baseline(torch.full((19,), 1/19, dtype=torch.float64), gift_probabilities())


def test_formula_and_distribution():
    rho, pi = np.array([.25, .75]), np.array([.8, .2])
    expected = np.sqrt(rho*pi)
    expected /= expected.sum()
    p = search_distribution([0, 0], np.log(rho), np.log(pi), 1, 1)
    np.testing.assert_allclose(p, expected)
    # Hand calculation: q=[0, 2 log(3)] multiplies candidate 2's weight by 3.
    expected = np.array([np.sqrt(.2), 3*np.sqrt(.15)])
    expected /= expected.sum()
    np.testing.assert_allclose(search_distribution([0, 2*np.log(3)], np.log(rho), np.log(pi), 1, 1), expected)
    assert search_distribution([0, 1], np.log(rho), np.log(pi))[1] > 1-1e-15
    rng = np.random.default_rng(9)
    freq = np.bincount(rng.choice(2, 20000, p=p), minlength=2)/20000
    np.testing.assert_allclose(freq, np.sqrt(rho*pi)/np.sqrt(rho*pi).sum(), atol=.015)
    for kwargs in (dict(depth=-1), dict(worlds=0), dict(alpha=0, beta=0), dict(beta=float('nan'))):
        with pytest.raises(ValueError):
            SearchConfig(**kwargs)


def test_true_hidden_hand_leak():
    a = VecEnv(8, 1, 741, track_projection=True)
    b = VecEnv(8, 1, 741, track_projection=True)
    old_labels = b.label_opp_hand.copy()
    for r in range(8):
        _core._test_replace_hidden_hand(b, r, 123+r)
    assert not np.array_equal(old_labels, b.label_opp_hand)
    for key in OBS_FIELDS:
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    config = SearchConfig(worlds=3, candidates=4, candidate_samples=12, depth=3)
    agents = [SearchAgent(model(), baseline(), config=config, seed=421) for _ in range(2)]
    choices = []
    for agent, env in zip(agents, (a, b)):
        with _policy_rng(42, (agent,)):
            choices.append(agent(env, np.arange(8)))
    np.testing.assert_array_equal(*choices)
    assert agents[0].last_search == agents[1].last_search
    assert agents[0].counts == agents[1].counts
    assert len(agents[0].last_search) == 8


def test_many_roots_legal_and_reproducible():
    config = SearchConfig(worlds=2, candidates=3, candidate_samples=8, depth=2)
    summaries, counts = [], []
    for _ in range(2):
        agent = SearchAgent(model(), baseline(), config=config, seed=221)
        opponent = Actor(model(), VecEnv(1, 1, eval_pairs=0), 'cpu')
        summaries.append(evaluate_policies(agent, opponent, 24, 16, 1, 98, track_projection=True))
        counts.append(dict(agent.counts))
        assert agent.counts['searched'] > 100
        assert agent.counts['real_mismatches'] == 0
    assert summaries[0] == summaries[1]
    assert counts[0] == counts[1]


def test_rejection_exhaustion_and_fallback():
    pn = torch.zeros(19, dtype=torch.float64)
    pn[1] = 1
    pk = torch.zeros(297, dtype=torch.float64)
    pk[0] = 1  # explicitly invalid padding, checked by C++
    agent = SearchAgent(model(), Baseline(pn, pk), config=SearchConfig(worlds=2, max_attempts=3))
    env = VecEnv(2, 1, track_projection=True)
    choices = agent(env, np.arange(2))
    assert env.mask[np.arange(2), choices].all()
    assert agent.counts['exhausted_worlds'] == 4
    assert agent.counts['invalid_samples'] == 12
    assert agent.counts['resamples'] == 8
    assert agent.counts['fallback_no_candidates'] == 2


@pytest.mark.parametrize('kind', ['autoregressive', 'independent'])
def test_checkpoint_beliefs(kind, tmp_path):
    cls = Belief if kind == 'autoregressive' else IndependentBelief
    belief = cls(ModelConfig(d=16, heads=2, depth=1, ff=32), BeliefConfig(depth=1, heads=2))
    path = tmp_path/'belief.pt'
    torch.save(dict(model=belief.state_dict(), model_type=kind, model_config=asdict(belief.config),
                    belief_config=asdict(belief.belief_config)), path)
    loaded = load_search_belief(path)
    agent = SearchAgent(model(), loaded, config=SearchConfig(worlds=1, candidates=2, candidate_samples=3, depth=0, max_attempts=3))
    env = VecEnv(2, 1, track_projection=True)
    choices = agent(env, np.arange(2))
    assert env.mask[np.arange(2), choices].all()


def test_cli(tmp_path):
    checkpoint = tmp_path/'policy.pt'
    m = model()
    torch.save(dict(ema=m.state_dict(), model_config=asdict(m.config)), checkpoint)
    script = Path(__file__).resolve().parents[2]/'scripts/eval_search.py'
    for opponent in ('policy', 'greedy'):
        out = tmp_path/f'{opponent}.json'
        run = subprocess.run([sys.executable, str(script), '--checkpoint', str(checkpoint), '--belief', 'uniform',
                              '--opponent', opponent, '--pairs', '1', '--num-envs', '2', '--threads', '1',
                              '--worlds', '1', '--candidates', '1', '--candidate-samples', '2', '--depth', '0',
                              '--no-tf32', '--out', str(out)], capture_output=True, text=True, check=True)
        assert 'Wilson 95%' in run.stdout
        report = json.loads(out.read_text())
        assert report['games'] == 2 and report['search']['counts']['searched'] > 0
        assert report['config']['tf32'] is False
        assert out.with_suffix('.md').exists()


def test_one_candidate_no_rollout_matches_policy_sampling():
    # With one candidate sample and k=1 the search selects that full pi_theta
    # draw, regardless of the leaf value or magnet. Compare its first-stage
    # marginal against the actual policy head rather than a mirrored formula.
    env = VecEnv(1, 1, 41, track_projection=True)
    m = model()
    agent = SearchAgent(m, baseline(), config=SearchConfig(worlds=1, candidates=1, candidate_samples=1, depth=0), seed=78)
    actor = Actor(m, env, 'cpu')
    expected = np.exp(actor.act().log_probs[0])
    count = np.zeros(21)
    with _policy_rng(51, (agent,)):
        for _ in range(400):
            agent.pending.clear()  # resample the same, unadvanced root
            choices = agent(env, np.array([0]))
            count[choices[0]] += 1
            assert agent.counts['real_mismatches'] == 0
    assert np.abs(count/400-expected).sum() < .18


def test_engine_draw_at_exact_depth_uses_result():
    env = VecEnv(1, 1, 41, track_projection=True)
    _core._test_replace_hidden_hand(env, 0, 1, decisions=511)
    agent = SearchAgent(model(), baseline(), config=SearchConfig(worlds=2, candidates=2, candidate_samples=4, depth=0))
    agent(env, np.array([0]))
    assert agent.last_search
    assert agent.counts['terminal_leaves'] > 0
    assert agent.counts['depth_leaves'] == 0
    assert all(c['q'] == 0 for c in agent.last_search[0]['candidates'])


def test_known_danger_94_area_232_mask_difference_and_fallback():
    env = VecEnv(1, 1, track_projection=True)
    _core._test_danger_root(env, True)
    pn = torch.zeros(19, dtype=torch.float64)
    pn[0] = pn[1] = .5
    pk = torch.zeros(297, dtype=torch.float64)
    pk[209] = 1
    agent = SearchAgent(model(), Baseline(pn, pk), config=SearchConfig(
        worlds=16, candidates=32, candidate_samples=256, depth=0), seed=6)
    with _policy_rng(77, (agent,)):
        choices = agent(env, np.array([0]))
        assert agent.counts['invalid_candidate_worlds'] > 0
        # Locate a sampled candidate whose prefix is incompatible with the
        # real 209 world, then force its selection to exercise fallback.
        incompatible = None
        for candidate in agent.last_search[0]['candidates']:
            assignment = np.zeros((1, 297), np.int32)
            assignment[0, 209] = 1
            world, _ = agent._world_env(env, [0], assignment, [1], 1)
            path = candidate['path']
            for t, choice in enumerate(path):
                if world.decisions[0] or not world.mask[0, choice]:
                    incompatible = path
                    break
                world.step(np.array([choice], np.int32))
            if not world.decisions[0]:
                incompatible = path
            if incompatible is not None:
                break
        assert incompatible is not None
        agent = SearchAgent(model(), baseline(), seed=6)
        agent.search = lambda src, roots: {0: tuple(incompatible)}
        choices = agent(env, np.array([0]))
        for _ in range(20):
            assert env.mask[0, choices[0]]
            env.step(choices)
            if env.decision_done[0]:
                break
            choices = agent(env, np.array([0]))
        else:
            pytest.fail('root decision did not complete')
    assert agent.counts['real_mismatches'] > 0
