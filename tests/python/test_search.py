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


@pytest.mark.parametrize('case', json.loads((Path(__file__).parent/'fixtures/search_before.json').read_text()))
def test_pre_optimization_reference(case):
    # Frozen with the original search.py/actor.py, before compact transfer,
    # root encoding sharing, vectorized forcing, and workspace reuse.
    env = VecEnv(4, 1, 741, track_projection=True)
    agent = SearchAgent(model(), baseline(), config=SearchConfig(**case['config']), seed=421)
    with _policy_rng(42, (agent,)):
        choices = agent(env, np.arange(4, dtype=np.int32))
    assert choices.tolist() == case['choices']
    for actual, expected in zip(agent.last_search, case['reports'], strict=True):
        assert actual['selected'] == expected['selected']
        assert [c['path'] for c in actual['candidates']] == [c['path'] for c in expected['candidates']]
        np.testing.assert_allclose([c['q'] for c in actual['candidates']],
                                   [c['q'] for c in expected['candidates']], atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(actual['probabilities'], expected['probabilities'], atol=1e-6)


def test_fixed_candidates_and_workspace_reload():
    env = VecEnv(4, 1, 741, track_projection=True)
    agent = SearchAgent(model(), baseline(), config=SearchConfig(worlds=3, candidates=4, candidate_samples=12, depth=0))
    rows = np.arange(4, dtype=np.int32)
    with _policy_rng(42, (agent,)):
        agent.search(env, rows, report_values=True)
    fixed, policy = agent.last_candidates, agent.last_policy_paths
    saved = []
    for _ in range(2):
        agent.rng = np.random.default_rng(123)
        with _policy_rng(456, (agent,)):
            agent.search(env, rows, fixed_candidates=fixed, policy_paths=policy, report_values=True)
        saved.append(agent.last_search)
        assert agent.last_candidates == fixed
        for report in agent.last_search:
            assert [tuple(c['path']) for c in report['candidates']] == [c[0] for c in fixed[report['row']]]
            for c in report['candidates']:
                assert c['q'] == np.mean(c['world_values'])
                assert c['q_se'] == pytest.approx(np.std(c['world_values'], ddof=1)/np.sqrt(c['valid_worlds']))
        workspace = agent._workspaces['rollout'][1]
        if len(saved) == 1:
            previous = workspace
        else:
            assert workspace is previous
    assert saved[0] == saved[1]


def test_depth_zero_stops_at_next_policy_boundary():
    env = VecEnv(2, 1, 741, track_projection=True)
    agent = SearchAgent(model(), baseline(), config=SearchConfig(worlds=2, candidates=3, candidate_samples=8, depth=0))
    original = agent._world_env
    loaded = []
    def capture(src, rows, assignments, seeds, depth, role=None):
        if role == 'rollout':
            loaded.append((rows.copy(), assignments.copy(), seeds.copy()))
        return original(src, rows, assignments, seeds, depth, role)
    agent._world_env = capture
    with _policy_rng(42, (agent,)):
        agent.search(env, np.arange(2, dtype=np.int32))
    stopped = agent._workspaces['rollout'][1]
    rows, assignments, seeds = loaded[0]
    live, status = original(env, rows, assignments, seeds, 2)
    assert not status.any()
    # Complete just the forced root, stopping manually at the next live policy
    # boundary. Compare it to max_decisions=1's stopped leaf observations.
    paths = [tuple(c['path']) for r in agent.last_search for c in r['candidates'] for _ in range(2)]
    cursor = np.zeros(len(paths), int)
    boundaries = {}
    while np.any(live.decisions == 0):
        active = np.flatnonzero((live.decisions == 0) & (live.stage >= 0))
        assert len(active)
        choices = np.zeros(live.num_envs, np.int32)
        choices[live.stage >= 0] = live.mask[live.stage >= 0].argmax(1)
        for i in active:
            choices[i] = paths[i][cursor[i]]
            cursor[i] += 1
        live.step(choices)
        for i in np.flatnonzero(live.decisions == 1):
            if int(i) not in boundaries:
                observed = live.observe(np.array([i], np.int32), int(live.searcher_seat[i]))
                boundaries[int(i)] = ({k: v.copy() for k, v in observed.items()},
                                      int(live.stage[i]), int(live.substep[i]))
    assert np.all(stopped.decisions == 1)
    cutoff = np.flatnonzero(stopped.truncated).astype(np.int32)
    assert len(cutoff)
    for i in cutoff:
        boundary, stage, substep = boundaries[int(i)]
        assert stage >= 0 and substep == 0
        leaf = stopped.observe(np.array([i], np.int32), int(stopped.searcher_seat[i]))
        for key in OBS_FIELDS:
            np.testing.assert_array_equal(leaf[key], boundary[key])


def test_noise_cli_twenty_roots(tmp_path):
    checkpoint = tmp_path/'policy.pt'
    m = model()
    torch.save(dict(ema=m.state_dict(), model_config=asdict(m.config)), checkpoint)
    out = tmp_path/'noise.json'
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2]/'scripts/search_noise.py'),
                    '--checkpoint', str(checkpoint), '--belief', 'uniform', '--roots', '20', '--num-envs', '8',
                    '--worlds', '2', '--depth', '0', '3', '--beta', '.02', '.1', '--candidates', '3',
                    '--candidate-samples', '6', '--repeats', '3', '--out', str(out)], check=True, capture_output=True)
    report = json.loads(out.read_text())
    assert len(report['root_locations']) == 20
    assert len(report['results']) == 4
    for result in report['results']:
        assert len(result['roots']) == 20
        assert result['metrics']['failed_searches'] == 0
        assert result['metrics']['repeat_pairs'] == 60
        for repeats in result['roots']:
            assert all(r['candidates'] for r in repeats)
            assert len({tuple(tuple(c['path']) for c in r['candidates']) for r in repeats}) == 1
    assert 'centered_q_correlation' in out.with_suffix('.md').read_text()


def test_noise_metrics_defined_and_missing_cases():
    import importlib.util
    script = Path(__file__).resolve().parents[2]/'scripts/search_noise.py'
    spec = importlib.util.spec_from_file_location('noise_metrics', script)
    noise = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(noise)
    def report(q, probs, selected):
        return dict(candidates=[dict(path=[i], q=v, q_se=.1) for i, v in enumerate(q)],
                    probabilities=probs, selected=[selected], policy_path=[0])
    a, b = report([1., 3.], [.8, .2], 0), report([2., 4.], [.3, .7], 1)
    m = noise.metrics([[a, b]])
    assert m['q_correlation'] == pytest.approx(1)
    assert m['centered_q_correlation'] == pytest.approx(1)
    assert m['total_variation'] == pytest.approx(.5)
    assert m['argmax_agreement'] == m['sample_agreement'] == 0
    assert m['q_argmax_agreement'] == 1
    assert m['policy_agreement'] == .5
    assert m['se_gap_ratio'] == pytest.approx(np.sqrt(.02)/2)
    b['candidates'].pop()
    b['probabilities'] = [1.]
    m = noise.metrics([[a, b], [None, None]])
    assert m['missing_candidate_pairs'] == 1
    assert m['failed_searches'] == 2
    assert m['centered_q_correlation'] is None
    assert noise.metrics([])['total_variation'] is None
