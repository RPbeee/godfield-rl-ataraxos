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
    for kwargs in (dict(depth=-1), dict(worlds=0), dict(alpha=0, beta=0), dict(beta=float('nan')),
                   dict(leaf_view='invalid'), dict(max_inference_batch=-1), dict(search_cache_device='bad'),
                   dict(search_cache_mb=-1), dict(root_batch_size=-1)):
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


@pytest.mark.parametrize('belief', ['uniform', 'oracle'])
@pytest.mark.parametrize('progress_interval', ['0', '0.01'])
def test_cli(tmp_path, belief, progress_interval):
    checkpoint = tmp_path/'policy.pt'
    m = model()
    torch.save(dict(ema=m.state_dict(), model_config=asdict(m.config)), checkpoint)
    script = Path(__file__).resolve().parents[2]/'scripts/eval_search.py'
    for opponent in ('policy', 'greedy'):
        out = tmp_path/f'{opponent}.json'
        run = subprocess.run([sys.executable, str(script), '--checkpoint', str(checkpoint), '--belief', belief,
                              '--opponent', opponent, '--pairs', '1', '--num-envs', '2', '--threads', '1',
                              '--worlds', '1', '--candidates', '1', '--candidate-samples', '2', '--depth', '0',
                              '--no-tf32', '--progress-interval', progress_interval,
                              '--out', str(out)], capture_output=True, text=True, check=True)
        assert 'Wilson 95%' in run.stdout
        report = json.loads(out.read_text())
        assert report['games'] == 2 and report['search']['counts']['searched'] > 0
        assert report['config']['tf32'] is False
        assert report['belief'] == belief
        assert report['diagnostic_only'] is (belief == 'oracle')
        assert report['search']['counts']['invalid_oracle_assignments'] == 0
        assert out.with_suffix('.md').exists()
        partial_path = out.with_suffix('.partial.json')
        if progress_interval == '0':
            assert not partial_path.exists() and 'elapsed=' not in run.stdout
        else:
            partial = json.loads(partial_path.read_text())
            for key in ('games', 'wins', 'losses', 'draws', 'win_rate', 'wilson_interval',
                        'bootstrap_interval', 'mean_decisions', 'truncation_rate',
                        'first_win_rate', 'second_win_rate', 'search'):
                assert partial[key] == report[key]
            assert partial['total_games'] == 2 and partial['estimated_remaining_seconds'] == 0
            assert 'games=2/2' in run.stdout and 'ETA=0.0s' in run.stdout


def test_oracle_many_evaluation_roots():
    env = VecEnv(16, 2, eval_pairs=10000, opponent='external', eval_seed=741, track_projection=True)
    agent = SearchAgent(model(), load_search_belief('oracle'), config=SearchConfig(worlds=3), seed=421)
    rng = np.random.default_rng(51)
    expected_rng = np.random.default_rng(421)
    tested = 0
    for _ in range(600):
        fresh = np.flatnonzero((env.stage >= 0) & (env.substep == 0)).astype(np.int32)
        roots = fresh[env.determinize_support(fresh) == 0]
        if len(roots):
            rr, assignments, seeds = agent._sample_worlds(env, roots)
            np.testing.assert_array_equal(rr, np.repeat(roots, 3))
            np.testing.assert_array_equal(assignments, env.label_opp_hand[rr])
            np.testing.assert_array_equal(assignments.sum(1), env.label_opp_hand_count[rr])
            # Only chance seeds consume the search RNG in oracle mode.
            np.testing.assert_array_equal(seeds, expected_rng.integers(
                0, np.iinfo(np.uint64).max, len(rr), dtype=np.uint64))
            assert len(np.unique(seeds)) == len(seeds)
            world, status = agent._world_env(env, rr, assignments, seeds, 1)
            assert not status.any()
            np.testing.assert_array_equal(world.label_opp_hand, assignments)
            tested += len(roots)
        choices = np.array([rng.choice(np.flatnonzero(m)) if s >= 0 else 0
                            for m, s in zip(env.mask, env.stage)], np.int32)
        env.step(choices)
    assert tested > 1000
    assert agent.counts['invalid_oracle_assignments'] == 0
    assert agent.counts['invalid_samples'] == agent.counts['invalid_loads'] == 0


def test_invalid_oracle_assignment_is_counted_once_per_world():
    env = VecEnv(2, 1, track_projection=True)
    class BadLabels:
        label_opp_hand = np.zeros((2, 297), np.int32)
        label_opp_hand[:, 0] = 1
        unused_assignment = env.unused_assignment
    agent = SearchAgent(model(), load_search_belief('oracle'), config=SearchConfig(worlds=3))
    rr, assignments, seeds = agent._sample_worlds(BadLabels(), np.arange(2))
    assert len(rr) == len(assignments) == len(seeds) == 0
    assert agent.counts['invalid_oracle_assignments'] == agent.counts['invalid_samples'] == 6
    assert agent.counts['exhausted_worlds'] == 6
    assert agent.counts['resamples'] == 0


@pytest.mark.parametrize('skip', [False, True])
def test_one_candidate_no_rollout_matches_policy_sampling(skip):
    # With one candidate sample and k=1 the search selects that full pi_theta
    # draw, regardless of the leaf value or magnet. Compare its first-stage
    # marginal against the actual policy head rather than a mirrored formula.
    env = VecEnv(1, 1, 41, track_projection=True)
    m = model()
    agent = SearchAgent(m, baseline(), config=SearchConfig(worlds=1, candidates=1, candidate_samples=1, depth=0, skip_single_candidate=skip), seed=78)
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


def test_danger_94_area_232_worlds_share_restricted_masks():
    env = VecEnv(1, 1, track_projection=True)
    _core._test_danger_root(env, True)
    pn = torch.zeros(19, dtype=torch.float64)
    pn[0] = pn[1] = .5
    pk = torch.zeros(297, dtype=torch.float64)
    pk[209] = 1
    agent = SearchAgent(model(), Baseline(pn, pk), config=SearchConfig(
        worlds=16, candidates=32, candidate_samples=256, depth=1), seed=6)
    with _policy_rng(77, (agent,)):
        choices = agent(env, np.array([0]))
        assert agent.counts['invalid_candidate_worlds'] == 0
        for candidate in agent.last_search[0]['candidates']:
            for mortar in (False, True):
                assignment = np.zeros((1, 297), np.int32)
                if mortar:
                    assignment[0, 209] = 1
                world, _ = agent._world_env(env, [0], assignment, [1], 1)
                for choice in candidate['path']:
                    assert world.mask[0, choice]
                    world.step(np.array([choice], np.int32))
                assert world.decisions[0]
        for _ in range(20):
            assert env.mask[0, choices[0]]
            env.step(choices)
            if env.decision_done[0]:
                break
            choices = agent(env, np.array([0]))
        else:
            pytest.fail('root decision did not complete')
    assert agent.counts['real_mismatches'] == 0


@pytest.mark.parametrize('case', json.loads((Path(__file__).parent/'fixtures/search_before.json').read_text()))
def test_pre_optimization_reference(case):
    # Frozen with the original search.py/actor.py, before compact transfer,
    # root encoding sharing, vectorized forcing, and workspace reuse.
    env = VecEnv(4, 1, 741, track_projection=True)
    agent = SearchAgent(model(), baseline(), config=SearchConfig(**case['config'], leaf_view='searcher'), seed=421)
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
                actor_seat = int(live.actor_seat[i])
                actor_obs = live.observe(np.array([i], np.int32), actor_seat)
                # These are precisely the buffers Actor encodes at substep 0.
                for key in OBS_FIELDS:
                    np.testing.assert_array_equal(actor_obs[key][0], getattr(live, key)[i])
                boundaries[int(i)] = ({k: v.copy() for k, v in actor_obs.items()},
                                      int(live.stage[i]), int(live.substep[i]), actor_seat)
    assert np.all(stopped.decisions == 1)
    cutoff = np.flatnonzero(stopped.truncated).astype(np.int32)
    assert len(cutoff)
    assert {boundaries[int(i)][3] == stopped.searcher_seat[i] for i in cutoff} == {True, False}
    # A response (phase enum 1 -> category 2) immediately after root attack.
    phase = env.obs_schema()['cat']['phase']['index']
    assert any(boundaries[int(i)][0]['obs_cat'][0, 0, phase] == 2
               and boundaries[int(i)][3] != stopped.searcher_seat[i] for i in cutoff)
    for i in cutoff:
        boundary, stage, substep, actor_seat = boundaries[int(i)]
        assert stage >= 0 and substep == 0
        assert stopped.next_actor_seats(np.array([i], np.int32))[0] == actor_seat
        leaf = stopped.observe(np.array([i], np.int32), actor_seat)
        for key in OBS_FIELDS:
            np.testing.assert_array_equal(leaf[key], boundary[key])


@pytest.mark.parametrize('belief', ['uniform', 'oracle'])
def test_noise_cli_twenty_roots(tmp_path, belief):
    checkpoint = tmp_path/'policy.pt'
    m = model()
    torch.save(dict(ema=m.state_dict(), model_config=asdict(m.config)), checkpoint)
    out = tmp_path/'noise.json'
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2]/'scripts/search_noise.py'),
                    '--checkpoint', str(checkpoint), '--belief', belief, '--roots', '20', '--num-envs', '8',
                    '--worlds', '2', '--depth', '0', '3', '--beta', '.02', '.1', '--candidates', '3',
                    '--candidate-samples', '6', '--repeats', '3', '--out', str(out)], check=True, capture_output=True)
    report = json.loads(out.read_text())
    assert report['belief'] == belief
    assert report['diagnostic_only'] is (belief == 'oracle')
    assert len(report['root_locations']) == 20
    assert len(report['results']) == 4
    for result in report['results']:
        assert result['counts']['invalid_oracle_assignments'] == 0
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


@pytest.mark.parametrize('view', ['actor', 'searcher'])
@pytest.mark.parametrize('depth', [0, 3])
def test_search_inference_chunks(view, depth):
    from dataclasses import replace
    config = SearchConfig(worlds=3, candidates=4, candidate_samples=12, depth=depth, leaf_view=view)
    reports, choices = [], []
    for cap in (0, 1, 5):
        env = VecEnv(4, 1, 741, track_projection=True)
        agent = SearchAgent(model(), baseline(), config=replace(config, max_inference_batch=cap), seed=421)
        if cap:
            encode, heads = agent.model.encode, agent.model.heads
            def bounded_encode(obs):
                assert len(obs['obs_type']) <= cap
                return encode(obs)
            def bounded_heads(tokens, **prefix):
                assert len(tokens) <= cap
                return heads(tokens, **prefix)
            agent.model.encode, agent.model.heads = bounded_encode, bounded_heads
        with _policy_rng(42, (agent,)):
            choices.append(agent.search(env, np.arange(4, dtype=np.int32)))
        reports.append(agent.last_search)
        world = agent._workspaces['rollout'][1]
        rows = np.arange(world.num_envs, dtype=np.int32)
        phase = env.obs_schema()['cat']['phase']['index']
        leaf = world.observe(rows, world.searcher_seat.astype(np.int32))
        np.testing.assert_array_equal(world.next_actor_seats(rows) < 0, leaf['obs_cat'][:, 0, phase] == 3)
        assert agent.counts['depth_leaves'] == (agent.counts['depth_leaves_searcher']
                                              + agent.counts['depth_leaves_opponent'])
    for actual, choice in zip(reports[1:], choices[1:]):
        assert choice == choices[0]
        for a, b in zip(actual, reports[0], strict=True):
            assert a['selected'] == b['selected']
            assert [c['path'] for c in a['candidates']] == [c['path'] for c in b['candidates']]
            for key in ('q', 'log_pi', 'log_rho'):
                np.testing.assert_allclose([c[key] for c in a['candidates']],
                                           [c[key] for c in b['candidates']], atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(a['probabilities'], b['probabilities'], atol=2e-6, rtol=2e-6)


def test_opponent_leaf_value_sign():
    env = VecEnv(4, 1, 741, track_projection=True)
    m = model()
    with torch.no_grad():
        m.value_head.weight.zero_()
        m.value_head.bias.copy_(torch.tensor([2., 0., 0.]))
    agent = SearchAgent(m, baseline(), config=SearchConfig(worlds=2, candidates=4, candidate_samples=12, depth=0))
    with _policy_rng(42, (agent,)):
        agent.search(env, np.arange(4, dtype=np.int32), report_values=True)
    world = agent._workspaces['rollout'][1]
    rows = np.arange(world.num_envs, dtype=np.int32)
    seats = world.next_actor_seats(rows)
    v = float((torch.tensor([2., 0., 0.]).softmax(0)[0]-torch.tensor([2., 0., 0.]).softmax(0)[1]))
    expected = np.where(seats == world.searcher_seat, v, -v)
    actual = [v for report in agent.last_search for c in report['candidates'] for v in c['world_values']]
    assert (seats != world.searcher_seat).any()
    np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize('cls', [Belief, IndependentBelief])
def test_belief_inference_chunks(cls):
    from dataclasses import replace
    torch.manual_seed(34)
    belief = cls(ModelConfig(d=16, heads=2, depth=1, ff=32), BeliefConfig(depth=1, heads=2)).eval()
    env = VecEnv(4, 1, 741, track_projection=True)
    config = SearchConfig(worlds=2, candidates=3, candidate_samples=8, depth=3)
    reports = []
    for cap in (0, 1, 5):
        agent = SearchAgent(model(), belief, config=replace(config, max_inference_batch=cap), seed=421)
        with _policy_rng(42, (agent,)):
            agent.search(env, np.arange(4, dtype=np.int32))
        reports.append(agent.last_search)
    for actual in reports[1:]:
        for a, b in zip(actual, reports[0], strict=True):
            assert a['selected'] == b['selected']
            assert [c['path'] for c in a['candidates']] == [c['path'] for c in b['candidates']]
            np.testing.assert_allclose([c['q'] for c in a['candidates']],
                                       [c['q'] for c in b['candidates']], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('cache', ['host', 'auto', 'device'])
def test_exact_pre_task_games_and_reports(cache):
    # Frozen using the pre-task implementation, including its chunked actor.
    from gfrl.evaluation import summarize
    case = json.loads((Path(__file__).parent/'fixtures/search_exact_before.json').read_text())
    agent = SearchAgent(model(), baseline(), config=SearchConfig(**case['config'],
        search_cache_device=cache, skip_forced_inference=False, skip_single_candidate=False,
        root_batch_size=0), seed=421)
    env = VecEnv(2, 1, eval_pairs=2, opponent='greedy', eval_seed=98, track_projection=True)
    actions, reports = [], []
    with _policy_rng(42, (agent,)):
        while not env.eval_done():
            rows = np.flatnonzero(env.stage >= 0)
            result = agent(env, rows)
            actions.append(result.tolist())
            reports.extend(agent.last_search if np.any(env.substep[rows] == 0) else [])
            env.step(result)
    assert actions == case['actions']
    assert json.dumps(reports, sort_keys=True) == json.dumps(case['reports'], sort_keys=True)
    summary = json.loads(json.dumps(summarize(env.eval_results())))
    # Existing fields remain exact; game-length diagnostics are additive.
    assert {k: summary[k] for k in case['summary']} == case['summary']
    # Frozen from HEAD parent + original (pre-response) goodfield in an isolated
    # build. Preserve every existing counter; new counters are additive.
    expected_counts = json.loads((Path(__file__).parent/'fixtures/search_counts_before.json').read_text())
    assert {k: agent.counts[k] for k in expected_counts} == expected_counts
    assert agent.counts['skip_response_disabled'] == expected_counts['skip_not_main']
    assert agent.counts['searched_main'] == expected_counts['searched']
    assert agent.counts['searched_response'] == 0


@pytest.mark.parametrize('cap', [0, 5])
@pytest.mark.parametrize('depth', [0, 3])
def test_forced_inference_coupled_world_values(cap, depth):
    # Couple rollout categorical draws by (row, decision, substep), independently
    # of calls/batch sizes. Unused root draws cannot alter the suffix's draws.
    # This is an explicit coupling of the SAME categorical action distributions.
    from gfrl.actor import ActResult
    env = VecEnv(4, 1, 741, track_projection=True)
    reports, candidates = [], []
    inferred = []
    for skip in (False, True):
        agent = SearchAgent(model(), baseline(), config=SearchConfig(worlds=3, candidates=4,
            candidate_samples=12, depth=depth, max_inference_batch=cap,
            skip_forced_inference=skip), seed=421)
        original = agent._act
        root_rows = []
        def coupled(actor, rows, phase):
            if phase != 'rollout':
                return original(actor, rows, phase)
            actor.sample = False
            act = original(actor, rows, phase)
            choices = act.choices.copy()
            for r in rows:
                d, t = int(actor.env.decisions[r]), int(actor.env.substep[r])
                root_rows.append(d == 0)
                rng = np.random.default_rng([91, int(r), d, t])
                probs = np.exp(act.log_probs[r].astype(np.float64))
                choices[r] = rng.choice(21, p=probs/probs.sum())
            return ActResult(choices, act.log_prob, act.value_logits, act.log_probs)
        agent._act = coupled
        with _policy_rng(42, (agent,)):
            agent.search(env, np.arange(4, dtype=np.int32), report_values=True)
        reports.append(agent.last_search)
        candidates.append(agent.last_candidates)
        inferred.append(sum(root_rows))
    assert candidates[0] == candidates[1]  # paths, pi_theta, rho, ordering
    assert reports[0] == reports[1]  # every world leaf, q and search probabilities
    assert inferred[0] > 0 and inferred[1] == 0


@pytest.mark.parametrize('cap', [0, 5])
def test_singleton_validates_paths_and_omits_values(cap):
    from dataclasses import replace
    env = VecEnv(2, 1, 741, track_projection=True)
    config = SearchConfig(worlds=3, candidates=1, candidate_samples=1, depth=3,
                          max_inference_batch=cap)
    results = []
    for skip in (False, True):
        agent = SearchAgent(model(), baseline(), config=replace(config, skip_single_candidate=skip), seed=421)
        with _policy_rng(42, (agent,)):
            results.append(agent.search(env, np.arange(2, dtype=np.int32)))
        if not skip:
            candidates = agent.last_candidates
        else:
            assert agent.last_candidates == candidates
            assert agent.counts['rollout_worlds'] == agent.counts['depth_leaves'] == 0
            assert agent.counts['singleton_roots'] == 2
            assert all(r['probabilities'] == [1.] and r['candidates'][0]['q'] is None
                       and r['rollout_skipped'] == 'single_candidate' for r in agent.last_search)
            # Even a fixed singleton must be rejected if its first choice is illegal.
            invalid = {r: [((20,), (0., 0.))] for r in range(2)}
            assert not env.mask[:, 20].any()
            with _policy_rng(42, (agent,)):
                assert agent.search(env, np.arange(2, dtype=np.int32), fixed_candidates=invalid) == {}
            assert agent.counts['invalid_candidate_worlds'] == 6
    assert results[0] == results[1]


def test_bounded_root_workspaces_and_diagnostics():
    config = SearchConfig(worlds=2, candidates=3, candidate_samples=8, depth=3,
                          root_batch_size=2, skip_forced_inference=True, skip_single_candidate=True)
    saved = []
    for _ in range(2):
        env = VecEnv(8, 1, 741, track_projection=True)
        agent = SearchAgent(model(), baseline(), config=config, seed=421)
        with _policy_rng(42, (agent,)):
            result = agent.search(env, np.arange(8, dtype=np.int32), report_values=True)
        assert len(result) == len(agent.last_candidates) == len(agent.last_search) == 8
        assert agent.counts['max_batch_roots'] == 2
        assert agent.counts['search_batches'] == 4
        assert agent.counts['singleton_roots'] == 0  # report_values requires q
        assert agent._workspaces['rollout'][1].num_envs <= 2*3*2
        assert all(c['world_values'] for r in agent.last_search for c in r['candidates'])
        saved.append(agent.last_search)
    assert saved[0] == saved[1]


def test_cache_budget_falls_back_to_host():
    from gfrl.search import _SearchActor
    env = VecEnv(2, 1, 741, track_projection=True)
    actor = _SearchActor(model(), env, 'cpu', max_batch=1, cache_device='device', cache_mb=0)
    actor.act()
    assert actor.token_cache.device.type == 'cpu'


def test_singleton_danger_pair_has_no_partial_world_rejection():
    from dataclasses import replace
    env = VecEnv(1, 1, track_projection=True)
    _core._test_danger_root(env, True)
    pn = torch.zeros(19, dtype=torch.float64)
    pn[0] = pn[1] = .5
    pk = torch.zeros(297, dtype=torch.float64)
    pk[209] = 1
    belief = Baseline(pn, pk)
    config = SearchConfig(worlds=16, candidates=32, candidate_samples=256, depth=0)
    probe = SearchAgent(model(), belief, config=config, seed=6)
    with _policy_rng(77, (probe,)):
        probe.search(env, np.array([0], np.int32), report_values=True)
    assert all(c['valid_worlds'] == 16 for c in probe.last_search[0]['candidates'])
    candidate = probe.last_search[0]['candidates'][0]
    fixed = {0: [(tuple(candidate['path']), (candidate['log_pi'], candidate['log_rho']))]}
    reports = []
    for skip in (False, True):
        agent = SearchAgent(model(), belief, config=replace(config, skip_single_candidate=skip), seed=6)
        with _policy_rng(77, (agent,)):
            agent.search(env, np.array([0], np.int32), fixed_candidates=fixed)
        reports.append(agent.last_search[0])
        assert agent.counts['invalid_candidate_worlds'] == 0
    assert reports[0]['selected'] == reports[1]['selected']
    assert reports[0]['probabilities'] == reports[1]['probabilities'] == [1.]
    for key in ('path', 'valid_worlds', 'log_pi', 'log_rho'):
        assert reports[0]['candidates'][0][key] == reports[1]['candidates'][0][key]


def test_device_cache_budget_selection(monkeypatch):
    from gfrl.search import _SearchActor
    actor = _SearchActor(model(), VecEnv(2, 1), 'cpu', max_batch=1,
                         cache_device='device', cache_mb=0)
    # Exercise placement decisions without requiring a CUDA device.
    actor.device = torch.device('cuda')
    encoded = torch.empty(1, 7, 4)
    assert actor._cache_location(encoded).type == 'cpu'
    actor.cache_bytes = 1024
    assert actor._cache_location(encoded).type == 'cuda'
    actor.cache_device = 'auto'
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (100, 10000))
    assert actor._cache_location(encoded).type == 'cpu'
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (10000, 10000))
    assert actor._cache_location(encoded).type == 'cuda'
    actor.cache_device = 'host'
    assert actor._cache_location(encoded).type == 'cpu'


def response_scene():
    from gfrl.search import response_roots
    env = VecEnv(16, 1, 741, track_projection=True)
    rng = np.random.default_rng(51)
    for _ in range(1000):
        fresh = np.flatnonzero((env.stage >= 0) & (env.substep == 0)).astype(np.int32)
        roots = fresh[(env.determinize_support(fresh) == 0) & response_roots(env, fresh) & (env.mask[fresh].sum(1) >= 3)]
        if len(roots):
            return env, roots[:1]
        env.step(np.array([rng.choice(np.flatnonzero(m)) for m in env.mask], np.int32))
    pytest.fail('no response fixture')


@pytest.mark.parametrize('belief_kind', ['uniform', 'oracle', 'autoregressive', 'independent'])
@pytest.mark.parametrize('cap', [0, 3])
def test_response_search_observation_paths_and_depth(belief_kind, cap):
    env, roots = response_scene()
    if belief_kind in ('uniform', 'oracle'):
        belief = load_search_belief(belief_kind)
    else:
        cls = Belief if belief_kind == 'autoregressive' else IndependentBelief
        torch.manual_seed(34)
        belief = cls(ModelConfig(d=16, heads=2, depth=1, ff=32), BeliefConfig(depth=1, heads=2)).eval()
    config = SearchConfig(search_response=True, worlds=3, candidates=8, candidate_samples=32,
                          depth=0, max_inference_batch=cap, skip_forced_inference=True)
    agent = SearchAgent(model(), belief, config=config, seed=421)
    with _policy_rng(42, (agent,)):
        rr, assignments, seeds = agent._sample_worlds(env, roots)
        world, status = agent._world_env(env, rr, assignments, seeds, 1)
        assert not status.any() and len(rr) == 3
        for key in OBS_FIELDS:
            np.testing.assert_array_equal(getattr(world, key), getattr(env, key)[rr])
        if belief_kind == 'oracle':
            np.testing.assert_array_equal(world.label_opp_hand, assignments)
        paths = agent.search(env, roots, report_values=True)
    assert set(paths) == set(roots)
    assert agent.counts['searched_main'] == 0 and agent.counts['searched_response'] == 1
    assert agent.counts['invalid_loads'] == 0
    # The ordinary path represents every externally selected response substep.
    assert any(len(c['path']) > 1 for r in agent.last_search for c in r['candidates'])
    stopped = agent._workspaces['rollout'][1]
    assert np.all(stopped.decisions == 1)  # root response is exactly one decision
    assert agent.counts['depth_leaves'] + agent.counts['terminal_leaves'] > 0
    row = int(roots[0])
    for choice in paths[row]:
        assert env.mask[row, choice]
        choices = env.mask.argmax(1).astype(np.int32)
        choices[row] = choice
        env.step(choices)
    assert env.decision_done[row]


def test_response_disabled_preserves_old_reason_and_rng():
    from gfrl.search import search_support
    env, roots = response_scene()
    agent = SearchAgent(model(), baseline(), seed=421)
    state = agent.rng.bit_generator.state
    assert np.all(search_support(env, roots) == 3)
    assert agent.search(env, roots) == {}
    assert agent.rng.bit_generator.state == state
    with _policy_rng(42, (agent,)):
        choices = agent(env, roots)
    assert env.mask[roots, choices[roots]].all()
    assert agent.rng.bit_generator.state == state
    assert agent.counts['skip_response_disabled'] == agent.counts['skip_not_main'] == 1
    assert agent.counts['searched'] == 0


@pytest.mark.parametrize('belief_kind', ['oracle', 'autoregressive'])
def test_response_paired_games_complete(belief_kind):
    if belief_kind == 'oracle':
        belief = load_search_belief('oracle')
    else:
        torch.manual_seed(34)
        belief = Belief(ModelConfig(d=16, heads=2, depth=1, ff=32), BeliefConfig(depth=1, heads=2)).eval()
    agent = SearchAgent(model(), belief, config=SearchConfig(search_response=True, worlds=2,
        candidates=3, candidate_samples=8, depth=3, skip_forced_inference=True,
        skip_single_candidate=True), seed=221)
    opponent = Actor(model(), VecEnv(1, 1, eval_pairs=0), 'cpu')
    result = evaluate_policies(agent, opponent, 3, 4, 1, 98, track_projection=True)
    assert result['games'] == 6
    assert agent.counts['searched_main'] > 0 and agent.counts['searched_response'] > 0
    assert agent.counts['searched'] == agent.counts['searched_main'] + agent.counts['searched_response']
    assert agent.counts['real_mismatches'] == agent.counts['invalid_loads'] == 0
    assert agent.counts['invalid_oracle_assignments'] == 0
    assert agent.statistics()['seconds_per_search_by_root']['response'] > 0


@pytest.mark.parametrize('kind', ['chance', 'fog', 'gamble', 'fog_gamble'])
def test_public_chance_response_search_and_hidden_source_independence(kind):
    from gfrl.search import search_support
    if kind == 'chance':
        card = next(cid for cid, c in _core.play_catalog().items()
                    if c['hit_rate'] and c['category'] == 'weapons')
    else:
        card = 202 if 'gamble' in kind else 6
    envs = [VecEnv(1, 1, track_projection=True) for _ in range(2)]
    for env in envs:
        _core._test_response_root(env, card, 'fog' in kind, 42)
    root = np.array([0], np.int32)
    _core._test_replace_hidden_hand(envs[1], 0, 1234)
    assert search_support(envs[0], root, True).tolist() == [0]
    assert search_support(envs[0], root, False).tolist() == [3]
    for key in OBS_FIELDS:
        np.testing.assert_array_equal(getattr(envs[0], key), getattr(envs[1], key))
    agents = [SearchAgent(model(), baseline(), config=SearchConfig(search_response=True,
              worlds=3, candidates=4, candidate_samples=16, depth=2), seed=421) for _ in range(2)]
    choices = []
    for agent, env in zip(agents, envs):
        with _policy_rng(42, (agent,)):
            choices.append(agent(env, root))
        assert agent.counts['searched_response'] == 1
        assert agent.counts['invalid_loads'] == 0
    np.testing.assert_array_equal(*choices)
    assert agents[0].last_search == agents[1].last_search
    assert agents[0].counts == agents[1].counts
    _core._test_response_root(envs[0], card, 'fog' in kind, 42)
    disabled = SearchAgent(model(), baseline(), seed=421)
    rng = disabled.rng.bit_generator.state
    assert disabled.search(envs[0], root) == {}
    assert disabled.rng.bit_generator.state == rng
    with _policy_rng(42, (disabled,)):
        choices = disabled(envs[0], root)
    assert envs[0].mask[root, choices[root]].all()
    assert disabled.rng.bit_generator.state == rng
    assert disabled.counts['skip_not_main'] == disabled.counts['skip_response_disabled'] == 1


@pytest.mark.parametrize('response', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_own_dream_search_hidden_actual_independence(response, enabled):
    from gfrl import SearchEnv
    from gfrl.search import search_support
    envs = [VecEnv(1, 1, track_projection=True) for _ in range(3)]
    # Include identity disguise: eligibility and sampling cannot branch on truth.
    for env, actual in zip(envs, (6, 8, 7)):
        _core._test_dream_root(env, response, actual)
    root = np.array([0], np.int32)
    for env in envs:
        assert search_support(env, root, enabled).tolist() == ([3] if response and not enabled else [0])
        for key in (*OBS_FIELDS, 'mask'):
            np.testing.assert_array_equal(getattr(env, key), getattr(envs[0], key))
    # Same seeds give the same complete root observations in either seat and
    # same future transitions, after replacing every hidden own actual.
    worlds = [SearchEnv(1, 1, max_decisions=4) for _ in envs]
    assignments = np.zeros((1, 297), np.int32)
    assignments[0, 7] = 1
    for seed in range(32):
        for world, env in zip(worlds, envs):
            assert world.load_worlds(env, root, assignments, np.array([seed], np.uint64)).tolist() == [0]
            for key in (*OBS_FIELDS, 'mask'):
                np.testing.assert_array_equal(getattr(world, key), getattr(env, key))
        for seat in (0, 1):
            leaves = [{k: v.copy() for k, v in w.observe(root, seat).items()} for w in worlds]
            for leaf in leaves[1:]:
                for key in OBS_FIELDS:
                    np.testing.assert_array_equal(leaf[key], leaves[0][key])
    agents = [SearchAgent(model(), baseline(), config=SearchConfig(search_response=enabled,
              worlds=4, candidates=4, candidate_samples=16, depth=3), seed=421) for _ in envs]
    choices = []
    for agent, env in zip(agents, envs):
        with _policy_rng(42, (agent,)):
            choices.append(agent(env, root))
        assert agent.counts['invalid_loads'] == agent.counts['real_mismatches'] == 0
        assert agent.counts['searched_response' if response else 'searched_main'] == (0 if response and not enabled else 1)
    for i in (1, 2):
        np.testing.assert_array_equal(choices[i], choices[0])
        assert agents[i].last_search == agents[0].last_search
        assert agents[i].counts == agents[0].counts


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('kind', ['main', 'response', 'percentage', 'fog', 'gamble', 'fog_gamble'])
def test_pre_dream_supported_roots_exact_fixture(enabled, kind):
    from gfrl.search import search_support
    env = VecEnv(1, 1, 741, track_projection=True)
    if kind != 'main':
        percentage = next(cid for cid, c in _core.play_catalog().items()
                          if c['hit_rate'] and c['category'] == 'weapons')
        card = percentage if kind == 'percentage' else 202 if 'gamble' in kind else 6
        _core._test_response_root(env, card, 'fog' in kind, 42)
    roots = np.array([0], np.int32)
    support = search_support(env, roots, enabled).tolist()
    agent = SearchAgent(model(), load_search_belief('uniform'), seed=421,
                        config=SearchConfig(search_response=enabled, worlds=3,
                                            candidates=4, candidate_samples=16, depth=2))
    with _policy_rng(42, (agent,)):
        choices = agent(env, roots)
    actual = dict(support=support, choices=choices.tolist(),
                  search=agent.last_search, counts=dict(agent.counts))
    expected = json.loads((Path(__file__).parent/'fixtures/search_supported_before.json').read_text())
    assert actual == expected[f'{enabled}:{kind}']
