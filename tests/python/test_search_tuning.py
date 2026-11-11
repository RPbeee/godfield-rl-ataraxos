"""Search-time belief transforms, offline isolation, grid and session contracts."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from gfrl import VecEnv
from gfrl.belief import Belief, BeliefConfig
from gfrl.belief_metrics import Baseline
from gfrl.model import Model, ModelConfig, OBS_FIELDS
from gfrl.search import SearchAgent, SearchConfig, load_search_belief
from gfrl.search_belief import TemperedBelief, MixedBelief, HybridBelief, wrap_belief
from gfrl.official_search import add_search_arguments, search_config, search_agent

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import eval_search
import sweep_search
from search_diagnostics import DiagnosticSearchAgent


class ToyAR(Belief):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def memory(self, obs):
        x = torch.zeros(len(obs['obs_type']), 1, 1)
        return x[:, 0], x, torch.zeros(len(x), 1, dtype=torch.bool)

    def count_head(self, cls):
        logits = torch.full((len(cls), 19), -torch.inf)
        logits[:, 1] = 0
        return logits

    def decode(self, memory, padding, inputs, counts):
        logits = torch.full((*inputs.shape, 296), -torch.inf)
        logits[..., 294] = 0
        logits[..., 295] = np.log(3)
        return logits.log_softmax(-1)


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('cap', [0, 2])
def test_neutral_exactness_and_rng(tmp_path, cap):
    torch.manual_seed(25)
    config = ModelConfig(d=16, heads=2, depth=1, ff=32)
    ar = Belief(config, BeliefConfig(depth=1, heads=2)).eval()
    path = tmp_path / 'ar.pt'
    torch.save(dict(model=ar.state_dict(), model_config=asdict(config),
                    belief_config=asdict(ar.belief_config)), path)
    default = load_search_belief(path)
    neutral = load_search_belief(path, temperature=1, mix=0, hybrid=False, count_temperature=1)
    assert type(default) is type(neutral) is Belief
    assert wrap_belief(ar, 1, 0) is ar
    env = VecEnv(3, 1, 17)
    model = Model(config)
    agents = [SearchAgent(model, b, config=SearchConfig(worlds=2, depth=0,
              max_inference_batch=cap), seed=39) for b in (default, neutral)]
    outputs, states = [], []
    for agent in agents:
        torch.manual_seed(71)
        with torch.inference_mode():
            outputs.append(agent._sample_worlds(env, np.arange(3)))
        states.append((torch.get_rng_state(), agent.rng.bit_generator.state))
    for left, right in zip(*outputs):
        np.testing.assert_array_equal(left, right)
    assert torch.equal(states[0][0], states[1][0])
    assert states[0][1] == states[1][1]


def test_temperature_factorization_and_samples():
    ar = ToyAR()
    obs = dict(obs_type=torch.zeros(1, 1))
    samples = []
    for temperature in (0.5, 1., 2.):
        model = wrap_belief(ar, temperature)
        hand = model.sample(obs, 12000, torch.Generator().manual_seed(7))[0]
        frequency = hand[:, 296].float().mean().item()
        expected = 3**(1/temperature) / (1 + 3**(1/temperature))
        assert frequency == pytest.approx(expected, abs=0.015)
        labels = torch.zeros(1, 297, dtype=torch.long)
        labels[0, 296] = 1
        assert model.log_prob(obs, labels)[0].exp().item() == pytest.approx(expected, abs=1e-6)
        samples.append(frequency)
    assert samples[0] > samples[1] > samples[2]


def test_mix_world_selection_and_repeatability():
    pn = torch.zeros(19, dtype=torch.float64); pn[1] = 1
    pk = torch.zeros(297, dtype=torch.float64); pk[1] = 1
    mixed = wrap_belief(ToyAR(), mix=.5, baseline=Baseline(pn, pk))
    rng = np.random.default_rng(17)
    chosen = mixed.choose(rng, 10000)
    assert chosen.mean() == pytest.approx(.5, abs=.02)
    np.testing.assert_array_equal(chosen, mixed.choose(np.random.default_rng(17), 10000))
    # Exercise the actual world sampler; disjoint card support identifies source.
    class Source:
        obs_type = np.zeros((10000, 1), dtype=np.int32)
        def unused_assignment(self, row, hand):
            return hand
    src = Source()
    for field in OBS_FIELDS:
        setattr(src, field, src.obs_type)
    agent = SearchAgent(Model(ModelConfig(d=16, heads=2, depth=1, ff=32)), mixed,
                        config=SearchConfig(worlds=1), seed=17)
    rows, hands, seeds = agent._sample_worlds(src, np.arange(10000))
    np.testing.assert_array_equal(hands[:, 1] == 1, chosen)
    assert (hands.sum(1) == 1).all()


@pytest.mark.parametrize('advisor', [False, True])
def test_official_refuses_true_hand_diagnostic(advisor):
    parser = argparse.ArgumentParser()
    add_search_arguments(parser, advisor=advisor)
    with pytest.raises(SystemExit):
        parser.parse_args(['--diagnostic-true-hand-nll'])
    args = parser.parse_args([])
    args.greedy = False
    args.diagnostic_true_hand_nll = True
    with pytest.raises(AssertionError, match='forbidden'):
        search_config(args)
    with pytest.raises(AssertionError, match='forbidden'):
        search_agent(None, args)


def test_named_and_cartesian_grid_resume(tmp_path):
    parser = eval_search.build_parser(sweep=True)
    parser.add_argument('--grid', nargs='+', type=sweep_search.grid_pair)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args(['--checkpoint', 'policy.pt', '--belief', 'ar.pt',
        '--out-dir', str(tmp_path), '--worlds', '1', '--depth', '0',
        '--alpha', '.002', '.02', '--beta', '.1', '.3', '--leaf-view', 'actor', 'searcher',
        '--belief-temperature', '.7', '1', '--belief-mix', '0', '.5'])
    settings, keys = sweep_search.make_settings(args)
    assert len(settings) == 32
    assert len({s.out for s in settings}) == 32
    assert set(keys) == {'alpha', 'beta', 'leaf_view', 'belief_temperature', 'belief_mix'}
    for setting in settings:
        for key in keys:
            assert f'{key.replace("_", "-")}-{getattr(setting, key)}' in setting.out.name
    args.grid = [sweep_search.grid_pair('alpha=.002,beta=.3,leaf_view=actor,belief_temperature=1,belief_mix=0')]
    settings, _ = sweep_search.make_settings(args)
    setting = settings[0]
    report = dict(config=vars(setting).copy(), games=2*setting.pairs, win_rate=.5,
                  wilson_interval=[.4,.6], bootstrap_interval=[.4,.6], decision_timing={},
                  search={}, sweep_inputs={})
    sweep_search.atomic_json(setting.out, {**report, 'config': {k: str(v) if isinstance(v, Path) else v for k,v in report['config'].items()}})
    assert sweep_search.completed_report(setting.out, setting, {}) is not None
    setting.beta = .8
    with pytest.raises(ValueError, match='mismatch'):
        sweep_search.completed_report(setting.out, setting, {})


@pytest.mark.parametrize('temperature,mix', [(0,0), (float('nan'),0), (1,-.1), (1,1.1)])
def test_invalid_variants(temperature, mix):
    with pytest.raises(ValueError):
        wrap_belief(ToyAR(), temperature, mix)


def test_diagnostic_scores_unused_hand_without_rng_changes(monkeypatch):
    import search_diagnostics
    ar = ToyAR()
    pn = torch.zeros(19, dtype=torch.float64); pn[1] = 1
    pk = torch.zeros(297, dtype=torch.float64); pk[295] = .75; pk[296] = .25
    agent = DiagnosticSearchAgent(Model(ModelConfig(d=16, heads=2, depth=1, ff=32)), ar)
    agent.diagnostic_models = dict(ar_T1=ar, ar_tempered=wrap_belief(ar, .5), b1=Baseline(pn, pk),
        hybrid=wrap_belief(ar, hybrid=True, baseline=Baseline(pn, pk)))
    agent.diagnostic_values = {k: [[], []] for k in agent.diagnostic_models}
    agent.diagnostic_cap = 1
    class Source:
        label_opp_hand = np.zeros((2, 297), dtype=np.int32)
        def unused_assignment(self, row, hand):
            assert hand[295] == hand[296] == 1  # public used 295 is removed
            result = hand.copy(); result[295] = 0
            return result
    src = Source()
    src.label_opp_hand[:, 295:297] = 1
    for field in OBS_FIELDS:
        setattr(src, field, np.zeros((2, 1), dtype=np.int32))
    monkeypatch.setattr(search_diagnostics, 'search_support', lambda *a: np.zeros(2))
    monkeypatch.setattr(SearchAgent, 'search', lambda *a, **kw: {})
    state = torch.get_rng_state().clone()
    np_state = agent.rng.bit_generator.state
    agent.search(src, [0, 1])
    assert torch.equal(state, torch.get_rng_state())
    assert np_state == agent.rng.bit_generator.state
    report = agent.diagnostic_report()
    assert report['positions'] == 2 and report['privileged']
    for key, probability in [('ar_T1', .75), ('ar_tempered', .9), ('b1', .25), ('hybrid', .25)]:
        result = report['models'][key]
        assert result['nll']['mean'] == pytest.approx(-np.log(probability), abs=1e-6)
        assert result['multiset_nll'] == result['nll']['mean']
        assert result['hand_size_accuracy']['mean'] == 1
        assert len(result['nll']['ci95']) == 2


@pytest.mark.parametrize('module_name', ['gfrl.official_bot', 'gfrl.official_advisor'])
def test_official_entrypoints_reject_diagnostic(module_name, capsys):
    from importlib import import_module
    required = (['--checkpoint', 'unused', '--out-dir', '/tmp/unused'] if module_name.endswith('official_bot')
                else ['--checkpoint', 'unused', '--input', '/tmp/unused', '--viewer-id', '1'])
    with pytest.raises(SystemExit) as error:
        import_module(module_name).main([*required, '--diagnostic-true-hand-nll'])
    assert error.value.code == 2
    assert 'unrecognized arguments: --diagnostic-true-hand-nll' in capsys.readouterr().err


@pytest.mark.parametrize('cap', [0, 137])
@pytest.mark.parametrize('temperature', [1., 1.5])
def test_hybrid_count_composition_and_public_used(cap, temperature):
    class CountsAR(ToyAR):
        def count_head(self, cls):
            logits = torch.full((len(cls), 19), -torch.inf)
            logits[:, 0] = 0
            logits[:, 2] = np.log(3)
            return logits
    pn = torch.zeros(19, dtype=torch.float64); pn[1] = 1
    pk = torch.zeros(297, dtype=torch.float64); pk[295] = .8; pk[296] = .2
    baseline = Baseline(pn, pk)
    hybrid = wrap_belief(CountsAR(), hybrid=True, baseline=baseline,
                         count_temperature=temperature)
    class Source:
        def obs_schema(self):
            return dict(positions=dict(opponent_used_start=39, history_start=55))
        def unused_assignment(self, row, hand):
            result = hand.copy()
            result[295] -= 1
            assert result[295] >= 0
            return result
    src = Source()
    size = 12000
    for field in OBS_FIELDS:
        setattr(src, field, np.zeros((size, 55), dtype=np.int32))
    src.obs_cards = np.zeros((size, 55, 8), dtype=np.int32)
    src.obs_cards[:, 39, 0] = 295  # public used card, restored for world loading
    agent = SearchAgent(Model(ModelConfig(d=16, heads=2, depth=1, ff=32)), hybrid,
        config=SearchConfig(worlds=1, max_inference_batch=cap), seed=31)
    torch.manual_seed(37)
    rows, hands, _ = agent._sample_worlds(src, np.arange(size))
    assert len(rows) == size
    hands[:, 295] -= 1
    counts = hands.sum(1)
    expected = 3**(1/temperature) / (1+3**(1/temperature))
    assert (counts == 2).mean() == pytest.approx(expected, abs=.015)
    assert set(counts) == {0, 2}
    nonempty = hands[counts == 2]
    assert nonempty[:, 296].mean()/2 == pytest.approx(.2, abs=.015)
    assert (nonempty[:, 296] == 2).mean() == pytest.approx(.04, abs=.01)
    labels = torch.zeros(2, 297, dtype=torch.long)
    labels[1, 295:297] = 1
    obs = dict(obs_type=torch.zeros(2, 1))
    lp, count_lp, composition_lp = hybrid.log_prob(obs, labels)
    assert count_lp.exp().tolist() == pytest.approx([1-expected, expected])
    assert composition_lp.exp().tolist() == pytest.approx([1, 2*.8*.2])
    assert lp.exp().tolist() == pytest.approx([1-expected, expected*2*.8*.2])


def test_hybrid_diagnostic_registration(monkeypatch):
    import search_diagnostics
    ar = ToyAR()
    baseline = Baseline(torch.ones(19)/19, torch.ones(297)/297)
    monkeypatch.setattr(search_diagnostics, 'load_search_belief', lambda *a, **kw: ar)
    monkeypatch.setattr(search_diagnostics.torch, 'load', lambda *a, **kw: {})
    monkeypatch.setattr(search_diagnostics, 'fit_baselines', lambda data: dict(B1=baseline))
    agent = DiagnosticSearchAgent(Model(ModelConfig(d=16, heads=2, depth=1, ff=32)), ar)
    agent.enable_diagnostic(argparse.Namespace(fit='fit.pt', belief='ar.pt', device='cpu',
        max_inference_batch=2, belief_count_temperature=1.5))
    assert isinstance(agent.diagnostic_models['hybrid'], HybridBelief)
    assert agent.diagnostic_models['hybrid'].count_temperature == 1.5
    assert 'hybrid' in agent.diagnostic_report()['models']


@pytest.mark.parametrize('advisor', [False, True])
def test_official_hybrid_options(advisor):
    parser = argparse.ArgumentParser()
    add_search_arguments(parser, advisor=advisor)
    args = parser.parse_args(['--belief-hybrid', '--belief-count-temperature', '1.5'])
    assert args.belief_hybrid == 1 and args.belief_count_temperature == 1.5


def test_hybrid_sweep_grid(tmp_path):
    parser = eval_search.build_parser(sweep=True)
    parser.add_argument('--grid', nargs='+', type=sweep_search.grid_pair)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args(['--checkpoint', 'policy.pt', '--belief', 'ar.pt',
        '--out-dir', str(tmp_path), '--grid',
        'belief_hybrid=1,belief_count_temperature=1',
        'belief_hybrid=1,belief_count_temperature=1.5', 'belief_hybrid=0',
        'belief_hybrid=0,belief=b1'])
    settings, keys = sweep_search.make_settings(args)
    assert len(settings) == len({s.out for s in settings}) == 4
    assert 'belief_hybrid' in keys and 'belief_count_temperature' in keys
    assert [s.belief_hybrid for s in settings] == [1, 1, 0, 0]
    for kwargs in (dict(hybrid=2), dict(hybrid=True, mix=.5),
                   dict(hybrid=True, temperature=2), dict(count_temperature=1.5),
                   dict(hybrid=True, count_temperature=0)):
        with pytest.raises(ValueError):
            wrap_belief(ToyAR(), **kwargs)
