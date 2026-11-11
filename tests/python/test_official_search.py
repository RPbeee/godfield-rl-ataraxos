"""Public-wire belief/search isolation, response replay and shared CLI contracts."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gfrl import _core
from gfrl.actor import PREFIX_FIELDS
from gfrl.belief import Belief, BeliefConfig
from gfrl.evaluation import _policy_rng
from gfrl.model import Model, ModelConfig, OBS_FIELDS
from gfrl.official import OfficialDecision, OfficialSession
from gfrl.official_advisor import advise
from gfrl.official_search import add_search_arguments, root_matches, search_config
from gfrl.search import SearchAgent, SearchConfig, load_search_belief

DATA = Path(__file__).resolve().parents[2]/'third_party/goodfield/data/official-v2'


def games(response=False, model=19):
    first = json.loads((DATA/'wire-game-updates-v73.json').read_text())['updates'][0]['game']
    if not response:
        return [first]
    first['players'][0]['items'].append(dict(id=990, modelId=190))
    first['attackTurnPlayerId'] = 2
    first['events'] = [dict(action='startGame'), dict(action='advanceGF', playerId=2)]
    pending = deepcopy(first)
    pending.update(updateCount=2, attacks=[dict(playerId=2, targetPlayerId=1, itemModelIds=[model])],
        events=[dict(action='useAttackItems', items=[dict(id=900, modelId=model)]),
                dict(action='setTargetPlayer', playerId=1)])
    if _core.play_catalog()[model]['category'] == 'miracles':
        pending['players'][1]['items'].append(dict(id=900, modelId=model, used=True))
        pending['players'][1]['mp'] -= _core._official_magic_cost([model])
    return [first, pending]


def result_for(sequence):
    session = OfficialSession(1)
    for game in sequence:
        result = session.ingest(deepcopy(game))
    assert result['status'] == 'decision', result
    return result


@pytest.fixture
def models(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(11)
    config = ModelConfig(d=16, heads=2, depth=1, ff=32)
    model = Model(config).eval()
    belief = Belief(config, BeliefConfig(depth=1, heads=2)).eval()
    path = tmp_path/'ar.pt'
    torch.save(dict(model=belief.state_dict(), model_type='autoregressive',
        model_config=asdict(config), belief_config=asdict(belief.belief_config)), path)
    return model, path


@pytest.mark.parametrize('response', [False, True, 'recorded'])
@pytest.mark.parametrize('belief_kind', ['uniform', 'ar'])
def test_wire_truth_cannot_reach_belief_or_search(models, response, belief_kind):
    model, ar_path = models
    sequence = (json.loads((DATA/'wire-game-updates-v73.json').read_text())['updates'][:2]
                if response == 'recorded' else None)
    sequence = [u['game'] for u in sequence] if sequence else games(response)
    altered = deepcopy(sequence)
    for game in altered:
        other = game['players'][1]
        other['items'] = [x for x in other['items'] if x.get('used')]+[
            dict(id=700+i, modelId=m, fakeModelId=6) for i, m in enumerate((209, 94, 232, 207))]
        other['name'] = 'synthetic-private-marker'
    outcomes = []; belief_inputs = []; samples = []
    for source in (sequence, altered):
        result = result_for(source)
        root = _core._official_search_root(result['public'], result['env'].events, 64)
        assert root_matches(root, result['env'])
        belief = load_search_belief(str(ar_path) if belief_kind == 'ar' else 'uniform')
        seen = []; generated = []
        if belief_kind == 'ar':
            original = belief.sample
            def spy(obs, n):
                assert set(obs) == set(OBS_FIELDS)
                # Every belief row equals the policy's immutable public observation.
                for key in OBS_FIELDS:
                    expected = np.repeat(getattr(result['env'], key), len(obs[key]), axis=0)
                    np.testing.assert_array_equal(obs[key].cpu().numpy(), expected)
                seen.append({k: v.clone() for k, v in obs.items()})
                sample = original(obs, n); generated.append(sample.clone())
                return sample
            belief.sample = spy
        agent = SearchAgent(model, belief, config=SearchConfig(worlds=2, depth=1,
            candidates=3, candidate_samples=8, search_response=True, select='argmax'), seed=17)
        agent.actor.sample = False
        with _policy_rng(31, (agent,)):
            advice = advise(result, agent, search=True)
        assert advice['search_status'] == ('response_public_worlds' if response else 'main_public_worlds')
        assert advice['search_worlds'] == 2 and advice['search_candidates'] > 0
        assert advice['pi_search_top'] and advice['pi_theta_top']
        outcomes.append((advice['action'], agent.last_search)); belief_inputs.append(seen); samples.append(generated)
        assert 'synthetic-private-marker' not in json.dumps(advice)
        assert not any('world_values' in c for r in agent.last_search for c in r['candidates'])
    assert outcomes[0] == outcomes[1]
    assert len(samples[0]) == len(samples[1])
    for left, right in zip(samples[0], samples[1]):
        assert torch.equal(left, right)
    for left, right in zip(belief_inputs[0], belief_inputs[1]):
        assert all(torch.equal(left[k], right[k]) for k in OBS_FIELDS)


@pytest.mark.parametrize('model', [19, 107, 202])
def test_response_public_replay_and_every_prefix(model):
    # Percentage launch: pending existence publicly conditions hit; 202 has a
    # public signed amount. Root construction must never inspect hidden hands.
    result = result_for(games(True, model))
    root = _core._official_search_root(result['public'], result['env'].events, 64)
    env = result['env']
    while not env.complete:
        assert root_matches(root, env)
        choice = np.array([np.flatnonzero(env.mask[0])[0]], np.int32)
        env.step(choice)
        if not env.complete:
            root.step(choice)


@pytest.mark.parametrize('gap', ['missing', 'skipped_update', 'redirect', 'changed_hand', 'special'])
def test_unprovable_response_falls_back(models, gap):
    sequence = games(True)
    if gap == 'missing': sequence = sequence[1:]
    elif gap == 'skipped_update': sequence[1]['updateCount'] = 4
    elif gap == 'redirect': sequence[1]['events'].append(dict(action='reflect'))
    elif gap == 'changed_hand': sequence[1]['players'][0]['items'].pop()
    elif gap == 'special': sequence[1]['events'].append(dict(action='nextAttack',
        attack=dict(playerId=2, itemModelIds=[181])))
    result = result_for(sequence)
    agent = SearchAgent(models[0], load_search_belief('uniform'), config=SearchConfig(search_response=True))
    advice = advise(result, agent, search=True)
    assert advice['search_status'].endswith('_policy_fallback')
    assert advice['search_worlds'] == advice['search_candidates'] == 0
    assert advice['search_seconds'] == 0


def test_mismatch_never_enters_search(models, monkeypatch):
    result = result_for(games())
    real_root = _core._official_search_root(result['public'], [], 64)
    # A pruned public cursor must not be searched with the original root mask.
    changed = OfficialDecision(result['public'], excluded_kinds=['discard'])
    result['env'] = changed
    monkeypatch.setattr(_core, '_official_search_root', lambda *args: real_root)
    agent = SearchAgent(models[0], load_search_belief('uniform'))
    advice = advise(result, agent, search=True)
    assert advice['search_status'] == 'public_root_mismatch_policy_fallback'
    assert not agent.counts['searched']


def test_cli_defaults_and_validation():
    for advisor in (False, True):
        parser = argparse.ArgumentParser(); add_search_arguments(parser, advisor=advisor)
        args = parser.parse_args([]); args.greedy = False
        config = search_config(args)
        assert args.belief == 'uniform' and args.device == 'cpu' and args.threads == 1
        assert config == (SearchConfig(depth=3, candidates=8) if advisor else SearchConfig())
        configured = parser.parse_args(['--belief', 'path.pt', '--fit', 'fit.pt', '--worlds', '8',
            '--depth', '1', '--candidates', '4', '--candidate-samples', '12', '--alpha', '.01',
            '--beta', '.03', '--leaf-view', 'searcher', '--search-response', '--device', 'mps', '--threads', '2'])
        configured.greedy = True
        c = search_config(configured)
        assert (c.worlds, c.depth, c.candidates, c.candidate_samples, c.select) == (8,1,4,12,'argmax')
        assert c.search_response and c.leaf_view == 'searcher'
        for key, invalid in [('belief','oracle'), ('threads',0), ('alpha',float('nan')), ('worlds',0)]:
            bad = deepcopy(args); setattr(bad, key, invalid)
            with pytest.raises(ValueError): search_config(bad)


def test_reflection_only_public_journal(models):
    session = OfficialSession(1)
    for game in games(True):
        result = session.ingest(deepcopy(game))
    first = deepcopy(result['public'])
    assert '_search_launch' in first
    first.update(update=3, actor=2, events=[dict(action='useDefenseItems', faces=[190], reveals=[]),
                                           dict(action='reflect')])
    first['pending'].update(source=1, target=2, reflections=1)
    # 190 is a consumed tool; only the known own public copy disappears.
    first['hand'] = [h for h in first['hand'] if h['face'] != 190]
    assert session.ingest_public(first)['status'] == 'waiting'
    second = deepcopy(first)
    second.update(update=4, actor=1)
    second['pending'].update(source=2, target=1, reflections=2)
    result = session.ingest_public(second)
    assert result['status'] == 'decision'
    root = _core._official_search_root(result['public'], result['env'].events, 64)
    assert root_matches(root, result['env'])
    assert root.determinize_support(np.array([0], np.int32))[0] == 0
    agent = SearchAgent(models[0], load_search_belief('uniform'), config=SearchConfig(
        worlds=2, candidates=3, candidate_samples=8, depth=1, search_response=True), seed=17)
    advice = advise(result, agent, search=True)
    assert advice['search_status'] == 'response_public_worlds'


@pytest.mark.parametrize('kind', ['uniform', 'ar'])
def test_recorded_response_without_launch_uses_policy(models, kind):
    data = json.loads((DATA/'wire-game-updates-v64.json').read_text())
    result = OfficialSession(1).ingest(data['updates'][0]['game'])
    agent = SearchAgent(models[0], load_search_belief(str(models[1]) if kind == 'ar' else 'uniform'),
                        config=SearchConfig(search_response=True))
    advice = advise(result, agent, search=True)
    assert advice['search_status'] == 'response_public_journal_unavailable_policy_fallback'
    assert advice['action'] and advice['search_seconds'] == 0


def test_prefix_mismatch_reencodes_policy_without_executing_root(models, monkeypatch):
    import gfrl.official_advisor as advisor
    result = result_for(games())
    original = advisor.root_matches
    calls = []
    def mismatch(root, env):
        calls.append(int(env.substep[0]))
        return original(root, env) and env.substep[0] == 0
    monkeypatch.setattr(advisor, 'root_matches', mismatch)
    agent = SearchAgent(models[0], load_search_belief('uniform'), config=SearchConfig(
        worlds=2, candidates=3, candidate_samples=8, depth=0), seed=17)
    advice = advise(result, agent, search=True)
    assert max(calls) > 0
    assert advice['search_status'] == 'public_prefix_mismatch_policy_fallback'
    assert advice['action']
