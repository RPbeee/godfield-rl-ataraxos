"""Public-wire belief/search isolation, response roots and shared CLI contracts."""
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
BATCHED = Path(__file__).with_name('fixtures')/'official_response_batched.json'


def batched_games(models=(55,), *, atk=None):
    # Entirely synthetic receipt: our main command, its resolution/refill, GF,
    # then the enemy launch arrive together. No enemy main boundary is received.
    sequence = json.loads(BATCHED.read_text())
    pending = sequence[-1]
    pending['attacks'][0]['itemModelIds'] = list(models)
    if atk is not None:
        pending['attacks'][0]['atk'] = atk
    pending['events'][-2]['items'] = [dict(id=900+i, modelId=m) for i, m in enumerate(models)]
    for m in models:
        if _core.play_catalog()[m]['category'] == 'miracles' and m != 218:
            pending['players'][1]['items'].append(dict(id=900+m, modelId=m, used=True))
    cost = _core._official_magic_cost(list(models))
    pending['players'][1]['mp'] -= cost
    if models[0] == 58:
        pending['players'][1]['mp'] = 0
    return sequence


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


@pytest.mark.parametrize('response', [False, True, 'recorded', 'batched'])
@pytest.mark.parametrize('belief_kind', ['uniform', 'ar'])
@pytest.mark.parametrize('rollouts', [1, 4])
def test_wire_truth_cannot_reach_belief_or_search(models, response, belief_kind, rollouts):
    model, ar_path = models
    sequence = (json.loads((DATA/'wire-game-updates-v73.json').read_text())['updates'][:2]
                if response == 'recorded' else None)
    sequence = ([u['game'] for u in sequence] if sequence else
                batched_games() if response == 'batched' else games(response))
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
            candidates=3, candidate_samples=8, search_response=True, select='argmax',
            rollouts_per_world=rollouts), seed=17)
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


@pytest.mark.parametrize('gap', ['redirect', 'special', 'missing_launch', 'later_hit'])
def test_unprovable_response_falls_back(models, gap):
    sequence = games(True)
    if gap == 'redirect': sequence[1]['events'].append(dict(action='reflect'))
    elif gap == 'special': sequence[1]['events'].append(dict(action='nextAttack',
        attack=dict(playerId=2, itemModelIds=[181])))
    elif gap == 'missing_launch': sequence = [dict(sequence[1], events=[])]
    elif gap == 'later_hit': sequence[1]['events'].append(dict(action='useDefenseItems', items=[]))
    result = result_for(sequence)
    agent = SearchAgent(models[0], load_search_belief('uniform'), config=SearchConfig(search_response=True))
    advice = advise(result, agent, search=True)
    assert advice['search_status'].endswith('_policy_fallback')
    assert advice['search_worlds'] == advice['search_candidates'] == 0
    assert advice['search_seconds'] == 0


@pytest.mark.parametrize('gap', ['missing', 'skipped_update', 'changed_hand'])
def test_current_launch_does_not_require_previous_boundary(models, gap):
    sequence = games(True)
    if gap == 'missing': sequence = sequence[1:]
    elif gap == 'skipped_update': sequence[1]['updateCount'] = 4
    elif gap == 'changed_hand': sequence[1]['players'][0]['items'].pop()
    result = result_for(sequence)
    assert '_search_launch' not in result['public']
    agent = SearchAgent(models[0], load_search_belief('uniform'), config=SearchConfig(
        worlds=2, depth=1, candidates=3, candidate_samples=8, search_response=True))
    advice = advise(result, agent, search=True)
    assert advice['search_status'] == 'response_public_worlds'
    assert advice['search_worlds'] == 2 and advice['search_candidates'] > 0


@pytest.mark.parametrize('cards,atk', [((55,), None), ((107,), None), ((218,), None),
    ((55, 232), None), ((202,), -10), ((205,), None), ((206,), None), ((58,), 40)])
def test_batched_partial_history_response_search(models, cards, atk):
    sequence = batched_games(cards, atk=atk)
    session = OfficialSession(1)
    previous = session.ingest(sequence[0])
    assert previous['status'] == 'decision' and previous['public']['phase'] == 0
    assert previous['public']['actor'] == previous['public']['viewer']
    assert not previous['history_complete']
    result = session.ingest(sequence[1])
    assert result['status'] == 'decision' and not result['history_complete']
    assert '_search_launch' not in result['public']
    # A changed hand, GF, HP and partial ledger must all come from the current
    # public pane, while the old (partially observed) history remains unchanged.
    assert result['public']['hand'] != previous['public']['hand']
    env = result['env']
    root = _core._official_search_root(result['public'], env.events, env.history_len)
    assert root_matches(root, env)
    agent = SearchAgent(models[0], load_search_belief('uniform'), config=SearchConfig(
        worlds=2, depth=1, candidates=3, candidate_samples=8, search_response=True), seed=17)
    advice = advise(result, agent, search=True)
    assert advice['search_status'] == 'response_public_worlds'
    assert advice['search_worlds'] == 2 and advice['search_candidates'] > 0
    assert not advice['history_complete']
    assert 'partial_history_and_decision_count' in advice['warnings']


@pytest.mark.parametrize('last', [False, True])
def test_current_response_observation_and_prefixes(last):
    result = result_for(batched_games())
    env = result['env']
    root = _core._official_search_root(result['public'], env.events, env.history_len)
    while not env.complete:
        assert root_matches(root, env)
        choice = np.array([np.flatnonzero(env.mask[0])[-1 if last else 0]], np.int32)
        env.step(choice)
        if not env.complete:
            root.step(choice)


def test_current_response_default_still_uses_policy(models):
    agent = SearchAgent(models[0], load_search_belief('uniform'))
    advice = advise(result_for(batched_games()), agent, search=True)
    assert advice['search_status'] == 'response_search_disabled_policy_fallback'
    assert not agent.counts['searched']


@pytest.mark.parametrize('invalid', ['power', 'mp', 'used', 'multiple_frames'])
def test_current_response_mismatch_never_searches(models, invalid):
    sequence = batched_games((218,))
    if invalid == 'power': sequence[-1]['attacks'][0]['atk'] = 99
    elif invalid == 'mp': sequence[-1]['players'][1]['mp'] = 99
    elif invalid == 'used': sequence[-1]['players'][1]['items'] = []
    elif invalid == 'multiple_frames': sequence[-1]['attacks'] *= 2
    agent = SearchAgent(models[0], load_search_belief('uniform'), config=SearchConfig(search_response=True))
    advice = advise(result_for(sequence), agent, search=True)
    assert advice['search_status'] in (
        'response_current_launch_mismatch_policy_fallback', 'public_root_mismatch_policy_fallback',
        'response_public_journal_unavailable_policy_fallback')
    assert not agent.counts['searched']
    assert advice['search_seconds'] == 0


@pytest.mark.parametrize('response', [False, True])
def test_mismatch_never_enters_search(models, monkeypatch, response):
    result = result_for(batched_games() if response else games())
    real_root = _core._official_search_root(result['public'], result['env'].events, 64)
    # A pruned public cursor must not be searched with the original root mask.
    changed = OfficialDecision(result['public'], result['env'].events,
        excluded_ids=[result['public']['hand'][0]['id']] if response else [],
        excluded_kinds=[] if response else ['discard'])
    result['env'] = changed
    monkeypatch.setattr(_core, '_official_search_root', lambda *args: real_root)
    agent = SearchAgent(models[0], load_search_belief('uniform'), config=SearchConfig(search_response=True))
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
            '--depth', '1', '--rollouts-per-world', '4', '--rollout-crn',
            '--candidates', '4', '--candidate-samples', '12', '--alpha', '.01',
            '--beta', '.03', '--leaf-view', 'searcher', '--search-response', '--device', 'mps', '--threads', '2'])
        configured.greedy = True
        c = search_config(configured)
        assert (c.worlds, c.depth, c.candidates, c.candidate_samples, c.select) == (8,1,4,12,'argmax')
        assert c.search_response and c.leaf_view == 'searcher'
        assert c.rollouts_per_world == 4 and c.rollout_crn
        for key, invalid in [('belief','oracle'), ('threads',0), ('alpha',float('nan')), ('worlds',0), ('rollouts_per_world',0)]:
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
def test_recorded_response_without_previous_launch_boundary_searches(models, kind):
    data = json.loads((DATA/'wire-game-updates-v64.json').read_text())
    result = OfficialSession(1).ingest(data['updates'][0]['game'])
    agent = SearchAgent(models[0], load_search_belief(str(models[1]) if kind == 'ar' else 'uniform'),
                        config=SearchConfig(search_response=True))
    advice = advise(result, agent, search=True)
    assert not result['history_complete'] and '_search_launch' not in result['public']
    assert advice['search_status'] == 'response_public_worlds'
    assert advice['action'] and advice['search_worlds'] > 0


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


def research_game():
    game = games()[0]
    game['players'][0]['items'] = [dict(id=991, modelId=55, fakeModelId=55),
                                    dict(id=992, modelId=19)]
    game['players'][0]['curses'] = ['dream']
    return game


def small_search(model, seed=7):
    agent = SearchAgent(model, load_search_belief('uniform'), config=SearchConfig(
        worlds=2, depth=1, candidates=3, candidate_samples=8, select='argmax'), seed=seed)
    agent.actor.sample = False
    return agent


def test_dream_allow_keeps_search_choice_and_budget(models):
    from gfrl.official_advisor import DREAM_REASON, advise_supported
    agent = small_search(models[0])
    with _policy_rng(7, (agent,)):
        initial = advise(result_for([research_game()]), agent, search=True)
    assert initial['action']['cards'] == [dict(model=55, item_id=991)]
    agent = small_search(models[0])
    with _policy_rng(7, (agent,)):
        advice = advise_supported(result_for([research_game()]), agent, search=True)
    assert advice['status'] == 'advice' and advice['unsupported'] == []
    assert advice['action'] == initial['action']
    assert advice['dream_played'] is True and DREAM_REASON in advice['warnings']
    assert advice['search_status'] == initial['search_status'] == 'main_public_worlds'
    assert advice['pi_search_top'] == initial['pi_search_top']
    assert advice['pi_theta_top'] == initial['pi_theta_top']
    assert agent.counts['searched'] == 1
    assert not advice.get('mask_exclusions') and not advice.get('dream_last_resort')


def test_ui_exclusion_searches_again(models, monkeypatch):
    from gfrl.official_advisor import advise_supported
    agent = small_search(models[0])
    with _policy_rng(7, (agent,)):
        initial = advise(result_for([research_game()]), agent, search=True)
    assert initial['action']['cards'] == [dict(model=55, item_id=991)]
    import gfrl.official_advisor as advisor
    seconds = []
    original = advisor.advise
    def timed(*args, **kwargs):
        advice = original(*args, **kwargs)
        seconds.append(advice['search_seconds'])
        return advice
    monkeypatch.setattr(advisor, 'advise', timed)
    agent = small_search(models[0])
    with _policy_rng(7, (agent,)):
        advice = advise_supported(result_for([research_game()]), agent, search=True, dream='last-resort')
    assert advice['search_status'] == 'ui_mask_search'
    assert advice['status'] == 'advice'
    assert advice['action']['cards'] == [dict(model=19, item_id=992)]
    assert agent.counts['searched'] == 2
    assert len(seconds) == 2 and min(seconds) > 0
    assert advice['search_seconds'] == pytest.approx(sum(seconds))
    assert all(c['action']['cards'] != [dict(model=55, item_id=991)] for c in advice['pi_search_top'])


@pytest.mark.parametrize('response', [False, True])
def test_restricted_root_equals_cursor_at_every_branch(response):
    result = result_for(games(response))
    public = result['public']
    public['hand'] = [dict(id=900, face=6, used=False, disguised=False),
                      dict(id=901, face=6, used=False, disguised=True),
                      dict(id=902, face=7, used=False, disguised=False),
                      dict(id=903, face=190, used=False, disguised=False),
                      dict(id=904, face=55, used=False, disguised=False),
                      dict(id=905, face=130, used=False, disguised=False),
                      dict(id=906, face=138, used=False, disguised=False),
                      dict(id=907, face=3, used=False, disguised=False)]
    # Response roots must retain the journal's identical own hand and tokens.
    if response:
        public['_search_launch']['before']['hand'] = deepcopy(public['hand'])
    public['players'][0]['status']['curses'] = ['dream']
    if response:
        public['_search_launch']['before']['players'][0]['status']['curses'] = ['dream']
    restrictions = dict(excluded_ids=[900, 901], excluded_kinds=['discard'], exclude_enemy_default_self=True)
    pending = [()]; visited = 0
    while pending:
        path = pending.pop()
        env = OfficialDecision(public, **restrictions)
        root = _core._official_search_root(public, [], 64, env.restrictions)
        for choice in path:
            env.step(np.array([choice], np.int32))
            if not env.complete:
                root.step(np.array([choice], np.int32))
        if env.complete:
            assert not {900, 901} & {c['item_id'] for c in env.semantic_action()['cards']}
            continue
        assert root_matches(root, env)
        visited += 1
        pending.extend(path + (int(c),) for c in np.flatnonzero(env.mask[0]))
    assert visited > 2


@pytest.mark.parametrize('kind', ['uniform', 'ar'])
def test_restricted_search_does_not_see_hidden_own_or_opponent_truth(models, kind):
    from gfrl.official_advisor import advise_supported
    outcomes = []; inputs = []
    for changed in (False, True):
        game = research_game()
        game['players'][0]['items'].append(dict(id=993, modelId=60))
        if changed:
            game['players'][0]['items'][0]['modelId'] = 60
            game['players'][1]['items'] = [dict(id=880+i, modelId=m, fakeModelId=6)
                                           for i,m in enumerate((94,232,207))]
        result = result_for([game])
        env = OfficialDecision(result['public'], excluded_ids=[991], excluded_kinds=['discard'],
                               exclude_enemy_default_self=True)
        root = _core._official_search_root(env.public, [], 64, env.restrictions)
        snapshots = [{k: getattr(root, k).copy() for k in (*OBS_FIELDS, *PREFIX_FIELDS, 'substep')}]
        agent = small_search(models[0])
        if kind == 'ar':
            agent.belief = load_search_belief(str(models[1]))
        original = agent._world_env
        def spy(src, rows, assignments, seeds, depth, role=None):
            world, status = original(src, rows, assignments, seeds, depth, role)
            snapshots.append(dict(assignments=np.array(assignments), seeds=np.array(seeds), status=status,
                **{k: getattr(world, k).copy() for k in (*OBS_FIELDS, *PREFIX_FIELDS)}))
            return world, status
        agent._world_env = spy
        with _policy_rng(7, (agent,)):
            advice = advise(dict(result, env=env), agent, search=True)
        assert advice['search_status'] == 'main_public_worlds'
        assert agent.counts['invalid_loads'] == 0
        outcomes.append((advice['action'], agent.last_search)); inputs.append(snapshots)
    assert outcomes[0] == outcomes[1]
    assert len(inputs[0]) == len(inputs[1]) > 1
    for left, right in zip(*inputs):
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize('reason', ['restricted_root_slot_mismatch', 'restricted_root_world_mismatch'])
@pytest.mark.parametrize('failure_at', ['root', 'world'])
def test_research_unsupported_reason(models, monkeypatch, reason, failure_at):
    from gfrl.official_advisor import advise_supported
    original = _core._official_search_root
    calls = []
    def root(public, events, k, restrictions):
        calls.append(restrictions)
        if restrictions['excluded_slots'] and failure_at == 'root':
            raise ValueError(reason)
        return original(public, events, k, restrictions)
    monkeypatch.setattr(_core, '_official_search_root', root)
    agent = small_search(models[0])
    if failure_at == 'world':
        original_world = agent._world_env
        def worlds(*args, **kwargs):
            if calls[-1]['excluded_slots']:
                raise ValueError(reason)
            return original_world(*args, **kwargs)
        monkeypatch.setattr(agent, '_world_env', worlds)
    with _policy_rng(7, (agent,)):
        advice = advise_supported(result_for([research_game()]), agent, search=True, dream='last-resort')
    assert advice['status'] == 'advice'
    assert advice['search_status'] == 'ui_mask_policy_fallback'
    assert advice['search_fallback_reason'] == reason + '_policy_fallback'
    assert len(calls) == 2


def test_search_cap_includes_further_retries(models, monkeypatch):
    from gfrl.official_advisor import advise_supported
    calls = []
    original = OfficialDecision.action_support
    def support(env):
        calls.append(env.semantic_action())
        return original(env) if len(calls) < 3 else []
    # A genuine Dream exclusion followed by an unmaskable gap needs a third
    # selection; only the first two may invoke search.
    def gaps(env):
        if len(calls) == 1:
            calls.append(env.semantic_action())
            return ['synthetic_unmaskable_gap']
        return support(env)
    monkeypatch.setattr(OfficialDecision, 'action_support', gaps)
    agent = small_search(models[0])
    with _policy_rng(7, (agent,)):
        advice = advise_supported(result_for([research_game()]), agent, search=True, dream='last-resort')
    assert advice['status'] == 'advice'
    assert agent.counts['searched'] == 2
    assert advice['search_status'] == 'ui_retry_policy_fallback'
    assert advice['search_fallback_reason'] == 'search_limit'


def test_dream_last_resort_uses_remaining_search_budget(models):
    from gfrl.official_advisor import advise_supported
    game = research_game()
    game['players'][0]['items'] = [dict(id=991, modelId=6, fakeModelId=6)]
    agent = small_search(models[0], seed=0)
    with _policy_rng(0, (agent,)):
        advice = advise_supported(result_for([game]), agent, search=True, dream='last-resort')
    assert advice['search_status'] == 'dream_last_resort_search'
    assert advice['dream_last_resort'] and advice['status'] == 'advice'
    assert agent.counts['searched'] == 2


@pytest.mark.parametrize('response', [False, True])
@pytest.mark.parametrize('count', range(2, 7))
@pytest.mark.parametrize('kind', ['uniform', 'ar'])
@pytest.mark.parametrize('singleton', [False, True])
def test_dream_singleton_faces_produce_candidates_without_truth(models, response, count, kind, singleton):
    """Official disguises include faces with no natural Dream partners."""
    outcomes = []; snapshots = []
    for changed in (False, True):
        sequence = games(response)
        for game in sequence:
            game['players'][0].update(hp=22, mp=15, cp=40, curses=['dream'])
            game['players'][0]['items'] = [
                dict(id=990+i, modelId=6 if changed else face, fakeModelId=face)
                for i, face in enumerate([189, 58, 20, 91, 195, 63][:count])]
            game['players'][0]['items'] += [dict(id=980, modelId=190)]
            if changed:
                game['players'][1]['items'] = [dict(id=880+i, modelId=m)
                                               for i, m in enumerate((94, 232, 209))]
        result = result_for(sequence)
        root = _core._official_search_root(result['public'], result['env'].events, 64,
                                           result['env'].restrictions)
        assert root_matches(root, result['env'])
        assert root.determinize_support(np.array([0], np.int32))[0] == 0
        agent = SearchAgent(models[0], load_search_belief(str(models[1]) if kind == 'ar' else 'uniform'),
            config=SearchConfig(worlds=4, candidates=1 if singleton else 4, candidate_samples=8,
                                depth=2, search_response=True, skip_single_candidate=singleton), seed=17)
        seen = []
        original = agent._world_env
        def spy(*args, **kwargs):
            env, status = original(*args, **kwargs)
            seen.append(dict(status=status.copy(), **{k: getattr(env, k).copy()
                             for k in (*OBS_FIELDS, *PREFIX_FIELDS)}))
            return env, status
        agent._world_env = spy
        with _policy_rng(31, (agent,)):
            selected = agent.search(root, [0])
        assert selected and agent.last_search[0]['candidates']
        diagnostic = agent.last_candidate_diagnostics[0]
        assert diagnostic['sampled'] == 8
        assert diagnostic['singleton'] == singleton
        assert all(n == 4 for n in diagnostic['valid_worlds'])
        assert agent.counts['invalid_loads'] == agent.counts['invalid_candidate_worlds'] == 0
        assert agent.counts['singleton_validated_worlds'] == (4 if singleton else 0)
        outcomes.append((selected, agent.last_search, diagnostic)); snapshots.append(seen)
    assert outcomes[0] == outcomes[1]
    for left, right in zip(*snapshots, strict=True):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


def test_world_rejection_does_not_repeat_search_on_ui_exclusion(models, monkeypatch):
    from gfrl.official_advisor import advise_supported
    agent = small_search(models[0])
    class FirstCardPolicy:
        sample = False
        def __call__(self, env, rows):
            choices = np.zeros(env.num_envs, np.int32)
            for r in rows:
                legal = np.flatnonzero(env.mask[r])
                choices[r] = 2 if env.stage[r] == 2 and 2 in legal else legal[0]
            return choices
    agent.actor = FirstCardPolicy()
    original = agent._world_env
    calls = []
    def rejected(src, rows, assignments, seeds, depth, role=None):
        calls.append(role)
        invalid = np.zeros_like(assignments)
        invalid[:, 6] = 19  # impossible hand: reject through the real loader
        return original(src, rows, invalid, seeds, depth, role)
    monkeypatch.setattr(agent, '_world_env', rejected)
    with _policy_rng(7, (agent,)):
        advice = advise_supported(result_for([research_game()]), agent, search=True, dream='last-resort')
    assert advice['status'] == 'advice'
    assert calls == ['candidate']
    assert advice['initial_search_status'] == 'no_candidates_policy_fallback'
    assert advice['search_fallback_reason'] == 'world_rejection'
    assert agent.counts['searched'] == 1
