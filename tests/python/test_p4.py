"""P4 end-to-end reports, resume identity, and both search seats' boundary."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from gfrl import VecEnv, _core
from gfrl.belief import Belief, BeliefConfig
from gfrl.belief_metrics import fit_baselines
from gfrl.evaluation import evaluate_policies, _policy_rng
from gfrl.model import Model, ModelConfig, OBS_FIELDS
from gfrl.search import SearchAgent, SearchConfig


def fixtures():
    torch.manual_seed(31)
    config = ModelConfig(d=16, depth=1, heads=2, ff=32)
    model = Model(config).eval()
    belief = Belief(config, BeliefConfig(depth=1, heads=2, ff=32)).eval()
    labels = torch.zeros(8, 297, dtype=torch.uint8)
    labels[:, 7] = 1
    data = dict(label_opp_hand=labels, label_opp_hand_count=labels.sum(1))
    return model, belief, data


def test_p4_all_pairs_and_resume(tmp_path):
    model, belief, data = fixtures()
    policy, ar, fit = [tmp_path / name for name in ('policy.pt', 'ar.pt', 'fit.pt')]
    torch.save(dict(ema=model.state_dict(), model_config=asdict(model.config)), policy)
    torch.save(dict(model=belief.state_dict(), model_config=asdict(model.config),
                    belief_config=asdict(belief.belief_config)), ar)
    torch.save(data, fit)
    out = tmp_path / 'p4'
    command = [sys.executable, str(Path(__file__).resolve().parents[2] / 'scripts/eval_p4.py'),
               '--checkpoint', str(policy), '--belief', str(ar), '--fit', str(fit),
               '--out-dir', str(out), '--pairs', '1', '--num-envs', '2', '--worlds', '1',
               '--candidates', '1', '--candidate-samples', '2', '--depth', '0',
               '--search-response', '--with-oracle', '--progress-interval', '0']
    subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['completed_matches'] == summary['expected_matches'] == 4
    assert [r['match'] for r in summary['results']] == ['B_vs_A', 'C_vs_A', 'B_vs_C', 'D_vs_A']
    for report in summary['results']:
        assert report['games'] == 2
        assert len(report['wilson_interval']) == len(report['bootstrap_interval']) == 2
        for record in report['agents'].values():
            assert record['decision_timing']['decisions'] > 0
            assert record['decision_timing']['seconds'] > 0
    direct = summary['results'][2]['agents']
    assert direct['B']['search_config'] == direct['C']['search_config']
    assert direct['B']['search_config']['search_response'] is True
    assert direct['B']['belief'] == str(ar) and direct['C']['belief'] == 'b1'
    assert all(direct[n]['search']['counts']['searched'] > 0 for n in ('B', 'C'))
    assert 'Pair bootstrap 95%' in (out / 'summary.md').read_text()
    paths = [out / f'{r["match"]}.json' for r in summary['results']]
    mtimes = [p.stat().st_mtime_ns for p in paths]
    resumed = subprocess.run(command+['--resume'], check=True, capture_output=True, text=True)
    assert resumed.stdout.count('Skip completed') == 4
    assert [p.stat().st_mtime_ns for p in paths] == mtimes
    paths[-1].unlink()
    resumed = subprocess.run(command+['--resume'], check=True, capture_output=True, text=True)
    assert resumed.stdout.count('Run ') == 1
    mismatch = subprocess.run(command+['--resume', '--seed', '99'], capture_output=True, text=True)
    assert mismatch.returncode and 'mismatch' in mismatch.stderr
    # Optional diagnostic selection does not invalidate the three fair matches.
    fair = subprocess.run([arg for arg in command if arg != '--with-oracle']+['--resume'],
                          check=True, capture_output=True, text=True)
    assert fair.stdout.count('Skip completed') == 3
    assert json.loads((out / 'summary.json').read_text())['expected_matches'] == 3


def test_search_vs_search_each_seat_hidden_hand_invariance():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, ar, data = fixtures()
        beliefs = {'B': ar, 'C': fit_baselines(data)['B1']}
        config = SearchConfig(worlds=1, candidates=2, candidate_samples=3, depth=1)
        seen = set()

        def policy(name):
            live = SearchAgent(model, beliefs[name], config=config, seed=12)

            def call(env, rows):
                for r in rows[env.substep[rows] == 0]:
                    seat = int(env.actor_seat[r])
                    key = (name, seat)
                    phase = env.obs_schema()['cat']['phase']['index']
                    if key in seen or env.obs_cat[r, 0, phase] != 1 or env.determinize_support(np.array([r], np.int32))[0] != 0:
                        continue
                    before = {k: getattr(env, k)[r].copy() for k in OBS_FIELDS}
                    first = SearchAgent(model, beliefs[name], config=config, seed=917)
                    with _policy_rng(67, (first,)):
                        a = first(env, np.array([r], np.int32)).copy()
                    # Change true opposing inventory and source RNG. The acting
                    # seat's observation (including its private history) is fixed.
                    try:
                        _core._test_replace_hidden_hand(env, int(r), 559)
                    except ValueError:
                        continue  # Legacy fixture cannot reconstruct own Dream.
                    for k in OBS_FIELDS:
                        np.testing.assert_array_equal(before[k], getattr(env, k)[r])
                    second = SearchAgent(model, beliefs[name], config=config, seed=917)
                    with _policy_rng(67, (second,)):
                        b = second(env, np.array([r], np.int32)).copy()
                    assert a[r] == b[r]
                    assert first.last_search == second.last_search
                    assert first.counts == second.counts
                    seen.add(key)
                return live(env, rows)
            call.device = 'cpu'
            return call

        result = evaluate_policies(policy('B'), policy('C'), 2, 4, 1, 73, track_projection=True)
        assert result['games'] == 4
        assert seen == {('B', 0), ('B', 1), ('C', 0), ('C', 1)}
    finally:
        torch.set_num_threads(previous)


def test_danger_probe_is_read_only_and_pair_is_always_masked():
    env = VecEnv(1, 1, track_projection=True)
    for opponent in (False, True):
        _core._test_danger_root(env, opponent)
        fields = {k: getattr(env, k).copy() for k in (*OBS_FIELDS, 'mask', 'label_opp_hand')}
        # False fixture has an empty enemy hand: adding a card would change the
        # observation, so it must be untestable. True has one hidden 209 slot.
        assert _core._measure_danger_probe(env, 0) == [True, opponent, False, opponent]
        for k, before in fields.items():
            np.testing.assert_array_equal(before, getattr(env, k))

    # A play can finish with 94, but cannot append 232 (including forced stages).
    _core._test_danger_root(env, False)
    assert not _core._measure_danger_selected(env, 0, 0)
    env.step(np.array([0], np.int32))
    # Finish-cards is now forced because the only possible extra is forbidden.
    assert env.stage[0] == 2
    assert not _core._measure_danger_selected(env, 0, 0)
    env.step(np.array([0], np.int32))
    assert env.decision_done[0]
