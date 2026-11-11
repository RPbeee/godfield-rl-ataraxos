"""End-to-end paired budget sweep, timing and interrupted-run recovery."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import torch

from gfrl.model import Model, ModelConfig


SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


def test_sweep_grid_resume_and_eval_equivalence(tmp_path):
    torch.manual_seed(7)
    model = Model(ModelConfig(d=16, depth=1, heads=2, ff=32))
    checkpoint = tmp_path / 'policy.pt'
    torch.save(dict(ema=model.state_dict(), model_config=asdict(model.config)), checkpoint)
    common = ['--checkpoint', str(checkpoint), '--belief', 'uniform', '--pairs', '1',
              '--num-envs', '2', '--candidates', '1', '--candidate-samples', '2',
              '--seed', '7', '--search-response', '--progress-interval', '0']
    out_dir = tmp_path / 'sweep'
    command = [sys.executable, str(SCRIPTS / 'sweep_search.py'), *common,
               '--worlds', '1', '2', '--depth', '0', '1', '--out-dir', str(out_dir)]

    def run(args):
        return subprocess.run(args, check=True, capture_output=True, text=True)

    first = run(command)
    assert first.stdout.count('Run ') == 5
    summary = json.loads((out_dir / 'summary.json').read_text())
    assert summary['completed_settings'] == summary['expected_settings'] == 4
    assert [(r['worlds'], r['depth']) for r in summary['results'][1:]] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert 'Pair bootstrap 95%' in (out_dir / 'summary.md').read_text()
    assert 'baseline (no search)' in (out_dir / 'summary.md').read_text()
    for row in summary['results']:
        assert row['games'] == 2
        assert row['decisions'] > 0 and row['seconds_per_decision'] > 0
        assert row['seconds_per_decision'] == row['seconds'] / row['decisions']
        if row['kind'] == 'search':
            report = json.loads((out_dir / row['report']).read_text())
            assert row['decisions'] == report['search']['counts']['decisions']
            assert row['searched_fraction'] == report['search']['searched_fraction']
            assert row['seconds_per_search_main'] > 0
            assert row['seconds_per_search_response'] > 0

    reports = [out_dir / row['report'] for row in summary['results']]
    mtimes = {p: p.stat().st_mtime_ns for p in reports}
    resumed = run(command + ['--resume'])
    assert resumed.stdout.count('Skip completed ') == 5 and 'Run ' not in resumed.stdout
    assert {p: p.stat().st_mtime_ns for p in reports} == mtimes
    assert json.loads((out_dir / 'summary.json').read_text()) == summary

    # Recover an interrupted final configuration; completed reports stay intact.
    missing = reports[-1]
    missing.unlink()
    (missing.with_suffix('.partial.json')).write_text('{}')
    recovered = run(command + ['--resume'])
    assert recovered.stdout.count('Run ') == 1
    assert recovered.stdout.count('Skip completed ') == 4
    assert all(p.stat().st_mtime_ns == mtimes[p] for p in reports[:-1])

    # An explicit sparse grid reuses identical reports without rerunning the baseline.
    sparse = run(command + ['--grid', '1:0', '2:1', '--resume'])
    assert sparse.stdout.count('Skip completed ') == 3 and 'Run ' not in sparse.stdout

    changed = subprocess.run(command + ['--seed', '8', '--resume'], capture_output=True, text=True)
    assert changed.returncode != 0 and 'mismatch' in changed.stderr

    # Timing wrappers must not change paired outcomes, policy RNG or search counts.
    eval_out = tmp_path / 'eval.json'
    run([sys.executable, str(SCRIPTS / 'eval_search.py'), *common,
         '--worlds', '1', '--depth', '0', '--out', str(eval_out)])
    single = json.loads(eval_out.read_text())
    sweep = json.loads(reports[1].read_text())
    for key in ('games', 'wins', 'losses', 'draws', 'win_rate', 'wilson_interval',
                'bootstrap_interval', 'mean_decisions', 'first_win_rate', 'second_win_rate'):
        assert single[key] == sweep[key]
    assert single['search']['counts'] == sweep['search']['counts']
    assert 'decision_timing' not in single  # Preserve eval_search's CLI schema.
