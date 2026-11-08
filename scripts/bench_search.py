#!/usr/bin/env python3
"""Compare eval_search.py with a saved pre-optimization search.py + actor.py.

Before editing, copy both files into --baseline-dir. Runs alternate order;
all policy choices, candidate paths, q estimates and game summaries are checked.
Timings exclude model loading and use eval_search.py's normal phase counters.
"""
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import numpy as np
import gfrl.search as current


def load(name, path):
    spec = importlib.util.spec_from_file_location('gfrl.'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--out', type=Path, default=Path('bench-search.json'))
    args, evaluation_args = p.parse_known_args()
    if args.repeats < 1:
        p.error('positive repeats required')
    original = current.SearchAgent
    before = load('bench_old_search', args.baseline_dir/'search.py')
    before.Actor = load('bench_old_actor', args.baseline_dir/'actor.py').Actor
    runs = {'before': [], 'after': []}
    traces = {}
    max_q_error = 0.0
    with TemporaryDirectory() as tmp:
        for rep in range(args.repeats):
            for name in (('before', 'after') if rep % 2 == 0 else ('after', 'before')):
                actions, reports = [], []
                base = before.SearchAgent if name == 'before' else original
                class Traced(base):
                    def __init__(self, *a, **kw):
                        if name == 'before':
                            if kw.pop('profile', False):
                                raise ValueError('baseline benchmark requires --profile off')
                        super().__init__(*a, **kw)
                    def search(self, *a, **kw):
                        result = super().search(*a, **kw)
                        reports.extend(self.last_search)
                        return result
                    def __call__(self, env, rows):
                        result = super().__call__(env, rows)
                        actions.append(result.tolist())
                        return result
                current.SearchAgent = Traced
                out = Path(tmp)/f'{name}-{rep}.json'
                saved_argv = sys.argv
                try:
                    sys.argv = ['eval_search.py', *evaluation_args, '--out', str(out)]
                    with redirect_stdout(io.StringIO()):
                        runpy.run_path(str(Path(__file__).with_name('eval_search.py')), run_name='__main__')
                finally:
                    sys.argv = saved_argv
                    current.SearchAgent = original
                report = json.loads(out.read_text())
                runs[name].append(report)
                traces[name] = actions, reports
                print(f'{name} repeat {rep+1}: {report["elapsed_seconds"]:.3f}s', flush=True)
            a, b = traces['before'], traces['after']
            assert a[0] == b[0], 'policy choices changed'
            assert len(a[1]) == len(b[1])
            for x, y in zip(a[1], b[1]):
                assert x['selected'] == y['selected']
                assert [c['path'] for c in x['candidates']] == [c['path'] for c in y['candidates']]
                max_q_error = max(max_q_error, max(abs(c['q']-d['q']) for c, d in zip(x['candidates'], y['candidates'])))
                for key in ('q', 'log_pi', 'log_rho'):
                    np.testing.assert_allclose([c[key] for c in x['candidates']], [c[key] for c in y['candidates']], atol=1e-7, rtol=1e-6)
            for key in ('games', 'wins', 'losses', 'draws', 'mean_decisions', 'first_win_rate', 'second_win_rate'):
                assert runs['before'][-1][key] == runs['after'][-1][key], f'{key} changed'
    medians = {name: {key: float(np.median([r['search']['seconds_per_search'][key] for r in values]))
                      for key in ('sample', 'candidate', 'rollout', 'leaf', 'total')} for name, values in runs.items()}
    result = dict(runs=runs, median_seconds_per_search=medians, equality_verified=True,
                  searched_decisions=len(traces['after'][1]), max_q_error=max_q_error)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print('| Phase | Before s/root | After s/root |\n|---|---:|---:|')
    for key in medians['before']:
        print(f'| {key} | {medians["before"][key]:.6f} | {medians["after"][key]:.6f} |')


if __name__ == '__main__':
    main()
