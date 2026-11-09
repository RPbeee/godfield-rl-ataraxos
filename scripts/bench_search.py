#!/usr/bin/env python3
"""Compare eval_search.py with a saved pre-optimization search.py + actor.py.

Before editing, copy both files into --baseline-dir. Runs alternate order;
all policy choices, candidate paths, q estimates and game summaries are checked.
Timings exclude model loading and use eval_search.py's normal phase counters.
--micro measures fixed-root ablations, allowing rollout RNG changes while
requiring exact before/host/cache equality and unchanged candidate priors.
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


def micro(args, argv):
    """Alternate fixed-root ablations; exclude root collection and model loading."""
    from dataclasses import asdict, replace
    from time import perf_counter
    import platform
    import torch
    from gfrl import VecEnv
    from gfrl.actor import Actor, synchronize
    from gfrl.evaluation import load_policy_model, _policy_rng
    from gfrl.runtime import configure_precision

    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--belief', default='uniform')
    p.add_argument('--fit')
    p.add_argument('--device', default='cpu', choices=('cpu', 'mps', 'cuda'))
    p.add_argument('--roots', type=int, default=4, help='0: every supported root in --num-envs')
    p.add_argument('--num-envs', type=int, default=0)
    p.add_argument('--variants', nargs='+', help='subset of before/host/cache/forced/singleton/all/bounded')
    p.add_argument('--root-steps', type=int, default=20)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=741)
    p.add_argument('--worlds', type=int, default=64)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--candidates', type=int, default=16)
    p.add_argument('--candidate-samples', type=int, default=64)
    p.add_argument('--max-inference-batch', type=int, default=8192)
    p.add_argument('--search-cache-mb', type=int, default=256)
    p.add_argument('--root-batch-size', type=int, default=0)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args(argv)
    if (a.roots < 0 or a.num_envs < 0 or a.threads < 1 or a.root_steps < 0
            or (not a.roots and not a.num_envs) or (a.num_envs and a.roots > a.num_envs)):
        p.error('positive threads and roots (or num-envs with roots=0) required')
    torch.set_num_threads(a.threads)
    configure_precision(a.device, a.tf32)
    m = load_policy_model(a.checkpoint, a.device)
    belief = current.load_search_belief(a.belief, a.fit, a.device)
    env = VecEnv(a.num_envs or max(a.roots * 4, 16), a.threads, a.seed, track_projection=True)
    actor = Actor(m, env, a.device)
    with _policy_rng(a.seed, (actor,)):
        for _ in range(a.root_steps):
            env.step(actor.act().choices)
        # Oversample live rows, then take supported boundaries from one snapshot.
        for _ in range(1000):
            rows = np.flatnonzero((env.stage >= 0) & (env.substep == 0)).astype(np.int32)
            roots = rows[env.determinize_support(rows) == 0]
            if len(roots) and (not a.roots or len(roots) >= a.roots):
                roots = roots[:a.roots] if a.roots else roots
                break
            env.step(actor.act().choices)
        else:
            raise RuntimeError('could not collect requested simultaneous supported roots; use --root-steps 0')
    before = load('bench_old_search', args.baseline_dir/'search.py')
    before.Actor = load('bench_old_actor', args.baseline_dir/'actor.py').Actor
    config = current.SearchConfig(worlds=a.worlds, depth=a.depth, candidates=a.candidates,
        candidate_samples=a.candidate_samples, beta=.1, max_inference_batch=a.max_inference_batch,
        search_cache_mb=a.search_cache_mb)
    variants = [
        ('before', before.SearchAgent, config),
        ('host', current.SearchAgent, replace(config, search_cache_device='host')),
        ('cache', current.SearchAgent, config),
        ('forced', current.SearchAgent, replace(config, skip_forced_inference=True)),
        ('singleton', current.SearchAgent, replace(config, skip_single_candidate=True)),
        ('all', current.SearchAgent, replace(config, skip_forced_inference=True, skip_single_candidate=True)),
    ]
    if a.root_batch_size:
        variants.append(('bounded', current.SearchAgent, replace(config, skip_forced_inference=True,
                         skip_single_candidate=True, root_batch_size=a.root_batch_size)))
    if a.variants:
        unknown = set(a.variants)-{name for name, _, _ in variants}
        if unknown:
            p.error(f'unknown variants (bounded requires --root-batch-size): {unknown}')
        variants = [v for v in variants if v[0] in a.variants]
    runs = {name: [] for name, _, _ in variants}
    equality = {}
    for rep in range(args.repeats):
        traces = {}
        order = variants[rep % len(variants):] + variants[:rep % len(variants)]
        for name, cls, c in order:
            agent = cls(m, belief, a.device, config=c, seed=421, num_threads=a.threads, profile=a.profile)
            # Warm the workspace and accelerator, then restore RNG for matching inputs.
            with _policy_rng(42, (agent,)):
                agent.search(env, roots)
            agent.rng = np.random.default_rng(421)
            agent.seconds.clear()
            agent.counts.clear()
            synchronize(agent.device)
            if agent.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(agent.device)
            start = perf_counter()
            with _policy_rng(42, (agent,)):
                chosen = agent.search(env, roots)
            synchronize(agent.device)
            elapsed = perf_counter()-start
            phases = dict(agent.seconds)
            phases['total'] = elapsed
            runs[name].append(dict(seconds_per_call=phases,
                seconds_per_root={k: v/len(roots) for k, v in phases.items()},
                counts=dict(agent.counts), roots=len(roots),
                candidates=sum(map(len, agent.last_candidates.values())),
                singleton_candidates=sum(len(x) == 1 for x in agent.last_candidates.values()),
                peak_cuda_bytes=(torch.cuda.max_memory_allocated(agent.device)
                                 if agent.device.type == 'cuda' else None)))
            traces[name] = chosen, agent.last_search, agent.last_candidates
            print(f'{name} repeat {rep+1}: {elapsed:.6f}s / {len(roots)} roots', flush=True)
            del agent
        for name in ('host', 'cache'):
            if 'before' not in traces or name not in traces:
                continue
            assert json.dumps(traces['before'], sort_keys=True) == json.dumps(traces[name], sort_keys=True), f'{name}: exact baseline equality failed'
            equality[name] = True
        # For a single batch, RNG-changing rollout flags must preserve all candidate priors.
        for name in ('forced', 'singleton', 'all'):
            if 'before' not in traces or name not in traces:
                continue
            assert traces['before'][2] == traces[name][2], f'{name}: candidates or priors changed'
    medians = {name: {key: float(np.median([r['seconds_per_call'].get(key, 0) for r in values]))
                     for key in ('sample', 'candidate', 'rollout', 'leaf', 'total')}
               for name, values in runs.items()}
    result = dict(arguments=vars(a), model_config=asdict(m.config), config=asdict(config),
                  median_seconds_per_call=medians, runs=runs, exact_equality=equality,
                  source_rows=roots.tolist(), source_num_envs=env.num_envs,
                  platform=platform.platform(), torch_version=torch.__version__,
                  mps_available=torch.backends.mps.is_available())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print('| Variant | Sample | Candidate | Rollout | Leaf | Total s/call |\n|---|---:|---:|---:|---:|---:|')
    for name, phases in medians.items():
        print('| '+name+' | '+' | '.join(f'{v:.6f}' for v in phases.values())+' |')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--micro', action='store_true', help='fixed-root per-call ablation benchmark')
    p.add_argument('--baseline-dir', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--out', type=Path, default=Path('bench-search.json'))
    args, evaluation_args = p.parse_known_args()
    if args.repeats < 1:
        p.error('positive repeats required')
    if args.micro:
        return micro(args, evaluation_args)
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
                    assert [c[key] for c in x['candidates']] == [c[key] for c in y['candidates']], key
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
