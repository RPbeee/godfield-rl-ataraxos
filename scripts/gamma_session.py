#!/usr/bin/env python3
"""Implementation of session_gamma.sh (stdlib + existing torch only)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch
import train


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def invoke(command, log):
    begin = perf_counter()
    with log.open('a') as f:
        process = subprocess.Popen([sys.executable, *command], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            f.write(line)
            f.flush()
        if process.wait():
            raise RuntimeError(f'command failed; see {log}')
    print(f'{log.name}: {perf_counter()-begin:.2f}s', flush=True)


def summary(out, runs, initial_iteration):
    lines = ['# Gamma experiment', '',
             'Training metrics: mean of the last 20 iterations; lengths count own decisions per player/game.', '',
             '| Gamma | Iterations done | Value accuracy | Entropy | KL | Own decisions | Win own decisions | Loss own decisions |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return '—' if value is None else f'{value:.6g}'
    for gamma, directory in runs:
        path = directory/'metrics.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        rows = rows[-20:]
        def mean(key):
            values = [r[key] for r in rows if r.get(key) is not None]
            return fmt(sum(values)/len(values)) if values else '—'
        done = load(directory/'checkpoint.pt')['iteration'] - initial_iteration
        keys = ('value_accuracy', 'entropy', 'kl', 'mean_own_decisions',
                'mean_win_own_decisions', 'mean_loss_own_decisions')
        lines.append(f'| {gamma:g} | {done} | ' + ' | '.join(mean(k) for k in keys) + ' |')
    lines += ['', 'Evaluation: paired deals, seed 74100; game lengths count external decisions across both seats.', '',
              '| Match | Games | A win rate | Wilson 95% | Pair bootstrap 95% | First | Second | Draws | Mean decisions | A win decisions | A loss decisions | Truncations |',
              '|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for path in sorted(out.glob('eval-*.json')):
        for r in json.loads(path.read_text())['results']:
            ci = lambda key: '–'.join(f'{x:.2%}' for x in r[key])
            lines.append(f"| {path.stem[5:]} | {r['games']} | {r['win_rate']:.2%} | "
                         f"{ci('wilson_interval')} | {ci('bootstrap_interval')} | "
                         f"{r['first_win_rate']:.2%} | {r['second_win_rate']:.2%} | {r['draws']} | " +
                         ' | '.join(fmt(r[k]) for k in ('mean_game_length', 'mean_win_decisions',
                                                        'mean_loss_decisions', 'truncations')) + ' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    init = Path(os.environ.get('INIT', 'runs/play/m-2166.pt')).resolve()
    out = Path(os.environ.get('OUT_DIR', 'runs/gamma'))
    out.mkdir(parents=True, exist_ok=True)
    gammas = [float(x) for x in os.environ.get('GAMMAS', '1 0.995 0.98').split()]
    if not gammas or len(set(gammas)) != len(gammas) or any(not 0 < g <= 1 for g in gammas):
        raise ValueError('GAMMAS must be distinct values in (0,1]')
    if any(g < 1 for g in gammas) and 1 not in gammas:
        raise ValueError('GAMMAS must include 1 for comparisons against gamma=1')
    source = load(init)
    defaults = vars(train.parse_args([]))
    config = {k: v for k, v in source['config'].items() if k in defaults}
    # Environment overrides enable tiny CPU sessions without changing the model.
    for key, default in defaults.items():
        if key.upper() in os.environ and key not in ('config', 'resume', 'init_from'):
            raw = os.environ[key.upper()]
            config[key] = (raw.lower() in ('1', 'true', 'yes') if isinstance(default, bool)
                           else type(default)(raw) if default is not None else raw)
    config['device'] = os.environ.get('DEVICE', 'cuda')
    config['eval_every'] = int(os.environ.get('EVAL_EVERY', '0'))
    budget = float(os.environ.get('HOURS_EACH', '3.3'))*3600
    iterations = int(os.environ.get('ITERATIONS', '100000'))
    if budget <= 0 or iterations <= 0:
        raise ValueError('HOURS_EACH and ITERATIONS must be positive')
    with init.open('rb') as f:
        fingerprint = hashlib.file_digest(f, 'md5').hexdigest()
    runs = [(g, out/f'g{g:g}') for g in gammas]
    for gamma, directory in runs:
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory/'checkpoint.pt'
        completed = directory/'complete.json'
        manifest = dict(init=str(init), md5=fingerprint, gamma=gamma,
                        hours_each=budget/3600, iterations=iterations, config=config)
        manifest_path = directory/'session.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f'session settings changed: {manifest_path}')
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        if completed.exists():
            continue
        current = load(checkpoint) if checkpoint.exists() else None
        remaining = budget-(current['elapsed_s'] if current else 0)
        additional = source['iteration']+iterations-(current['iteration'] if current else source['iteration'])
        if remaining > 0 and additional > 0 and not (current and current.get('stop_reason') == 'time_budget'):
            settings = dict(config, config=None, gamma=gamma, run_dir=str(directory),
                            max_hours=remaining/3600, iterations=additional,
                            resume='auto' if current else None,
                            init_from=None if current else str(init))
            settings_path = directory/'session-config.json'
            settings_path.write_text(json.dumps(settings, indent=2)+'\n')
            invoke(['scripts/train.py', '--config', str(settings_path)], directory/'train.log')
        current = load(checkpoint)
        completed.write_text(json.dumps(dict(iteration=current['iteration'], elapsed_s=current['elapsed_s'],
                                            reason=current.get('stop_reason') or 'iteration_budget'), indent=2)+'\n')
    pairs = os.environ.get('PAIRS', '1000')
    baseline = next((d/'checkpoint.pt' for g, d in runs if g == 1), None)
    for gamma, directory in runs:
        opponents = [('init', init)]
        if gamma < 1 and baseline is not None:
            opponents.append(('g1', baseline))
        for name, opponent in opponents:
            report = out/f'eval-g{gamma:g}-vs-{name}.json'
            if report.exists():
                continue
            invoke(['scripts/eval_policies.py', '--a', str(directory/'checkpoint.pt'), '--b', str(opponent),
                    '--pairs', pairs, '--num-envs', os.environ.get('EVAL_NUM_ENVS', '512'),
                    '--threads', str(config.get('threads', 16)), '--device', config['device'],
                    '--eval-seed', '74100', '--out', str(report),
                    '--tf32' if config.get('tf32', True) else '--no-tf32'], report.with_suffix('.log'))
    summary(out, runs, source['iteration'])
    print(f'Summary: {out / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
