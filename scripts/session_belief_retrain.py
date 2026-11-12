#!/usr/bin/env python3
"""Implementation of session_belief_retrain.sh; all budgets come from env."""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def main():
    def env(key, default):
        return os.environ.get(key, str(default))

    out = Path(env('OUT', env('OUT_DIR', 'runs/belief-m2166')))
    out.mkdir(parents=True, exist_ok=True)
    policy = env('CHECKPOINT', 'runs/play/m-2166.pt')
    old = env('BELIEF', 'runs/play/belief-ar-m.pt')
    hours = float(env('MAX_HOURS', 6.5))
    batch = int(env('BATCH_SIZE', 1024))
    ratio = float(env('REPLAY_RATIO', 2))
    rate = float(env('COLLECT_POSITIONS_PER_S', 290))
    warmup = int(env('WARMUP_POSITIONS', 50000))
    # Collection-limited upper bound, less initial replay warmup. Reserve optional
    # measured update/evaluation overhead with SCHEDULE_OVERHEAD_HOURS.
    overhead = float(env('SCHEDULE_OVERHEAD_HOURS', 0))
    steps = int(env('STEPS', max(1, int(((hours-overhead)*3600*rate-warmup)*ratio/batch))))
    common = ['--device', env('DEVICE', 'cuda'), '--num-envs', env('NUM_ENVS', 512),
              '--threads', env('THREADS', 4)]
    tf32 = ['--tf32' if env('TF32', 1) == '1' else '--no-tf32']
    fit, self_data, greedy = [str(out / f'{name}.pt') for name in ('fit', 'self', 'greedy')]
    trained = str(out / 'train/checkpoint.pt')
    timings_path = out / 'timings.json'
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    # Prevent mixing completed outputs with a different recipe or replaced input.
    recipe = {k: v for k, v in os.environ.items() if k in {
        'COLLECT_POSITIONS_PER_S', 'SCHEDULE_OVERHEAD_HOURS', 'DEVICE', 'NUM_ENVS', 'THREADS', 'TF32', 'FIT_POSITIONS', 'VALIDATION_POSITIONS',
        'DIAGNOSTIC_POSITIONS', 'DIAGNOSTIC_SAMPLES', 'BOOTSTRAP', 'BATCH_SIZE', 'MICROBATCH',
        'MAX_HOURS', 'STEPS', 'REPLAY_RATIO', 'BUFFER_SIZE', 'WARMUP_POSITIONS', 'WARMUP_STEPS',
        'EVAL_EVERY', 'SAVE_EVERY', 'LOG_EVERY', 'EVAL_BATCH_SIZE', 'PRESENCE_SAMPLES',
        'PRESENCE_SAMPLE_BATCH_SIZE', 'PAIRS', 'WORLDS', 'DEPTH', 'CANDIDATES', 'CANDIDATE_SAMPLES',
        'MAX_INFERENCE_BATCH', 'SEARCH_CACHE_MB', 'LR', 'PROGRESS_INTERVAL'}}
    recipe.update(policy=policy, belief=old, steps=steps, max_hours=hours,
                  inputs={p: dict(size=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns)
                          for p in (policy, old)})
    manifest = out / 'session-config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != recipe:
        raise ValueError('session configuration/input mismatch; use a new OUT')
    manifest.write_text(json.dumps(recipe, indent=2)+'\n')

    def atomic_json(path, data):
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2)+'\n')
        temp.replace(path)

    def run(name, script, args, output, *, train=False):
        output = Path(output)
        complete = output.exists()
        if complete and output.suffix == '.json':
            try:
                json.loads(output.read_text())
                complete = output.with_suffix('.md').exists()
            except (ValueError, OSError):
                complete = False
        if complete and train:
            complete = Path(trained).exists()
        if complete:
            print(f'Skip {name}: {output}', flush=True)
            return
        command = [sys.executable, f'scripts/{script}.py', *map(str, args)]
        print('Run ' + ' '.join(command), flush=True)
        start = perf_counter()
        with (out / f'{name}.log').open('a') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        timings[name] = timings.get(name, 0) + perf_counter()-start
        atomic_json(timings_path, timings)
        if code:
            raise subprocess.CalledProcessError(code, command)
        if train:
            output.write_text('Training invocation completed (steps or time limit).\n')
        if not output.exists():
            raise RuntimeError(f'{name} did not produce {output}')

    for name, opponent, positions, seed in (
            ('fit', 'self', env('FIT_POSITIONS', 1000000), 111),
            ('self', 'self', env('VALIDATION_POSITIONS', 20000), 212),
            ('greedy', 'greedy', env('VALIDATION_POSITIONS', 20000), 212)):
        target = out / f'{name}.pt'
        # Publish only complete data; interrupted collection is regenerated.
        temporary = out / f'{name}.collecting.pt'
        if target.exists():
            continue
        temporary.unlink(missing_ok=True)
        run(name, 'make_belief_data', [*common, *tf32, '--checkpoint', policy,
            '--pool-fraction', .2, '--opponent', opponent, '--positions', positions,
            '--sample-prob', .1, '--seed', seed, '--out', temporary,
            *(['--labels-only'] if name == 'fit' else [])], temporary)
        temporary.replace(target)
    training = [*common, *tf32, '--checkpoint', policy, '--run-dir', out / 'train',
                '--fit', fit, '--validation', self_data, greedy, '--pool-fraction', .2,
                '--sample-prob', .25, '--steps', steps, '--max-hours', hours]
    defaults = dict(lr=.0001, warmup_steps=200, replay_ratio=ratio, buffer_size=200000,
                    warmup_positions=50000, batch_size=batch, microbatch=128, eval_every=500,
                    save_every=500, log_every=50, eval_batch_size=64, presence_samples=32)
    for key, default in defaults.items():
        training += ['--'+key.replace('_', '-'), env(key.upper(), default)]
    sampling = ['--presence-sample-batch-size', env('PRESENCE_SAMPLE_BATCH_SIZE', 32)]
    training += sampling
    training += ['--resume'] if Path(trained).exists() else ['--init-belief', old]
    run('train', 'train_belief', training, out / 'train.done', train=True)
    run('likelihood', 'eval_belief', ['--device', env('DEVICE', 'cuda'), *tf32,
        '--threads', env('THREADS', 4), '--checkpoint', trained, old,
        '--validation', self_data, greedy, '--fit', fit, '--out', out / 'likelihood.json',
        '--eval-batch-size', env('EVAL_BATCH_SIZE', 64),
        '--presence-samples', env('PRESENCE_SAMPLES', 32), *sampling], out / 'likelihood.json')
    for name, belief in (('new', trained), ('old', old)):
        run(f'diagnostic-{name}', 'belief_diagnostic', [*common, '--policy', policy,
            '--belief', belief, '--fit', fit, '--positions', env('DIAGNOSTIC_POSITIONS', 3000),
            '--samples', env('DIAGNOSTIC_SAMPLES', 64), '--bootstrap', env('BOOTSTRAP', 1000),
            '--max-inference-batch', env('MAX_INFERENCE_BATCH', 8192),
            '--out', out / f'diagnostic-{name}.json'], out / f'diagnostic-{name}.json')
    search = [*common, *tf32, '--checkpoint', policy, '--opponent', 'policy', '--fit', fit,
              '--pairs', env('PAIRS', 1000), '--seed', 74100,
              '--alpha', .002, '--beta', .1, '--leaf-view', 'actor', '--search-response',
              '--skip-forced-inference', '--skip-single-candidate',
              '--search-cache-device', 'auto', '--search-cache-mb', env('SEARCH_CACHE_MB', 8192),
              '--progress-interval', env('PROGRESS_INTERVAL', 600)]
    for key, default in dict(worlds=16, depth=3, candidates=16, candidate_samples=64,
                             max_inference_batch=8192).items():
        search += ['--'+key.replace('_', '-'), env(key.upper(), default)]
    for name, belief, opponent in (('new-vs-policy', trained, None),
                                  ('old-vs-policy', old, None), ('new-vs-old', trained, old)):
        run(name, 'eval_search', [*search, '--belief', belief,
            *(['--opponent-belief', opponent] if opponent else []),
            '--out', out / f'{name}.json'], out / f'{name}.json')
    lines = ['# m-2166 belief 再学習', '',
             f'学習上限 {hours} h / {steps} steps。pool fraction は学習・fit・検証とも 0.2。',
             f'cosine 総ステップ = floor(((hours − overhead_hours) × 3600 × rate − warmup_positions) × replay_ratio / batch_size)、rate={rate}/s。',
             '290 positions/s・ratio 2・batch 1024 は約 2039 steps/h（3300 ではない）。更新・検証の所要時間は別途差し引く。',
             '探索は共通 seed 74100 の paired deals、引き分けは 0.5 勝。CPU smoke は精度判定には使わない。', '',
             '| Phase | Wall seconds (completed attempts) |', '|---|---:|']
    lines += [f'| {k} | {v:.2f} |' for k, v in timings.items()]
    metrics = [json.loads(line) for line in (out / 'train/metrics.jsonl').read_text().splitlines()]
    lines += ['', '## 学習検証', '', '| Step | Data | Model | Multiset NLL | Count NLL |', '|---:|---|---|---:|---:|']
    for row in metrics:
        for data, results in row.get('evaluation', {}).items():
            for name, result in results.items():
                lines.append(f'| {row["step"]} | {Path(data).stem} | {name} | {result["multiset_nll"]:.4f} | {result["count_nll"]:.4f} |')
    for name in ('likelihood', 'diagnostic-new', 'diagnostic-old', 'new-vs-policy', 'old-vs-policy', 'new-vs-old'):
        lines += ['', f'## {name}', '', (out / f'{name}.md').read_text()]
    (out / 'summary.md').write_text('\n'.join(lines)+'\n')
    print(f'Summary: {out / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
