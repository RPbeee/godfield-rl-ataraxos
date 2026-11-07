"""Small shared benchmark reporting helpers."""
from argparse import Namespace
from itertools import product


def sweep_cases(args):
    if not args.sweep:
        yield args
        return
    actor_modes = ('default',) if args.device == 'cpu' else ('default', 'reduce-overhead')
    if hasattr(args, 'games'):
        learner_modes = actor_modes
        actor_modes = ('default',) if args.device == 'cpu' else (args.compile_mode_actor,)
    else:
        learner_modes = (args.compile_mode_learner,)
    for tf32 in (False, True):
        yield Namespace(**{**vars(args), 'tf32': tf32, 'compile': 'off'})
        for actor_mode, learner_mode in product(actor_modes, learner_modes):
            yield Namespace(**{**vars(args), 'tf32': tf32, 'compile': 'on',
                               'compile_mode_actor': actor_mode,
                               'compile_mode_learner': learner_mode})


def print_table(results, rate):
    print('| TF32 | Compile | Actor mode | Learner mode | Warmup s | Compiler s (warmup / measured) | Decisions/s |')
    print('|---|---|---|---|---:|---:|---:|')
    for r in results:
        print(f"| {r['tf32']} | {r['compile']} | {r['compile_mode_actor']} | "
              f"{r.get('compile_mode_learner', '—')} | {r['warmup_s']:.3f} | "
              f"{r['warmup_compiler']['compile_s']:.3f} / "
              f"{r['measured_compiler']['compile_s']:.3f} | {r[rate]:.1f} |")
