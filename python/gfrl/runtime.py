"""Execution settings; deliberately excluded from model/checkpoint parameters."""
import argparse
from contextlib import contextmanager
from time import perf_counter

import torch

ROW_BUCKETS = (32, 64, 128, 256)
WIDTH_BUCKETS = tuple(range(16, 129, 16))
COMPILE_MODES = ('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs')


def add_runtime_args(parser):
    parser.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile', choices=('off', 'on'), default='off')
    parser.add_argument('--compile-mode-actor', choices=COMPILE_MODES, default='reduce-overhead')
    parser.add_argument('--compile-mode-learner', choices=COMPILE_MODES, default='default')


def configure_precision(device, tf32=True):
    if torch.device(device).type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32


def row_bucket(rows):
    if not 1 <= rows <= 256:
        raise ValueError('compiled encoder chunks must have 1..256 rows')
    return next(n for n in ROW_BUCKETS if rows <= n)


def encoder_bucket_shape(rows, width):
    if not 1 <= width <= 128:
        raise ValueError('compiled encoder supports 1..128 tokens')
    return row_bucket(rows), ((width + 15) // 16) * 16


@contextmanager
def encoder_fastpath_disabled():
    # The fused eval-only Transformer fast path has different compiler support
    # across torch versions. Trace the ordinary attention/FFN in both roles.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


class CompiledEncoder:
    def __init__(self, encoder, device, actor_mode, learner_mode):
        import torch._dynamo.config as config
        # 32 shapes x inference/no-grad/grad, live + EMA; leave room for dtypes.
        config.cache_size_limit = max(config.cache_size_limit, 256)
        config.accumulated_cache_size_limit = max(config.accumulated_cache_size_limit, 1024)
        from torch._dynamo import utils
        if hasattr(utils, 'set_compilation_metrics_limit'):
            # Keep all finite bucket specializations available for accounting.
            utils.set_compilation_metrics_limit(4096)
        self.device = torch.device(device)
        self.modes = dict(actor=actor_mode, learner=learner_mode)
        if self.device.type == 'cpu':
            self.modes = dict(actor='default', learner='default')
        self.calls = {}
        self.encoder = encoder

    def __deepcopy__(self, memo):
        # Pool opponents are transient snapshots and must remain eager. Never
        # deepcopy a compiled callable that closes over another model's weights.
        return None

    def __call__(self, packed, padding):
        role = 'learner' if torch.is_grad_enabled() else 'actor'
        if role not in self.calls:
            self.calls[role] = torch.compile(self.encoder, mode=self.modes[role], dynamic=False)
        with encoder_fastpath_disabled():
            if role == 'actor' and self.device.type == 'cuda':
                torch.compiler.cudagraph_mark_step_begin()
            output = self.calls[role](packed, src_key_padding_mask=padding)
            # Outputs can survive another chunk, act(), or an optimizer/EMA
            # update. Copy outside the captured graph before the next replay.
            if role == 'actor' and self.device.type == 'cuda':
                output = output.clone()
        return output


def configure_compile(model, device, compile='off', *, actor_mode='reduce-overhead',
                      learner_mode='default'):
    device = torch.device(device)
    if compile not in ('off', 'on'):
        raise ValueError('compile must be off or on')
    if compile == 'on' and device.type == 'mps':
        raise ValueError('torch.compile is unsupported on MPS; use --compile off')
    model._compiled_encoder = (CompiledEncoder(model.encoder, device, actor_mode, learner_mode)
                               if compile == 'on' else None)
    return model


def configure_model(model, args):
    return configure_compile(model, args.device, args.compile,
                             actor_mode=args.compile_mode_actor,
                             learner_mode=args.compile_mode_learner)


def compiler_metrics():
    """Process-wide compiler counters, best effort across supported torch versions.

    compile_s sums outer Dynamo compilation and lazy backward compilation;
    it excludes execution/CUDA Graph capture. New shape specializations count
    as recompiles when they fail guards, as in Dynamo's recompilation logs.
    """
    from torch._dynamo import utils
    times = utils.compilation_time_metrics
    outer = times.get('_compile.compile_inner', times.get('compile_attempt_0', ()))
    seconds = sum(outer) + sum(times.get('compile_fx.<locals>.bw_compiler', ()))
    events = utils.get_compilation_metrics()
    recompiles = sum(
        (bool(getattr(event, 'recompile_reason', None)) or (getattr(event, 'cache_size', 0) or 0) > 0)
        and getattr(event, 'is_forward', True) is not False
        and getattr(event, 'has_guarded_code', None) is not False
        and getattr(event, 'fail_type', None) is None
        for event in events)
    return dict(compile_s=seconds,
                compile_graphs=int(utils.counters['stats']['unique_graphs']),
                compile_recompiles=recompiles,
                compile_guard_failures=sum(len(v) for v in utils.guard_failures.values()))


def metric_delta(before):
    return {key: value-before[key] for key, value in compiler_metrics().items()}


def warmup_encoder(model, device, *, training=False, dtype='fp32', all_buckets=True):
    """Compile every encoder shape, outside steady-state benchmark timing.

    Heads stay eager. CPU callers may choose observed-only warmup to keep quick
    smoke benchmarks small; any remaining compilation is reported separately.
    """
    from contextlib import nullcontext
    from .actor import synchronize
    if model._compiled_encoder is None or not all_buckets:
        return 0.
    device = torch.device(device)
    synchronize(device)
    start = perf_counter()
    for rows in ROW_BUCKETS:
        for width in WIDTH_BUCKETS:
            grad = torch.enable_grad() if training else torch.inference_mode()
            autocast = (nullcontext() if dtype == 'fp32' else torch.autocast(
                device.type, dtype={'fp16': torch.float16, 'bf16': torch.bfloat16}[dtype]))
            with grad, autocast:
                x = torch.zeros(rows, width, model.config.d, device=device,
                                requires_grad=training)
                padding = torch.zeros(rows, width, device=device, dtype=torch.bool)
                # CUDA Graph capture needs warmup/capture/replay invocations,
                # in addition to the initial lazy Dynamo/Inductor compilation.
                repeats = 4 if device.type == 'cuda' else 1
                for _ in range(repeats):
                    y = model._compiled_encoder(x, padding)
                    if training:
                        y.square().mean().backward()
                        model.zero_grad(set_to_none=True)
    synchronize(device)
    return perf_counter()-start
