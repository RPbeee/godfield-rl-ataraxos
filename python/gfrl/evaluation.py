"""Paired-seed evaluation; truncated games count as draws."""
from contextlib import contextmanager
import math
from time import monotonic

import numpy as np

from . import _core


def wilson_interval(wins, n, z=1.96):
    """Wilson interval with draws counted as 0.5 wins in `wins` (effective wins).

    Return (0, 1) for an empty sample. This is a score interval approximation
    when fractional wins represent draws rather than Bernoulli observations.
    """
    if n < 0 or not 0 <= wins <= n or z <= 0:
        raise ValueError("require 0 <= wins <= n and z > 0")
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pair_bootstrap(results, iters=2000, seed=0):
    """95% percentile interval, resampling complete two-game pairs as clusters.

    A win scores 1, a draw 0.5 and a loss 0. Empty input returns (0, 1).
    Incomplete or duplicate pairs are rejected to avoid silently biasing scores.
    """
    if iters <= 0:
        raise ValueError("iters must be positive")
    groups = {}
    for r in results:
        pair = groups.setdefault(r['pair'], {})
        if r['agent_seat'] in pair or r['agent_seat'] not in (0, 1):
            raise ValueError("duplicate or invalid seat in pair")
        pair[r['agent_seat']] = (r['result'] + 1) / 2
    if not groups:
        return (0.0, 1.0)
    if any(set(pair) != {0, 1} for pair in groups.values()):
        raise ValueError("bootstrap requires complete pairs")
    values = np.array([sum(groups[p].values()) / 2 for p in sorted(groups)])
    rng = np.random.default_rng(seed)
    means = np.empty(iters)
    # Bound peak memory independently of the pair count and iteration count.
    batch = max(1, min(128, 1_000_000 // len(values)))
    for start in range(0, iters, batch):
        count = min(batch, iters - start)
        means[start:start + count] = values[rng.integers(len(values), size=(count, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize(results, *, complete_pairs_only=False):
    """Summarize agent-relative game records, giving draws half credit.

    complete_pairs_only permits partial evaluations: only the bootstrap uses
    complete pairs; every other statistic includes all completed games.
    """
    results = list(results)
    n = len(results)
    if any(r['result'] not in (-1, 0, 1) for r in results):
        raise ValueError("result must be -1, 0 or 1")
    wins = sum(r['result'] == 1 for r in results)
    losses = sum(r['result'] == -1 for r in results)
    draws = n - wins - losses
    bootstrap_results = results
    if complete_pairs_only:
        seats = {}
        for r in results:
            seats.setdefault(r['pair'], []).append(r['agent_seat'])
        complete = {p for p, s in seats.items() if sorted(s) == [0, 1]}
        bootstrap_results = [r for r in results if r['pair'] in complete]
    def rate(rows):
        return sum((r['result'] + 1) / 2 for r in rows) / len(rows) if rows else None
    return {
        'games': n, 'win_rate': rate(results),
        'wilson_interval': wilson_interval(wins + draws / 2, n),
        'bootstrap_interval': pair_bootstrap(bootstrap_results),
        'wins': wins, 'losses': losses, 'draws': draws,
        'truncation_rate': sum(r['truncated'] for r in results) / n if n else None,
        'mean_decisions': sum(r['decisions'] for r in results) / n if n else None,
        'first_win_rate': rate([r for r in results if r['agent_seat'] == 0]),
        'second_win_rate': rate([r for r in results if r['agent_seat'] == 1]),
    }


class _EvaluationProgress:
    """Read completed records only at reporting boundaries; never sample RNGs."""

    def __init__(self, callback, interval_s):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError('progress_interval_s must be finite and positive')
        self.callback = callback
        self.interval_s = interval_s
        self.last_time = monotonic()
        self.last_games = None

    def __call__(self, env, *, final=False):
        now = monotonic()
        if not final and now - self.last_time < self.interval_s:
            return
        # eval_results() already retains all completed games, including those
        # whose env slots have been reset/reused. No C++ accessor is needed.
        results = env.eval_results()
        if not final or len(results) != self.last_games:
            self.callback(summarize(results, complete_pairs_only=True))
            self.last_games = len(results)
        self.last_time = monotonic()


def evaluate_policy(policy_fn, opponent, pairs, num_envs, num_threads, eval_seed):
    """Run policy_fn(env) -> int32 choices; choices for idle rows are ignored.

    Only stage >= 0 rows require actions. In evaluation, env.episode_id is the
    global game number g. A stateful stochastic policy should
    maintain its own per-game RNG to preserve reproducibility across batch sizes.
    """
    env = _core.VecEnv(num_envs, num_threads, eval_pairs=pairs,
                       opponent=opponent, eval_seed=eval_seed)
    while not env.eval_done():
        env.step(np.asarray(policy_fn(env), dtype=np.int32))
    return summarize(env.eval_results())


def evaluate_bots(bot_a, bot_b, pairs, threads, eval_seed):
    """Evaluate two C++ bots over paired deals and summarize agent=bot_a."""
    return summarize(_core.evaluate_bots(bot_a, bot_b, pairs, threads, eval_seed))


def evaluate_policies(policy_a, policy_b, pairs, num_envs, num_threads, eval_seed, *,
                      track_projection=False, progress=None, progress_interval_s=600):
    """Evaluate A against B on paired deals, returning A-relative statistics.

    Callbacks receive (env, rows) and return int32 choices of shape [num_envs],
    as Actor does. Only selected active rows are used; empty groups are skipped.
    Actor sampling uses an isolated, seeded Torch RNG, restored on return.
    track_projection enables search roots and defaults to the existing fast path.
    Results are reproducible for the same batching and devices; stochastic
    results need not be invariant to num_envs.
    progress(summary), when provided, runs between steps at the requested wall
    time interval and at completion. Only finished games are included; bootstrap
    uses complete pairs. Reporting uses a local NumPy generator, leaving policy
    RNGs untouched. The callback itself must not consume policy RNGs.
    """
    reporter = _EvaluationProgress(progress, progress_interval_s) if progress is not None else None
    with _policy_rng(eval_seed, (policy_a, policy_b)):
        env = _core.VecEnv(num_envs, num_threads, eval_pairs=pairs,
                           opponent='external', eval_seed=eval_seed,
                           track_projection=track_projection)
        while not env.eval_done():
            active = env.stage >= 0
            a_turn = env.actor_seat == env.episode_id % 2
            choices = np.zeros(num_envs, dtype=np.int32)
            for policy, turn in ((policy_a, a_turn), (policy_b, ~a_turn)):
                rows = np.flatnonzero(active & turn)
                if len(rows):
                    selected = np.asarray(policy(env, rows), dtype=np.int32)
                    if selected.shape != choices.shape:
                        raise ValueError('policy must return choices of shape [num_envs]')
                    choices[rows] = selected[rows]
            env.step(choices)
            if reporter is not None:
                reporter(env)
        if reporter is not None:
            reporter(env, final=True)
        return summarize(env.eval_results())


@contextmanager
def _policy_rng(seed, policies):
    """Isolate new evaluation sampling without changing existing callers."""
    import torch
    devices = {torch.device(p.device) for p in policies if hasattr(p, 'device')}
    cuda = sorted({d.index if d.index is not None else torch.cuda.current_device()
                   for d in devices if d.type == 'cuda'})
    use_mps = any(d.type == 'mps' for d in devices)
    mps_state = torch.mps.get_rng_state() if use_mps else None
    with torch.random.fork_rng(devices=cuda):
        torch.random.default_generator.manual_seed(seed)
        for device in cuda:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        if use_mps:
            torch.mps.manual_seed(seed)
        try:
            yield
        finally:
            if use_mps:
                torch.mps.set_rng_state(mps_state)


def _checkpoint_spec(spec):
    path, marker, iteration = str(spec).rpartition('@pool:')
    if not marker:
        return str(spec), None
    try:
        iteration = int(iteration)
    except ValueError as exc:
        raise ValueError('pool selector must be @pool:<integer>') from exc
    return path, iteration


def list_pool_iterations(spec):
    """Return saved pool iterations in checkpoint order (empty for no pool)."""
    import torch
    path, _ = _checkpoint_spec(spec)
    state = torch.load(path, map_location='cpu', weights_only=False)
    return [entry['iteration'] for entry in state.get('pool', ())]


def load_policy_model(spec, device='cpu'):
    """Load checkpoint EMA, or checkpoint.pt@pool:N, using its model_config."""
    import torch
    from .model import Model, ModelConfig
    path, iteration = _checkpoint_spec(spec)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if iteration is None:
        weights = state['ema']
    else:
        entries = [e for e in state.get('pool', ()) if e['iteration'] == iteration]
        if not entries:
            available = [e['iteration'] for e in state.get('pool', ())]
            raise ValueError(f'pool iteration {iteration} not found in {path}; available: {available}')
        weights = entries[0]['weights']
    model = Model(ModelConfig(**state['model_config']))
    model.load_state_dict(weights)
    return model.to(device).eval().requires_grad_(False)
