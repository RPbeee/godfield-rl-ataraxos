"""Paired-seed evaluation; truncated games count as draws."""
import math

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


def summarize(results):
    """Summarize agent-relative game records, giving draws half credit."""
    results = list(results)
    n = len(results)
    if any(r['result'] not in (-1, 0, 1) for r in results):
        raise ValueError("result must be -1, 0 or 1")
    wins = sum(r['result'] == 1 for r in results)
    losses = sum(r['result'] == -1 for r in results)
    draws = n - wins - losses
    def rate(rows):
        return sum((r['result'] + 1) / 2 for r in rows) / len(rows) if rows else None
    return {
        'games': n, 'win_rate': rate(results),
        'wilson_interval': wilson_interval(wins + draws / 2, n),
        'bootstrap_interval': pair_bootstrap(results),
        'wins': wins, 'losses': losses, 'draws': draws,
        'truncation_rate': sum(r['truncated'] for r in results) / n if n else None,
        'mean_decisions': sum(r['decisions'] for r in results) / n if n else None,
        'first_win_rate': rate([r for r in results if r['agent_seat'] == 0]),
        'second_win_rate': rate([r for r in results if r['agent_seat'] == 1]),
    }


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
