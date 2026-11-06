"""Viewer-relative distributional lambda returns and terminal-reward GAE."""
import numpy as np


def sequence_targets(probabilities, outcome, lambda_v=0.8, lambda_a=0.5):
    """Classes are win/loss/draw. Truncations must pass outcome=0."""
    p = np.asarray(probabilities, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3 or outcome not in (-1, 0, 1):
        raise ValueError('expected [T,3] probabilities and outcome in {-1,0,1}')
    if not 0 <= lambda_v <= 1 or not 0 <= lambda_a <= 1:
        raise ValueError('lambda must be in [0,1]')
    target = np.zeros_like(p)
    advantage = np.zeros(len(p), dtype=np.float32)
    terminal = np.eye(3, dtype=np.float32)[{1: 0, -1: 1, 0: 2}[outcome]]
    next_p = next_target = terminal
    next_value, next_advantage = float(outcome), 0.0
    for t in range(len(p) - 1, -1, -1):
        target[t] = (1 - lambda_v) * next_p + lambda_v * next_target
        value = p[t, 0] - p[t, 1]
        advantage[t] = next_value - value + lambda_a * next_advantage
        next_p, next_target = p[t], target[t]
        next_value, next_advantage = value, advantage[t]
    return target, advantage


def compute_targets(decisions, lambda_v=0.8, lambda_a=0.5):
    """Preserve arrival order within each (env, game, seat), skip opponent turns."""
    groups = {}
    for i, d in enumerate(decisions):
        groups.setdefault((d.env, d.game, d.seat), []).append(i)
    targets = np.empty((len(decisions), 3), dtype=np.float32)
    advantages = np.empty(len(decisions), dtype=np.float32)
    for rows in groups.values():
        target, advantage = sequence_targets([decisions[i].old_value for i in rows],
                                             decisions[rows[-1]].outcome, lambda_v, lambda_a)
        targets[rows], advantages[rows] = target, advantage
    return targets, advantages
