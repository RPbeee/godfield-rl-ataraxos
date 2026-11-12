"""Viewer-relative distributional lambda returns and terminal-reward GAE."""
import numpy as np
import torch


def sequence_targets(probabilities, outcome, lambda_v=0.8, lambda_a=0.5, gamma=1.0):
    """Discount once per own decision; classes are win/loss/draw.

    Truncations must pass outcome=0. With terminal p_T = target_T =
    onehot(outcome), target_t = gamma*((1-lambda_v)*p_{t+1} +
    lambda_v*target_{t+1}) + (1-gamma)*onehot(draw). Mixing towards
    draw makes E[target] the discounted scalar lambda-return.
    GAE uses delta_t = gamma*V_{t+1}-V_t and gamma*lambda_a recursion.
    """
    p = np.asarray(probabilities, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3 or outcome not in (-1, 0, 1):
        raise ValueError('expected [T,3] probabilities and outcome in {-1,0,1}')
    if not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0,1]')
    if not 0 <= lambda_v <= 1 or not 0 <= lambda_a <= 1:
        raise ValueError('lambda must be in [0,1]')
    target = np.zeros_like(p)
    advantage = np.zeros(len(p), dtype=np.float32)
    terminal = np.eye(3, dtype=np.float32)[{1: 0, -1: 1, 0: 2}[outcome]]
    next_p = next_target = terminal
    next_value, next_advantage = float(outcome), 0.0
    for t in range(len(p) - 1, -1, -1):
        target[t] = (1 - lambda_v) * next_p + lambda_v * next_target
        if gamma != 1:
            target[t] *= gamma
            target[t, 2] += 1-gamma
        value = p[t, 0] - p[t, 1]
        advantage[t] = next_value - value + lambda_a * next_advantage
        if gamma != 1:
            advantage[t] = gamma*next_value - value + gamma*lambda_a*next_advantage
        next_p, next_target = p[t], target[t]
        next_value, next_advantage = value, advantage[t]
    return target, advantage


def compute_targets(decisions, lambda_v=0.8, lambda_a=0.5, gamma=1.0):
    """Preserve arrival order within (env, game, seat), skipping opponent turns.

    Discount per own decision with the draw-mixture lambda-return and GAE
    defined in sequence_targets, including terminal bootstraps.
    """
    if not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0,1]')
    if not 0 <= lambda_v <= 1 or not 0 <= lambda_a <= 1:
        raise ValueError('lambda must be in [0,1]')
    if hasattr(decisions, 'columns'):
        # Only these small columns need host access. CPU rollout storage
        # already has them here; device storage transfers no observations.
        c = {k: (decisions.columns[k].cpu().numpy() if torch.is_tensor(decisions.columns[k])
                 else decisions.columns[k])
             for k in ('env', 'game', 'seat', 'old_value', 'outcome')}
    else:
        # Small hand-calculated examples and legacy callers only.
        c = {k: np.asarray([getattr(d,k) for d in decisions])
             for k in ('env', 'game', 'seat', 'old_value', 'outcome')}
    n = len(decisions)
    target = np.empty((n,3), dtype=np.float32)
    advantage = np.empty(n, dtype=np.float32)
    if not n:
        return target, advantage
    order = np.lexsort((np.arange(n), c['seat'], c['game'], c['env']))
    starts = np.r_[0, 1+np.flatnonzero(
        (c['env'][order][1:] != c['env'][order][:-1]) |
        (c['game'][order][1:] != c['game'][order][:-1]) |
        (c['seat'][order][1:] != c['seat'][order][:-1]))]
    ends = np.r_[starts[1:], n]
    lengths = ends-starts
    outcome = c['outcome'][order[ends-1]]
    p = np.asarray(c['old_value'], dtype=np.float32)
    terminal = np.eye(3, dtype=np.float32)[np.where(outcome == 1, 0, np.where(outcome == -1, 1, 2))]
    next_p, next_target = terminal.copy(), terminal.copy()
    next_value = outcome.astype(np.float32)
    next_advantage = np.zeros(len(starts), dtype=np.float32)
    # One Python iteration per sequence depth, independent of env/decision count.
    for depth in range(int(lengths.max())):
        groups = np.flatnonzero(lengths > depth)
        rows = order[ends[groups]-1-depth]
        target[rows] = (1-lambda_v)*next_p[groups]+lambda_v*next_target[groups]
        if gamma != 1:
            target[rows] *= gamma
            target[rows, 2] += 1-gamma
        value = p[rows,0]-p[rows,1]
        advantage[rows] = next_value[groups]-value+lambda_a*next_advantage[groups]
        if gamma != 1:
            advantage[rows] = gamma*next_value[groups]-value+gamma*lambda_a*next_advantage[groups]
        next_p[groups], next_target[groups] = p[rows], target[rows]
        next_value[groups], next_advantage[groups] = value, advantage[rows]
    return target, advantage
