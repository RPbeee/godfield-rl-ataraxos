"""Fixed-policy belief validation data. Labels never enter the policy allowlist.

Files are torch.save dictionaries of CPU tensors plus a `config` dictionary.
Decision numbers are engine decisions (including automatic/bot decisions), zero
based within each game; game_id = episode_id * num_envs + environment row.
Opponent codes describe the opponent from the recorded viewer's perspective.
pool_index is -1 for non-pool opponents, otherwise a zero-based index into
config.policy_specs[1:]. Episode routing is fixed: seat 0 latest, seat 1 mixed.
Greedy games expose only seat 0 to Python. Seeds are reproducible with the same
checkpoint, device and batching; different num_envs need not reproduce the data.
"""
import numpy as np
import torch

from . import _core
from .actor import Actor
from .evaluation import _policy_rng, list_pool_iterations, load_policy_model
from .model import OBS_FIELDS

OPPONENT_TYPES = {'self': 0, 'pool': 1, 'greedy': 2}


def collect_positions(policies, *, positions, opponent='self', pool_fraction=.2,
                      sample_prob=.1, labels_only=False, seed=0, num_envs=64,
                      threads=1, on_record=None):
    """Collect exactly positions rows; policies = [latest, *uniform pool].

    Optional on_record(env, rows, saved_slice) is a test hook, invoked before
    actions. Determinism requires the same batching, device and policies.
    Buffers are preallocated once; there is no list/concatenation copy.
    """
    if positions <= 0 or num_envs <= 0 or threads <= 0:
        raise ValueError('positions, num_envs and threads must be positive')
    if opponent not in ('self', 'greedy') or not 0 <= pool_fraction <= 1 or not 0 < sample_prob <= 1:
        raise ValueError('invalid opponent, pool_fraction or sample_prob')
    if not policies or (opponent == 'self' and pool_fraction > 0 and len(policies) < 2):
        raise ValueError('self with pool_fraction > 0 requires a saved pool')
    env = _core.VecEnv(num_envs, threads, seed,
                       controllers=('external', 'external' if opponent == 'self' else 'greedy'))
    fields = ('label_opp_hand', 'label_opp_hand_count') + (() if labels_only else OBS_FIELDS)
    data = {}
    for name in fields:
        a = getattr(env, name)
        dtype = np.float16 if name == 'obs_num' else a.dtype
        data[name] = torch.from_numpy(np.empty((positions, *a.shape[1:]), dtype=dtype))
    for key, dtype in (('game_id', torch.int64), ('seat', torch.int8),
                       ('decision', torch.int32), ('opponent_type', torch.int8),
                       ('pool_index', torch.int32)):
        data[key] = torch.empty(positions, dtype=dtype)
    arrays = {k: v.numpy() for k, v in data.items()}
    sampling_rng, routing_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    previous = np.full(num_envs, -1, dtype=np.int64)
    route = np.zeros(num_envs, dtype=np.int32)
    written = 0
    decision_index = env.obs_schema()['num']['decisions']
    with _policy_rng(seed, policies):
        while written < positions:
            games = env.episode_id.astype(np.int64) * num_envs + np.arange(num_envs)
            for r in np.flatnonzero(games != previous):
                route[r] = (routing_rng.integers(1, len(policies))
                            if opponent == 'self' and routing_rng.random() < pool_fraction else 0)
            previous[:] = games
            active = env.stage >= 0
            fresh = np.flatnonzero(active & (env.substep == 0))
            selected = fresh[sampling_rng.random(len(fresh)) < sample_prob][:positions-written]
            if len(selected):
                dest = slice(written, written + len(selected))
                for name in fields:
                    arrays[name][dest] = getattr(env, name)[selected]
                arrays['game_id'][dest] = games[selected]
                arrays['seat'][dest] = env.actor_seat[selected]
                arrays['decision'][dest] = np.rint(env.obs_num[selected, 0, decision_index] * 512).astype(np.int32)
                other_pool = (env.actor_seat[selected] == 0) & (route[selected] > 0)
                arrays['opponent_type'][dest] = (2 if opponent == 'greedy' else other_pool.astype(np.int8))
                arrays['pool_index'][dest] = np.where(other_pool, route[selected] - 1, -1)
                if on_record is not None:
                    on_record(env, selected, {k: v[dest] for k, v in arrays.items()})
                written += len(selected)
            if written == positions:
                break
            choices = np.zeros(num_envs, dtype=np.int32)
            turn_route = np.where(env.actor_seat == 0, 0, route)
            for index in np.unique(turn_route[active]):
                rows = np.flatnonzero(active & (turn_route == index))
                choices[rows] = policies[index](env, rows)[rows]
            env.step(choices)
    data['config'] = dict(format_version=1, positions=positions, opponent=opponent,
                          pool_fraction=pool_fraction, sample_prob=sample_prob,
                          labels_only=labels_only, seed=seed, num_envs=num_envs,
                          threads=threads, opponent_types=OPPONENT_TYPES,
                          obs_schema=env.obs_schema())
    return data


def make_belief_data(checkpoint, *, device='cpu', **kwargs):
    specs = [str(checkpoint)]
    if kwargs.get('opponent', 'self') == 'self' and kwargs.get('pool_fraction', .2) > 0:
        specs += [f'{checkpoint}@pool:{n}' for n in list_pool_iterations(checkpoint)]
    idle = _core.VecEnv(1, 1, eval_pairs=0)
    policies = [Actor(load_policy_model(s, device), idle, device) for s in specs]
    data = collect_positions(policies, **kwargs)
    data['config'].update(checkpoint=str(checkpoint), policy_specs=specs, device=str(device))
    return data
