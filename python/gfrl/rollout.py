"""CPU trajectory records and batched self-play with pinned decision policies."""
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from . import VecEnv
from .actor import Actor, PREFIX_FIELDS
from .model import OBS_FIELDS


@dataclass
class Stage:
    prefix: dict
    choice: int
    old_log_probs: np.ndarray
    decision: int = -1


@dataclass
class Decision:
    obs: dict
    env: int
    seat: int
    game: int
    old_value: np.ndarray
    stages: list = field(default_factory=list)
    outcome: int = 0


@dataclass
class Rollout:
    decisions: list
    games: int
    truncated: int
    sampled_decisions: int


class Collector:
    """Carry unfinished games across collect calls; emit only completed games.

    pool entries are dictionaries containing an immutable CPU state_dict and id.
    A retired pool policy remains alive until its already assigned games finish.
    """
    def __init__(self, model, *, num_envs=64, threads=4, seed=0, device='cpu',
                 dtype='fp32', pool_fraction=0.2, env=None):
        if not 0 <= pool_fraction <= 1:
            raise ValueError('pool_fraction must be in [0,1]')
        self.env = env if env is not None else VecEnv(num_envs, threads, seed,
                                                   controllers=('external', 'external'))
        self.model, self.device, self.dtype = model, device, dtype
        self.rng = np.random.default_rng(seed)
        self.pool_fraction = pool_fraction
        self.actors = {}
        self.opponents = {}  # env -> (seat, policy key), assigned per game
        self.routing = {}    # env -> policy key, pinned until decision completes
        self.pending = [[] for _ in range(self.env.num_envs)]
        self.current = {}
        self.serial = 0

    def _actor(self, key, weights=None):
        if key not in self.actors:
            model = deepcopy(self.model)
            if weights is not None:
                model.load_state_dict(weights)
            self.actors[key] = Actor(model, self.env, self.device, dtype=self.dtype)
        return self.actors[key]

    def collect(self, games_per_iter, pool=()):
        if games_per_iter <= 0:
            raise ValueError('games_per_iter must be positive')
        self.serial += 1
        live = ('live', self.serial)
        self._actor(live)
        completed, games, truncated, sampled = [], 0, 0, 0
        env = self.env
        while games < games_per_iter:
            active = np.flatnonzero(env.stage >= 0)
            for i in active:
                i = int(i)
                if i not in self.opponents:
                    opponent = (-1, None)
                    if pool and self.rng.random() < self.pool_fraction:
                        entry = pool[int(self.rng.integers(len(pool)))]
                        key = ('pool', entry['iteration'])
                        self._actor(key, entry['weights'])
                        opponent = (int(self.rng.integers(2)), key)
                    self.opponents[i] = opponent
                if env.substep[i] == 0:
                    seat, key = self.opponents[i]
                    self.routing[i] = key if env.actor_seat[i] == seat else live
            choices = np.zeros(env.num_envs, dtype=np.int32)
            for key in set(self.routing.values()):
                rows = np.array([i for i in active if self.routing[int(i)] == key], dtype=np.int64)
                if not len(rows):
                    continue
                result = self.actors[key].act(rows)
                choices[rows] = result.choices[rows]
                for i in rows:
                    i = int(i)
                    if env.actor_seat[i] == self.opponents[i][0]:
                        continue
                    if env.substep[i] == 0:
                        obs = {k: np.array(getattr(env, k)[i], copy=True) for k in OBS_FIELDS}
                        obs['obs_num'] = obs['obs_num'].astype(np.float16)
                        logits = result.value_logits[i]
                        value = np.exp(logits - logits.max())
                        decision = Decision(obs, i, int(env.actor_seat[i]),
                                            int(env.episode_id[i]), value / value.sum())
                        self.pending[i].append(decision)
                        self.current[i] = decision
                        sampled += 1
                    prefix = {k: np.array(getattr(env, k)[i], copy=True) for k in PREFIX_FIELDS}
                    self.current[i].stages.append(Stage(prefix, int(choices[i]),
                                                       result.log_probs[i].copy()))
            env.step(choices)
            for i in np.flatnonzero(env.decision_done):
                self.routing.pop(int(i), None)
                self.current.pop(int(i), None)
            for i in np.flatnonzero(env.episode_done):
                i = int(i)
                for decision in self.pending[i]:
                    decision.outcome = 0 if env.truncated[i] else int(env.outcome[i, decision.seat])
                completed.extend(self.pending[i])
                self.pending[i] = []
                self.opponents.pop(i, None)
                self.routing.pop(i, None)
                self.current.pop(i, None)
                games += 1
                truncated += int(env.truncated[i])
            needed = {live, *self.routing.values(), *(key for _, key in self.opponents.values())}
            self.actors = {k: v for k, v in self.actors.items() if k in needed}
        for j, decision in enumerate(completed):
            for stage in decision.stages:
                stage.decision = j
        return Rollout(completed, games, truncated, sampled)
