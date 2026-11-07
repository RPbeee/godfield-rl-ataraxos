"""CPU trajectory records and batched self-play with pinned decision policies."""
from copy import copy, deepcopy
from dataclasses import dataclass, field
from time import perf_counter
import weakref

import numpy as np

from gfrl import VecEnv
from .columnar_actor import Actor, PREFIX_FIELDS, synchronize
from gfrl.model import OBS_FIELDS


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


class RolloutBatch:
    """Contiguous decision and stage columns. Object access is a debug adapter only."""
    def __init__(self, columns, stages):
        self.columns, self.stages = columns, stages

    def __len__(self):
        return len(self.columns['env'])

    def __getattr__(self, name):
        if name == 'decisions':
            return self
        if name in self.columns:
            return self.columns[name]
        raise AttributeError(name)

    def __getitem__(self, rows):
        if np.isscalar(rows):
            i = int(rows)
            obs = {k: self.columns[k][i] for k in OBS_FIELDS}
            d = Decision(obs, int(self.env[i]), int(self.seat[i]), int(self.game[i]),
                         self.old_value[i], outcome=int(self.outcome[i]))
            for j in range(int(self.stage_start[i]), int(self.stage_start[i]+self.stage_count[i])):
                d.stages.append(Stage({k: self.stages[k][j] for k in PREFIX_FIELDS},
                                      int(self.stages['choice'][j]), self.stages['old_log_probs'][j], i))
            return d
        return self.take(np.arange(len(self))[rows])

    def stage_rows(self, rows):
        counts = self.stage_count[rows]
        owners = np.repeat(np.arange(len(rows)), counts)
        offsets = np.arange(len(owners)) - np.repeat(np.cumsum(counts)-counts, counts)
        return np.repeat(self.stage_start[rows], counts)+offsets, owners

    def take(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        sr, owners = self.stage_rows(rows)
        columns = {k: np.ascontiguousarray(v[rows]) for k, v in self.columns.items()}
        columns['stage_start'] = np.cumsum(columns['stage_count'])-columns['stage_count']
        stages = {k: np.ascontiguousarray(v[sr]) for k, v in self.stages.items()}
        stages['decision'] = owners
        return RolloutBatch(columns, stages)


class ColumnBuffer:
    def __init__(self):
        self.data = {}
        self.size = self.capacity = 0

    def reserve(self, columns, capacity):
        if capacity > self.capacity or not self.data:
            self.capacity = max(capacity, max(256, self.capacity*2))
            for key, value in columns.items():
                array = np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
                if self.size:
                    array[:self.size] = self.data[key][:self.size]
                self.data[key] = array

    def append(self, columns):
        n = len(next(iter(columns.values())))
        end = self.size+n
        self.reserve(columns, end)
        rows = np.arange(self.size, end)
        for key, value in columns.items():
            self.data[key][self.size:end] = value
        self.size = end
        return rows

    def compact(self, columns):
        n = len(next(iter(columns.values())))
        for key, value in columns.items():
            self.data[key][:n] = value
        self.size = n


class Collector:
    """Live weights are shared; only heads of unfinished decisions are pinned."""
    def __init__(self, model, *, num_envs=64, threads=4, seed=0, device='cpu',
                 dtype='fp32', pool_fraction=0.2, env=None):
        if not 0 <= pool_fraction <= 1:
            raise ValueError('pool_fraction must be in [0,1]')
        self.env = env if env is not None else VecEnv(num_envs, threads, seed,
                                                   controllers=('external', 'external'))
        self.model, self.device, self.dtype = model, device, dtype
        self.rng = np.random.default_rng(seed)
        self.pool_fraction = pool_fraction
        self.actors = {-1: Actor(model, self.env, device, dtype=dtype)}
        n = self.env.num_envs
        self.opponent_seat = np.full(n, -1, dtype=np.int8)
        self.opponent_policy = np.full(n, -1, dtype=np.int64)
        self.assigned = np.zeros(n, dtype=bool)
        self.routing = np.full(n, -1, dtype=np.int64)
        self.current = np.full(n, -1, dtype=np.int64)
        self.retired_key = -2
        self.buffer, self.stage_buffer = ColumnBuffer(), ColumnBuffer()
        self._batch_buffers = []

    @property
    def pending(self):
        # Compatibility for inspection; never used by collection or learning.
        batch = self._batch()
        return [list(batch.take(np.flatnonzero(batch.env == i))) for i in range(self.env.num_envs)]

    def _batch(self, *, snapshot=True):
        c = {k: v[:self.buffer.size].copy() if snapshot else v[:self.buffer.size]
             for k,v in self.buffer.data.items()}
        s = {k: v[:self.stage_buffer.size] for k,v in self.stage_buffer.data.items()}
        order = np.argsort(s['decision'], kind='stable')
        s = {k: np.ascontiguousarray(v[order]) for k,v in s.items()}
        c['stage_count'] = np.bincount(s['decision'], minlength=self.buffer.size)
        c['stage_start'] = np.cumsum(c['stage_count'])-c['stage_count']
        return RolloutBatch(c, s)

    def _finish_batch(self):
        # Lease output storage while callers retain the batch. In the usual
        # collect/update loop two slots suffice, even during RHS evaluation of
        # `rollout = collector.collect(...)`. Keep the batch alive while using
        # its views; take() creates an independent snapshot when needed.
        for slot in self._batch_buffers:
            if slot[2]() is None:
                break
        else:
            slot = [ColumnBuffer(), ColumnBuffer(), lambda: None]
            self._batch_buffers.append(slot)
        decisions, stages, _ = slot
        source = self.buffer.data
        n = self.buffer.size
        completed = np.flatnonzero(source['outcome'][:n] != 2)
        pending = np.flatnonzero(source['outcome'][:n] == 2)
        order = np.concatenate((completed, pending))
        remap = np.empty(n, dtype=np.int64)
        remap[order] = np.arange(n)
        columns = {k: v[:n] for k, v in source.items()}
        columns.update(stage_count=np.empty(0, dtype=np.int64),
                       stage_start=np.empty(0, dtype=np.int64))
        decisions.reserve(columns, n)
        for key in source:
            np.take(source[key], order, axis=0, out=decisions.data[key][:n], mode='clip')
        m = self.stage_buffer.size
        owners = remap[self.stage_buffer.data['decision'][:m]]
        stage_order = np.argsort(owners, kind='stable')
        stages.reserve(self.stage_buffer.data, m)
        for key, value in self.stage_buffer.data.items():
            np.take(value, stage_order, axis=0, out=stages.data[key][:m], mode='clip')
        stages.data['decision'][:m] = owners[stage_order]
        counts = decisions.data['stage_count'][:n]
        counts[:] = np.bincount(owners, minlength=n)
        starts = decisions.data['stage_start'][:n]
        np.cumsum(counts, out=starts)
        starts -= counts
        nc = len(completed)
        ns = int(counts[:nc].sum())
        batch = RolloutBatch({k: v[:nc] for k, v in decisions.data.items()},
                             {k: v[:ns] for k, v in stages.data.items()})
        slot[2] = weakref.ref(batch)
        # Pending rows and their stages form the tail of the workspace. Copy
        # straight back into the existing collection storage, without take().
        self.buffer.compact({k: decisions.data[k][nc:n] for k in source})
        self.stage_buffer.compact({k: v[ns:m] for k, v in stages.data.items()})
        self.stage_buffer.data['decision'][:m-ns] -= nc
        active = self.current >= 0
        self.current[active] = remap[self.current[active]]-nc
        return batch, len(pending)

    def collect(self, games_per_iter, pool=()):
        if games_per_iter <= 0:
            raise ValueError('games_per_iter must be positive')
        env = self.env
        live = self.actors[-1]
        self.model.eval()
        timing = dict(env_step=0., inference=0., record=0.)
        games = truncated = sampled = calls = inference_rows = steps = 0
        encoded_envs = encoder_tokens = unpacked_tokens = 0
        carried = self.buffer.size
        synchronize(live.device)
        begin = perf_counter()
        while games < games_per_iter:
            snapshot = {k: getattr(env, k) for k in (*OBS_FIELDS, *PREFIX_FIELDS,
                        'substep', 'actor_seat', 'episode_id')}
            active = np.flatnonzero(snapshot['stage'] >= 0)
            fresh = active[snapshot['substep'][active] == 0]
            new_games = active[~self.assigned[active]]
            if len(new_games):
                self.assigned[new_games] = True
                if pool:
                    selected = new_games[self.rng.random(len(new_games)) < self.pool_fraction]
                    entries = self.rng.integers(len(pool), size=len(selected))
                    self.opponent_seat[selected] = self.rng.integers(2, size=len(selected))
                    for j, entry in enumerate(pool):
                        rows = selected[entries == j]
                        if not len(rows):
                            continue
                        key = entry['iteration']
                        if key not in self.actors:
                            model = deepcopy(self.model)
                            model.load_state_dict(entry['weights'])
                            self.actors[key] = Actor(model, env, self.device, dtype=self.dtype)
                        self.opponent_policy[rows] = key
            self.routing[fresh] = np.where(snapshot['actor_seat'][fresh] == self.opponent_seat[fresh],
                                            self.opponent_policy[fresh], -1)
            choices = np.full(env.num_envs, -1, dtype=np.int32)
            for key in np.unique(self.routing[active]):
                rows = active[self.routing[active] == key]
                synchronize(live.device)
                since = perf_counter()
                result = self.actors[key].act(rows, buffers=snapshot)
                synchronize(live.device)
                timing['inference'] += perf_counter()-since
                profile = self.actors[key].last_profile
                encoded_envs += profile['encoded_envs']
                encoder_tokens += profile['encoder_tokens']
                unpacked_tokens += profile['unpacked_tokens']
                calls += 1
                inference_rows += len(rows)
                since = perf_counter()
                choices[rows] = result.choices[rows]
                training = rows[snapshot['actor_seat'][rows] != self.opponent_seat[rows]]
                new = training[snapshot['substep'][training] == 0]
                logits = result.value_logits[new]
                value = np.exp(logits-logits.max(axis=1, keepdims=True))
                columns = {k: snapshot[k][new].astype(np.float16) if k == 'obs_num'
                           else snapshot[k][new] for k in OBS_FIELDS}
                columns.update(env=new, seat=snapshot['actor_seat'][new], game=snapshot['episode_id'][new],
                               old_value=value/value.sum(axis=1, keepdims=True),
                               outcome=np.full(len(new), 2, dtype=np.int8))
                self.current[new] = self.buffer.append(columns)
                sampled += len(new)
                stages = {k: snapshot[k][training] for k in PREFIX_FIELDS}
                stages.update(decision=self.current[training], choice=choices[training],
                              old_log_probs=np.where(snapshot['mask'][training],
                                  result.log_probs[training], -np.inf).astype(np.float16))
                self.stage_buffer.append(stages)
                timing['record'] += perf_counter()-since
            since = perf_counter()
            env.step(choices)
            timing['env_step'] += perf_counter()-since
            steps += 1
            since = perf_counter()
            done, ended, trunc, outcome = env.decision_done, env.episode_done, env.truncated, env.outcome
            self.current[done] = -1
            self.routing[done] = -1
            c = self.buffer.data
            for i in np.flatnonzero(ended):
                rows = np.flatnonzero((c['env'][:self.buffer.size] == i) &
                                      (c['outcome'][:self.buffer.size] == 2))
                c['outcome'][rows] = 0 if trunc[i] else outcome[i, c['seat'][rows]]
                self.assigned[i] = False
                self.opponent_seat[i] = self.opponent_policy[i] = -1
                self.routing[i] = self.current[i] = -1
                games += 1
                truncated += int(trunc[i])
            timing['record'] += perf_counter()-since
        since = perf_counter()
        completed, pending_count = self._finish_batch()
        # A decision may cross the synchronous update boundary. Its encoder
        # output is already cached; preserve only its small heads, not encoder.
        inflight = (self.routing == -1) & (self.current >= 0)
        inflight |= (self.routing == -1) & (env.substep != 0) & (env.stage >= 0)
        if np.any(inflight):
            frozen = copy(live)
            frozen.model = copy(self.model)
            frozen.model._modules = self.model._modules.copy()
            for name in ('kind_embedding', 'stage_embedding', 'target_embedding', 'hp_embedding',
                         'query', 'kind_head', 'card_query', 'stop_head', 'target_head',
                         'hp_head', 'mp_head', 'value_head'):
                frozen.model._modules[name] = deepcopy(self.model._modules[name])
            frozen._cached = live._cached.copy()
            self.actors[self.retired_key] = frozen
            self.routing[inflight] = self.retired_key
            self.retired_key -= 1
        needed = {-1, *self.opponent_policy.tolist(), *self.routing.tolist()}
        self.actors = {k: v for k,v in self.actors.items() if k in needed}
        timing['record'] += perf_counter()-since
        synchronize(live.device)
        timing['other'] = perf_counter()-begin-sum(timing.values())
        timing.update(actor_calls=calls, mean_actor_rows=inference_rows/max(calls,1),
                      steps=steps, carried_decisions=carried, pending_decisions=pending_count,
                      encoded_envs=encoded_envs, encoder_tokens=encoder_tokens,
                      mean_packed_tokens=encoder_tokens/max(encoded_envs,1),
                      mean_unpacked_tokens=unpacked_tokens/max(encoded_envs,1))
        completed.games, completed.truncated = games, truncated
        completed.sampled_decisions, completed.timings = sampled, timing
        return completed
