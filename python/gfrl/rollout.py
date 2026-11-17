"""Columnar trajectory storage and batched self-play with pinned decision policies."""
from copy import copy, deepcopy
from dataclasses import dataclass, field
from math import ceil
from time import perf_counter
import weakref

import numpy as np
import torch

from . import VecEnv
from .actor import Actor, PREFIX_FIELDS, synchronize
from .model import OBS_FIELDS
from .pool import assign_opponents
from .transfer import tensor_batch


COLUMN_BUFFER_GROWTH_FACTOR = 1.25


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
    def __init__(self, columns, stages, *, cpu_storage=False):
        self.columns, self.stages = columns, stages
        self.cpu_storage = cpu_storage

    def __len__(self):
        return len(self.columns['env'])

    def __getattr__(self, name):
        if name == 'decisions':
            return self
        if name in self.columns:
            return self.columns[name]
        raise AttributeError(name)

    def __getitem__(self, rows):
        if np.isscalar(rows) or (torch.is_tensor(rows) and rows.ndim == 0):
            i = int(rows)
            obs = {k: self.columns[k][i].cpu().numpy().copy() for k in OBS_FIELDS}
            d = Decision(obs, int(self.env[i]), int(self.seat[i]), int(self.game[i]),
                         self.old_value[i].cpu().numpy().copy(), outcome=int(self.outcome[i]))
            for j in range(int(self.stage_start[i]), int(self.stage_start[i]+self.stage_count[i])):
                d.stages.append(Stage({k: self.stages[k][j].cpu().numpy().copy() for k in PREFIX_FIELDS},
                                      int(self.stages['choice'][j]), self.stages['old_log_probs'][j].cpu().numpy().copy(), i))
            return d
        return self.take(torch.arange(len(self), device=self.env.device)[rows])

    def stage_rows(self, rows):
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.env.device)
        counts = self.stage_count[rows]
        owners = torch.repeat_interleave(torch.arange(len(rows), device=rows.device), counts)
        offsets = torch.arange(len(owners), device=rows.device) - torch.repeat_interleave(
            counts.cumsum(0)-counts, counts)
        return torch.repeat_interleave(self.stage_start[rows], counts)+offsets, owners

    def take(self, rows):
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.env.device)
        sr, owners = self.stage_rows(rows)
        columns = {k: v.index_select(0, rows) for k, v in self.columns.items()}
        columns['stage_start'] = columns['stage_count'].cumsum(0)-columns['stage_count']
        stages = {k: v.index_select(0, sr) for k, v in self.stages.items()}
        stages['decision'] = owners
        return RolloutBatch(columns, stages, cpu_storage=self.cpu_storage)


class ColumnBuffer:
    def __init__(self, device='cpu', capacity=256, *, pin_memory=False):
        self.device = torch.device(device)
        self.initial_capacity = capacity
        self.pin_memory = pin_memory
        self.data = {}
        self.size = self.capacity = 0

    def reserve(self, columns, capacity):
        if capacity > self.capacity or not self.data:
            self.capacity = max(capacity, self.initial_capacity,
                                ceil(self.capacity * COLUMN_BUFFER_GROWTH_FACTOR))
            for key, value in columns.items():
                value = torch.as_tensor(value)
                array = torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype,
                                    device=self.device, pin_memory=self.pin_memory)
                if self.size:
                    array[:self.size].copy_(self.data[key][:self.size])
                # Release this old column before allocating the next one.
                self.data[key] = array

    def append(self, columns):
        n = len(next(iter(columns.values())))
        end = self.size+n
        self.reserve(columns, end)
        rows = torch.arange(self.size, end, device=self.device)
        if self.pin_memory:
            # Queue all columns into contiguous pinned slices, then wait once
            # before CPU bookkeeping reads them or a source can be reused.
            sources = [torch.as_tensor(value) for value in columns.values()]
            for key, value in zip(columns, sources):
                self.data[key][self.size:end].copy_(value, non_blocking=True)
            for device in {v.device for v in sources if v.device.type == 'cuda'}:
                synchronize(device)
        else:
            for key, value in columns.items():
                self.data[key].index_copy_(0, rows, torch.as_tensor(value, device=self.device))
        self.size = end
        return rows

    def compact(self, columns):
        n = len(next(iter(columns.values())))
        for key, value in columns.items():
            self.data[key][:n].copy_(value)
        self.size = n


class Collector:
    """Live weights are shared; only heads of unfinished decisions are pinned."""
    def __init__(self, model, *, num_envs=64, threads=4, seed=0, device='cpu',
                 dtype='fp32', pool_fraction=0.2, env=None,
                 decision_capacity=4096, stage_capacity=16384, rollout_storage='device',
                 privileged_labels=False, pool_epsilon=0.):
        if not 0 <= pool_epsilon <= 1:
            raise ValueError('pool_epsilon must be in [0,1]')
        self.privileged_labels, self.pool_epsilon = privileged_labels, pool_epsilon
        if rollout_storage not in ('device', 'cpu'):
            raise ValueError('rollout_storage must be device or cpu')
        self.cpu_storage = rollout_storage == 'cpu'
        self.storage_device = torch.device('cpu' if self.cpu_storage else device)
        pin = self.cpu_storage and torch.device(device).type == 'cuda'
        if decision_capacity <= 0 or stage_capacity <= 0:
            raise ValueError('buffer capacities must be positive')
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
        self.current = torch.full((n,), -1, dtype=torch.long, device=self.storage_device)
        self.retired_key = -2
        self.buffer = ColumnBuffer(self.storage_device, decision_capacity, pin_memory=pin)
        self.stage_buffer = ColumnBuffer(self.storage_device, stage_capacity, pin_memory=pin)
        self._batch_buffers = []

    @property
    def pending(self):
        # Compatibility for inspection; never used by collection or learning.
        batch = self._batch()
        return [list(batch.take(torch.where(batch.env == i)[0])) for i in range(self.env.num_envs)]

    def _batch(self):
        # Debug snapshot only; the hot path leases reusable output buffers.
        c = {k: v[:self.buffer.size].clone() for k,v in self.buffer.data.items()}
        s = {k: v[:self.stage_buffer.size] for k,v in self.stage_buffer.data.items()}
        order = torch.argsort(s['decision'], stable=True)
        s = {k: v.index_select(0, order) for k,v in s.items()}
        c['stage_count'] = torch.zeros(self.buffer.size, dtype=torch.long, device=order.device)
        c['stage_count'].index_add_(0, s['decision'], torch.ones_like(s['decision']))
        c['stage_start'] = c['stage_count'].cumsum(0)-c['stage_count']
        return RolloutBatch(c, s, cpu_storage=self.cpu_storage)

    def _finish_batch(self):
        # Keep leased output storage alive until the caller releases its batch.
        for slot in self._batch_buffers:
            if slot[2]() is None:
                break
        else:
            slot = [ColumnBuffer(self.buffer.device, self.buffer.initial_capacity,
                                 pin_memory=self.buffer.pin_memory),
                    ColumnBuffer(self.stage_buffer.device, self.stage_buffer.initial_capacity,
                                 pin_memory=self.stage_buffer.pin_memory),
                    lambda: None]
            self._batch_buffers.append(slot)
        decisions, stages, _ = slot
        source = self.buffer.data
        n = self.buffer.size
        completed = torch.where(source['outcome'][:n] != 2)[0]
        pending = torch.where(source['outcome'][:n] == 2)[0]
        order = torch.cat((completed, pending))
        remap = torch.empty(n, dtype=torch.long, device=order.device)
        remap[order] = torch.arange(n, device=order.device)
        columns = {k: v[:n] for k, v in source.items()}
        columns.update(stage_count=order[:0], stage_start=order[:0])
        decisions.reserve(columns, n)
        for key in source:
            torch.index_select(source[key], 0, order, out=decisions.data[key][:n])
        m = self.stage_buffer.size
        owners = remap[self.stage_buffer.data['decision'][:m]]
        stage_order = torch.argsort(owners, stable=True)
        stages.reserve(self.stage_buffer.data, m)
        for key, value in self.stage_buffer.data.items():
            torch.index_select(value, 0, stage_order, out=stages.data[key][:m])
        stages.data['decision'][:m].copy_(owners[stage_order])
        counts = decisions.data['stage_count'][:n]
        counts.zero_().index_add_(0, owners, torch.ones_like(owners))
        starts = decisions.data['stage_start'][:n]
        torch.cumsum(counts, 0, out=starts)
        starts.sub_(counts)
        nc = len(completed)
        ns = int(counts[:nc].sum())
        batch = RolloutBatch({k: v[:nc] for k, v in decisions.data.items()},
                             {k: v[:ns] for k, v in stages.data.items()})
        batch.cpu_storage = getattr(self, 'cpu_storage', False)
        slot[2] = weakref.ref(batch)
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
            if self.privileged_labels:
                from .critic import PRIVILEGED_FIELDS
                labels = tensor_batch({k: getattr(env, k) for k in PRIVILEGED_FIELDS}, live.device)
            # Native uint64 episode counters use numpy's platform ulonglong
            # spelling on macOS; reinterpret the counter bits without a copy.
            snapshot['episode_id'] = snapshot['episode_id'].view(np.int64)
            if self.cpu_storage:
                snapshot['episode_id'] = snapshot['episode_id'].copy()
            active = np.flatnonzero(snapshot['stage'] >= 0)
            fresh = active[snapshot['substep'][active] == 0]
            new_games = active[~self.assigned[active]]
            if len(new_games):
                self.assigned[new_games] = True
                selected = assign_opponents(self.rng, new_games, pool, self.pool_fraction,
                                            self.opponent_seat, self.opponent_policy)
                for key, entry in selected.items():
                    if key not in self.actors:
                        model = deepcopy(self.model)
                        model.load_state_dict(entry['weights'])
                        self.actors[key] = Actor(model, env, self.device, dtype=self.dtype,
                                                 epsilon=self.pool_epsilon)
            self.routing[fresh] = np.where(snapshot['actor_seat'][fresh] == self.opponent_seat[fresh],
                                            self.opponent_policy[fresh], -1)
            # Transfer each persistent native buffer once, before any row selection.
            tensors = tensor_batch(snapshot, live.device)
            # CPU from_numpy aliases the native views, including across step().
            tensors['episode_id'] = tensors['episode_id'].clone()
            choices = np.full(env.num_envs, -1, dtype=np.int32)
            for key in np.unique(self.routing[active]):
                rows = active[self.routing[active] == key]
                synchronize(live.device)
                since = perf_counter()
                result = self.actors[key].act(rows, buffers=snapshot, tensors=tensors, device_result=True)
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
                new_t = torch.as_tensor(new, device=live.device)
                training_t = torch.as_tensor(training, device=live.device)
                logits = result.value_logits.index_select(0, new_t)
                # Preserve the previous numpy exp/normalize arithmetic.
                value = (logits-logits.max(dim=1, keepdim=True).values).exp()
                columns = {k: tensors[k].index_select(0, new_t).to(torch.float16)
                           if k == 'obs_num' else tensors[k].index_select(0, new_t)
                           for k in OBS_FIELDS}
                columns.update(env=new_t, seat=tensors['actor_seat'][new_t],
                               game=tensors['episode_id'][new_t],
                               old_value=value/value.sum(dim=1, keepdim=True),
                               outcome=torch.full((len(new),), 2, dtype=torch.int8, device=live.device))
                if self.privileged_labels:
                    columns.update({k: labels[k].index_select(0, new_t) for k in PRIVILEGED_FIELDS})
                new_storage = torch.as_tensor(new, device=self.storage_device)
                training_storage = torch.as_tensor(training, device=self.storage_device)
                self.current[new_storage] = self.buffer.append(columns)
                sampled += len(new)
                stages = {k: tensors[k].index_select(0, training_t) for k in PREFIX_FIELDS}
                stages.update(decision=self.current[training_storage],
                              choice=torch.as_tensor(choices, device=live.device)[training_t],
                              old_log_probs=result.log_probs.index_select(0, training_t).masked_fill(
                                  ~stages['mask'], -torch.inf).to(torch.float16))
                self.stage_buffer.append(stages)
                timing['record'] += perf_counter()-since
            since = perf_counter()
            env.step(choices)
            timing['env_step'] += perf_counter()-since
            steps += 1
            since = perf_counter()
            done, ended, trunc, outcome = env.decision_done, env.episode_done, env.truncated, env.outcome
            self.current[torch.as_tensor(done, device=self.storage_device)] = -1
            self.routing[done] = -1
            c = self.buffer.data
            # Gather results using env/seat columns on the storage device.
            ended_t = torch.as_tensor(ended, device=self.storage_device)
            trunc_t = torch.as_tensor(trunc, device=self.storage_device)
            outcome_t = torch.as_tensor(outcome, device=self.storage_device)
            envs = c['env'][:self.buffer.size]
            seats = c['seat'][:self.buffer.size].long()
            games_t = (torch.from_numpy(snapshot['episode_id']) if self.cpu_storage
                       else tensors['episode_id'])
            unfinished = c['outcome'][:self.buffer.size] == 2
            finished = ended_t[envs] & unfinished & (c['game'][:self.buffer.size] == games_t[envs])
            c['outcome'][:self.buffer.size][finished] = torch.where(
                trunc_t[envs[finished]], 0, outcome_t[envs[finished], seats[finished]])
            ended_rows = np.flatnonzero(ended)
            self.assigned[ended_rows] = False
            self.opponent_seat[ended_rows] = self.opponent_policy[ended_rows] = -1
            self.routing[ended_rows] = -1
            self.current[torch.as_tensor(ended_rows, device=self.storage_device)] = -1
            games += len(ended_rows)
            truncated += int(trunc[ended_rows].sum())
            timing['record'] += perf_counter()-since
        since = perf_counter()
        completed, pending_count = self._finish_batch()
        # A decision may cross the synchronous update boundary. Its encoder
        # output is already cached; preserve only its small heads, not encoder.
        inflight = (self.routing == -1) & (self.current.cpu().numpy() >= 0)
        inflight |= (self.routing == -1) & (env.substep != 0) & (env.stage >= 0)
        if np.any(inflight):
            frozen = copy(live)
            frozen.model = copy(self.model)
            frozen.model._compiled_encoder = None
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
