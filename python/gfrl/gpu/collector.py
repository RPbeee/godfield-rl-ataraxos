"""GPU self-play trajectories compatible with the existing PPO learner."""
from copy import deepcopy
from time import perf_counter

import numpy as np

import torch

from ..actor import PREFIX_FIELDS
from ..model import OBS_FIELDS
from ..pool import assign_opponents
from ..rollout import RolloutBatch
from . import GpuEnv
from .storage import NumericStorage
from .buffers import GpuColumnBuffer,PagedColumn
from .actor import GpuActor,RoutedGpuActor,ParallelGpuActor,MAX_POLICY_STAGES


def _compact_pending(array,indices,chunk=16384):
    # where() returns increasing indices, with indices[j]>=j. Forward chunks
    # never overwrite a later chunk's sources; gather each chunk completely
    # before copying it back to handle overlap within the chunk itself.
    for start in range(0,len(indices),chunk):
        stop=min(start+chunk,len(indices))
        retained=array.index_select(0,indices[start:stop])
        array[start:stop].copy_(retained)


class GpuCollector:
    """Game-pinned opponents; only the current policy supplies training rows.

    The host reads one episode counter every four decision rounds to bound an
    iteration. Observations, choices, histories, labels, routing of terminal
    results and pending trajectories stay on device. With pool routing, only
    per-row episode-ended flags return to the host for the shared CPU lottery.
    Final ragged compaction happens once at the learner boundary.
    """
    def __init__(self, model, *, num_envs=256, threads=4, seed=0, device='cuda',
                 dtype='bf16', pool_fraction=.2, env=None, decision_capacity=4096,
                 stage_capacity=16384, rollout_storage='device', capture=True,
                 pooled=False,arena_bytes=None,streams=1,privileged_labels=False,pool_epsilon=0.):
        if not 0 <= pool_epsilon <= 1:
            raise ValueError('pool_epsilon must be in [0,1]')
        self.privileged_labels, self.pool_epsilon = privileged_labels, pool_epsilon
        if not 0 <= pool_fraction <= 1:
            raise ValueError('pool_fraction must be in [0,1]')
        if decision_capacity <= 0 or stage_capacity <= 0:
            raise ValueError('buffer capacities must be positive')
        if rollout_storage != 'device':
            raise ValueError('GPU collector requires rollout_storage=device')
        self.env = env or GpuEnv(num_envs, base_seed=seed, device=device, pooled=pooled,arena_bytes=arena_bytes)
        self.model, self.device = model, self.env.device
        self.dtype, self.streams, self.capture = dtype, streams, capture
        self.pool_fraction = pool_fraction
        self.rng = np.random.default_rng(seed)
        n = self.env.num_envs
        self.assigned = np.zeros(n, dtype=bool)
        self.opponent_seat = np.full(n, -1, dtype=np.int8)
        self.opponent_policy = np.full(n, -1, dtype=np.int64)
        self._opponent_seat = torch.full((n,), -1, dtype=torch.int8, device=self.device)
        self._opponent_policy = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self._models = {}
        self._actor_keys = None
        self._configure_actor(())
        self.numeric=NumericStorage(self.device,self.env.history_len)
        self.columns = GpuColumnBuffer(self.device,decision_capacity,OBS_FIELDS,self.env.num_envs)
        self.stages = GpuColumnBuffer(self.device,decision_capacity,(*PREFIX_FIELDS,'choice','old_log_probs'),self.env.num_envs)
        self._env_rows = torch.arange(self.env.num_envs, device=self.device)

    def _configure_actor(self, pool):
        entries = {entry['iteration']: entry for entry in pool} if self.pool_fraction else {}
        needed = set(entries) | (set(self.opponent_policy.tolist()) - {-1})
        if needed == self._actor_keys:
            return
        # Keep an evicted snapshot alive until its last assigned game ends.
        for key in needed - self._models.keys():
            policy = deepcopy(self.model)
            policy.load_state_dict(entries[key]['weights'])
            self._models[key] = policy.eval()
        models = {key: self._models[key] for key in sorted(needed)}
        if hasattr(self, 'actor'):
            del self.actor
        self._models = models
        routing = dict(opponents=models, opponent_seat=self._opponent_seat,
                       opponent_policy=self._opponent_policy, pool_epsilon=self.pool_epsilon)
        if self.streams != 1:
            self.actor = ParallelGpuActor(self.model, self.env, streams=self.streams,
                                         dtype=self.dtype, privileged_labels=self.privileged_labels, **routing)
        elif models:
            self.actor = RoutedGpuActor(self.model, self.env, dtype=self.dtype,
                                        privileged_labels=self.privileged_labels, **routing)
        else:
            self.actor = GpuActor(self.model, self.env, dtype=self.dtype,
                                  privileged_labels=self.privileged_labels)
        if self.capture:
            self.actor.capture(record=True)
        self._actor_keys = needed

    def _assign(self, pool):
        new_games = np.flatnonzero(~self.assigned)
        self.assigned[new_games] = True
        assign_opponents(self.rng, new_games, pool, self.pool_fraction,
                         self.opponent_seat, self.opponent_policy)
        self._opponent_seat.copy_(torch.from_numpy(self.opponent_seat))
        self._opponent_policy.copy_(torch.from_numpy(self.opponent_policy))

    def collect(self, games_per_iter, pool=()):
        if games_per_iter <= 0:
            raise ValueError('games_per_iter must be positive')
        self._configure_actor(pool)
        self.model.eval()
        self.actor.refresh()
        torch.cuda.synchronize(self.device)
        begin = perf_counter()
        games = torch.zeros((), dtype=torch.int64, device=self.device)
        truncated = torch.zeros_like(games)
        sampled = torch.zeros_like(games)
        carried = self.columns.size
        rounds = 0
        encoded_tokens = torch.zeros((),dtype=torch.int64,device=self.device)
        n = self.env.num_envs
        while True:
            if self.pool_fraction:
                self._assign(pool)
            finished, data = self.actor.round(record=True)
            encoded_tokens += data['obs_valid'].sum() if self.actor.packed_encoder is not None else n*self.env.tokens
            training = data['actor_seat'] != self._opponent_seat
            sampled += training.sum()
            vl = data['stages'][0]['old_value']
            value = (vl-vl.max(dim=1, keepdim=True).values).exp()
            columns = {k:self.numeric.pack(data[k],data['obs_type']) if k=='obs_num' else data[k] for k in OBS_FIELDS}
            columns.update(env=self._env_rows, game=data['episode_id'].view(torch.int64),
                           seat=data['actor_seat'], old_value=value/value.sum(dim=1, keepdim=True),
                           outcome=torch.where(training, 2, 3).to(torch.int8),
                           label_opp_hand=data['label_opp_hand'],
                           label_opp_hand_count=data['label_opp_hand_count'])
            if self.privileged_labels:
                from ..critic import PRIVILEGED_FIELDS
                columns.update({k: data[k] for k in PRIVILEGED_FIELDS})
            self.columns.append(columns)
            dense = {k:torch.stack([s[k] for s in data['stages']], dim=1)
                     for k in (*PREFIX_FIELDS, 'valid', 'choice', 'old_log_probs')}
            dense['old_log_probs'] = dense['old_log_probs'].masked_fill(~dense['mask'], -torch.inf).half()
            self.stages.append(dense)
            c = self.columns.data
            m = self.columns.size
            rows, seat = c['env'][:m], c['seat'][:m].long()
            ended = (data['episode_done'][rows] & (c['game'][:m] == columns['game'][rows])
                     & (c['outcome'][:m] == 2))
            terminal = torch.where(data['truncated'][rows], 0, data['outcome'][rows,seat])
            c['outcome'][:m].copy_(torch.where(ended, terminal, c['outcome'][:m]))
            games += data['episode_done'].sum()
            truncated += (data['episode_done'] & data['truncated']).sum()
            if self.pool_fraction:
                ended = data['episode_done'].cpu().numpy()
                self.assigned[ended] = False
                self.opponent_seat[ended] = self.opponent_policy[ended] = -1
            rounds += 1
            if rounds % 4 == 0 and int(games) >= games_per_iter:
                break
        if not bool(finished.all()):
            raise RuntimeError('not all decision rounds completed')
        batch, pending = self._finish()
        torch.cuda.synchronize(self.device)
        elapsed = perf_counter()-begin
        token_count = int(encoded_tokens)
        batch.games, batch.truncated = int(games), int(truncated)
        batch.sampled_decisions = int(sampled)
        batch.timings = dict(other=elapsed, env_step=0., inference=0., record=0.,
                             actor_calls=rounds, mean_actor_rows=n, steps=rounds*MAX_POLICY_STAGES,
                             carried_decisions=carried, pending_decisions=pending,
                             encoded_envs=rounds*n, encoder_tokens=token_count,
                             total_decisions=rounds*n, opponent_decisions=rounds*n-int(sampled),
                             mean_packed_tokens=token_count/(rounds*n), mean_unpacked_tokens=self.env.tokens)
        return batch

    def _finish(self):
        n = self.columns.size
        # 3 is a private discard marker for opponent seats, never learner data.
        outcomes = self.columns.data['outcome'][:n]
        completed = torch.where((outcomes != 2) & (outcomes != 3))[0]
        pending = torch.where(self.columns.data['outcome'][:n] == 2)[0]
        npending=len(pending)
        output={}
        # Keep one pending-column temporary live at a time. Reordering every
        # column together can double the entire rollout's device footprint.
        for key,array in self.columns.data.items():
            if isinstance(array,PagedColumn):
                output[key]=array.gather(completed);array.compact(pending)
            else:
                output[key]=array[:n].index_select(0,completed)
                _compact_pending(array,pending)
        self.columns.size=npending
        dense_valid=self.stages.data['valid'][:n].index_select(0,completed)
        counts=dense_valid.sum(1)
        valid=torch.where(dense_valid.flatten())[0]
        stages={}
        for key,array in self.stages.data.items():
            if key!='valid':
                selected=array.gather(completed) if isinstance(array,PagedColumn) else array[:n].index_select(0,completed)
                stages[key]=selected.flatten(0,1).index_select(0,valid)
                del selected
            if isinstance(array,PagedColumn):array.compact(pending)
            else:_compact_pending(array,pending)
        self.stages.size=npending
        stages['decision']=valid//MAX_POLICY_STAGES
        output['obs_num']=self.numeric.expand(output['obs_num'],output['obs_type'])
        output.update(stage_count=counts,stage_start=counts.cumsum(0)-counts)
        return RolloutBatch(output,stages),npending

    def close(self):
        self.env.close()
