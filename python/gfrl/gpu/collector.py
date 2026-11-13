"""GPU self-play trajectories compatible with the existing PPO learner."""
from time import perf_counter

import torch

from ..actor import PREFIX_FIELDS
from ..model import OBS_FIELDS
from ..rollout import RolloutBatch
from . import GpuEnv
from .storage import NumericStorage
from .buffers import GpuColumnBuffer,PagedColumn
from .actor import GpuActor,ParallelGpuActor,MAX_POLICY_STAGES


def _compact_pending(array,indices,chunk=16384):
    # where() returns increasing indices, with indices[j]>=j. Forward chunks
    # never overwrite a later chunk's sources; gather each chunk completely
    # before copying it back to handle overlap within the chunk itself.
    for start in range(0,len(indices),chunk):
        stop=min(start+chunk,len(indices))
        retained=array.index_select(0,indices[start:stop])
        array[start:stop].copy_(retained)


class GpuCollector:
    """Both seats use the current model; complete decisions stay on the GPU.

    The host reads one episode counter every four decision rounds to bound an
    iteration. Observations, choices, histories, labels, routing of terminal
    results and pending trajectories never enter host memory. Final ragged
    compaction happens once at the learner boundary.
    """
    def __init__(self, model, *, num_envs=256, threads=4, seed=0, device='cuda',
                 dtype='bf16', pool_fraction=0, env=None, decision_capacity=4096,
                 stage_capacity=16384, rollout_storage='device', capture=True,
                 pooled=False,arena_bytes=None,streams=1):
        if pool_fraction != 0:
            raise ValueError('GPU collector currently requires pool_fraction=0')
        if rollout_storage != 'device':
            raise ValueError('GPU collector requires rollout_storage=device')
        self.env = env or GpuEnv(num_envs, base_seed=seed, device=device, pooled=pooled,arena_bytes=arena_bytes)
        self.model, self.device = model, self.env.device
        self.actor = (GpuActor(model,self.env,dtype=dtype) if streams==1 else
                      ParallelGpuActor(model,self.env,dtype=dtype,streams=streams))
        self.numeric=NumericStorage(self.device,self.env.history_len)
        self.columns = GpuColumnBuffer(self.device,decision_capacity,OBS_FIELDS,self.env.num_envs)
        self.stages = GpuColumnBuffer(self.device,decision_capacity,(*PREFIX_FIELDS,'choice','old_log_probs'),self.env.num_envs)
        self._env_rows = torch.arange(self.env.num_envs, device=self.device)
        if capture:
            self.actor.capture(record=True)

    def collect(self, games_per_iter, pool=()):
        if games_per_iter <= 0:
            raise ValueError('games_per_iter must be positive')
        # A saved pool can be present in a checkpoint while its sampling
        # fraction is zero; only the live current policy is used here.
        self.model.eval()
        self.actor.refresh()
        torch.cuda.synchronize(self.device)
        begin = perf_counter()
        games = torch.zeros((), dtype=torch.int64, device=self.device)
        truncated = torch.zeros_like(games)
        carried = self.columns.size
        rounds = 0
        encoded_tokens = torch.zeros((),dtype=torch.int64,device=self.device)
        n = self.env.num_envs
        while True:
            finished, data = self.actor.round(record=True)
            encoded_tokens += data['obs_valid'].sum() if self.actor.packed_encoder is not None else n*self.env.tokens
            vl = data['stages'][0]['old_value']
            value = (vl-vl.max(dim=1, keepdim=True).values).exp()
            columns = {k:self.numeric.pack(data[k],data['obs_type']) if k=='obs_num' else data[k] for k in OBS_FIELDS}
            columns.update(env=self._env_rows, game=data['episode_id'].view(torch.int64),
                           seat=data['actor_seat'], old_value=value/value.sum(dim=1, keepdim=True),
                           outcome=torch.full((n,), 2, dtype=torch.int8, device=self.device),
                           label_opp_hand=data['label_opp_hand'],
                           label_opp_hand_count=data['label_opp_hand_count'])
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
        batch.sampled_decisions = rounds*n
        batch.timings = dict(other=elapsed, env_step=0., inference=0., record=0.,
                             actor_calls=rounds, mean_actor_rows=n, steps=rounds*MAX_POLICY_STAGES,
                             carried_decisions=carried, pending_decisions=pending,
                             encoded_envs=rounds*n, encoder_tokens=token_count,
                             mean_packed_tokens=token_count/(rounds*n), mean_unpacked_tokens=self.env.tokens)
        return batch

    def _finish(self):
        n = self.columns.size
        completed = torch.where(self.columns.data['outcome'][:n] != 2)[0]
        pending = torch.where(self.columns.data['outcome'][:n] == 2)[0]
        nc = len(completed)
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
