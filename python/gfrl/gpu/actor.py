"""Fixed-shape, device-only inference across complete decision rounds."""
from contextlib import nullcontext

import torch

from ..actor import PREFIX_FIELDS
from ..model import OBS_FIELDS

# At most kind + 18 card choices + target + two resource allocations.
# Automatic singleton choices are performed by the engine itself.
MAX_POLICY_STAGES = 22


class GpuActor:
    def __init__(self, model, env, *, sample=True, dtype='bf16', packed=None):
        if dtype not in ('fp32', 'fp16', 'bf16'):
            raise ValueError('dtype must be fp32, fp16, or bf16')
        self.model = model.to(device=env.device, dtype=torch.float32).eval()
        self.env, self.sample, self.dtype = env, sample, dtype
        self._graphs = {}
        self._execute = torch.ones((), dtype=torch.bool, device=env.device)
        if packed is None:
            packed = dtype == 'bf16'
        self.packed_encoder = None
        self.sparse_heads = None
        if packed:
            if dtype != 'bf16':
                raise ValueError('packed inference requires dtype=bf16')
            from .packed import ChunkedPackedEncoder
            self.packed_encoder = ChunkedPackedEncoder(self.model,env)
            from .heads import SparseHeads
            self.sparse_heads = SparseHeads(self.model,env)

    def _autocast(self):
        return (nullcontext() if self.dtype == 'fp32' else torch.autocast(
            'cuda', dtype={'fp16':torch.float16, 'bf16':torch.bfloat16}[self.dtype]))

    @torch.inference_mode()
    def encode(self, active=None):
        # Fixed width avoids Model.encode's packed-width .tolist() sync. Heads
        # use only CLS and the original 18 hand slots, so discard other outputs.
        if self.packed_encoder is not None:
            return self.packed_encoder(active)
        with self._autocast():
            return self.model.encode({k:getattr(self.env,k) for k in OBS_FIELDS}, pack=False)[:, :21].contiguous()

    @torch.inference_mode()
    def act(self, encoded, active=None, value_logits=None):
        if self.sparse_heads is not None:
            lp = self.sparse_heads(encoded,active)
            value = value_logits if value_logits is not None else self.sparse_heads.values(encoded)
        else:
            with self._autocast():
                lp, value = self.model.heads(encoded, **{k:getattr(self.env,k) for k in PREFIX_FIELDS})
        choices = (torch.multinomial(lp.exp(), 1).squeeze(1) if self.sample else lp.argmax(-1))
        return choices.to(torch.int32), lp, value

    @torch.inference_mode()
    def _round(self, *, record=False):
        """Execute exactly one external decision per row without host reads.

        Completed rows freeze at their next decision while longer selections
        finish. Every policy head in the decision uses the same encoded view.
        A fixed number of launches avoids reading completion counts on the CPU.
        """
        env = self.env
        encoded = self.encode()
        values = (self.sparse_heads.values(encoded)
                  if self.sparse_heads is not None and isinstance(encoded, torch.Tensor) else None)
        finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        trajectory = None
        if record:
            trajectory = {k:getattr(env,k).clone() for k in (*OBS_FIELDS, 'actor_seat', 'episode_id',
                                                           'label_opp_hand', 'label_opp_hand_count')}
            trajectory['stages'] = []
        for _ in range(MAX_POLICY_STAGES):
            choices, lp, value = self.act(encoded,active=~finished,value_logits=values)
            if record:
                stage = {k:getattr(env,k).clone() for k in PREFIX_FIELDS}
                stage.update(valid=~finished, choice=choices, old_log_probs=lp,
                             old_value=value)
                trajectory['stages'].append(stage)
            active = ~finished & self._execute
            env.step(torch.where(active, choices, -1), _active=active)
            finished = finished | env.decision_done
        if record:
            trajectory.update(finished=finished, outcome=env.outcome.clone(),
                              episode_done=env.episode_done.clone(), truncated=env.truncated.clone())
        return finished, trajectory

    def refresh(self):
        if self.packed_encoder is not None:
            self.packed_encoder.refresh()
        if self.sparse_heads is not None:
            self.sparse_heads.refresh()

    def capture(self, *, record=False):
        """Capture the complete engine/inference round, including GPU resets."""
        if self.env._closed:
            raise RuntimeError('GpuEnv is closed')
        if record in self._graphs:
            return
        # Compile/capture with every engine lane frozen at its decision start.
        # Recapturing for a changed pool must never consume or discard a move
        # from an unfinished trajectory. The execution gate is read on replay.
        self._execute.fill_(False)
        stream = torch.cuda.Stream(device=self.env.device)
        stream.wait_stream(torch.cuda.current_stream(self.env.device))
        try:
            with torch.cuda.stream(stream):
                self._round(record=record)
            torch.cuda.current_stream(self.env.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self._round(record=record)
            torch.cuda.current_stream(self.env.device).wait_stream(stream)
            self._graphs[record] = (graph, output)
        finally:
            self._execute.fill_(True)

    def round(self, *, record=False):
        if self.env._closed:
            raise RuntimeError('GpuEnv is closed')
        if record in self._graphs:
            graph, output = self._graphs[record]
            graph.replay()
            return output
        return self._round(record=record)


class RoutedGpuActor(GpuActor):
    """Freeze game/seat routing for all stages of one external decision."""
    def __init__(self, model, env, *, opponents, opponent_seat, opponent_policy,
                 sample=True, dtype='bf16'):
        super().__init__(model, env, sample=sample, dtype=dtype)
        self.opponent_seat, self.opponent_policy = opponent_seat, opponent_policy
        self.opponents = {key: GpuActor(policy, env, sample=sample, dtype=dtype)
                          for key, policy in opponents.items()}

    def encode(self, active=None):
        route = torch.where(self.env.actor_seat == self.opponent_seat,
                            self.opponent_policy, -1)
        policies = [(-1, self), *self.opponents.items()]
        encoded = []
        for key, actor in policies:
            rows = route == key
            features = GpuActor.encode(actor, rows)
            values = actor.sparse_heads.values(features) if actor.sparse_heads is not None else None
            encoded.append((actor, rows, features, values))
        return encoded

    def act(self, encoded, active=None, value_logits=None):
        lp = torch.zeros((self.env.num_envs, 21), device=self.env.device)
        value = torch.zeros((self.env.num_envs, 3), device=self.env.device)
        for actor, rows, features, cached_values in encoded:
            live = rows if active is None else rows & active
            if actor.sparse_heads is not None:
                probs = actor.sparse_heads(features, live)
                values = cached_values
            else:
                with actor._autocast():
                    probs, values = actor.model.heads(features, **{
                        k: getattr(self.env, k) for k in PREFIX_FIELDS})
            lp = torch.where(rows[:, None], probs, lp)
            value = torch.where(rows[:, None], values, value)
        choices = torch.multinomial(lp.exp(), 1).squeeze(1) if self.sample else lp.argmax(-1)
        return choices.to(torch.int32), lp, value

    def refresh(self):
        super().refresh()
        # Opponent snapshots are immutable and have their own cached weights.


class ParallelGpuActor:
    """Fork/join decision rounds over disjoint views in one CUDA Graph."""
    capture=GpuActor.capture
    round=GpuActor.round

    def __init__(self,model,env,*,streams=2,sample=True,dtype='bf16',
                 opponents=None,opponent_seat=None,opponent_policy=None):
        if not 1<=streams<=env.num_envs:
            raise ValueError('streams must be in 1..num_envs')
        self.model,self.env,self.dtype=model,env,dtype
        self._graphs={}
        self._execute=torch.ones((),dtype=torch.bool,device=env.device)
        self.actors=[]
        self.streams=[]
        for i in range(streams):
            start=env.num_envs*i//streams;stop=env.num_envs*(i+1)//streams
            view=env._slice(start,stop)
            actor=(GpuActor(model,view,sample=sample,dtype=dtype) if not opponents else
                   RoutedGpuActor(model,view,sample=sample,dtype=dtype,opponents=opponents,
                                  opponent_seat=opponent_seat[start:stop],
                                  opponent_policy=opponent_policy[start:stop]))
            actor._execute=self._execute
            self.actors.append(actor)
            self.streams.append(torch.cuda.Stream(device=env.device))
        self.packed_encoder=self.actors[0].packed_encoder

    def refresh(self):
        for actor in self.actors:actor.refresh()

    @torch.inference_mode()
    def _round(self,*,record=False):
        current=torch.cuda.current_stream(self.env.device)
        outputs=[]
        for actor,stream in zip(self.actors,self.streams):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):outputs.append(actor._round(record=record))
        for stream in self.streams:current.wait_stream(stream)
        finished=torch.cat([x[0] for x in outputs])
        data=None
        if record:
            pieces=[x[1] for x in outputs]
            data={k:torch.cat([p[k] for p in pieces]) for k in pieces[0] if k!='stages'}
            data['stages']=[{k:torch.cat([p['stages'][stage][k] for p in pieces])
                             for k in pieces[0]['stages'][stage]} for stage in range(MAX_POLICY_STAGES)]
        return finished,data
