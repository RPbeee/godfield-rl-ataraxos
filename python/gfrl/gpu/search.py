"""Device-resident information-set search with CPU SearchAgent RNG semantics.

World state, public histories, candidate paths/prefixes, policy actions and leaf
observations stay on device. Small counters, selection probabilities (NumPy's
established selection stream), and final public distillation records are read
by the host. No hidden label buffer is an input to this module.
"""
from collections import Counter
import os
from contextlib import nullcontext, contextmanager
from time import perf_counter

import numpy as np
import torch

from ..actor import PREFIX_FIELDS, synchronize
from ..belief import IndependentBelief
from ..belief_metrics import Baseline
from ..evaluation import _policy_rng
from ..model import OBS_FIELDS
from ..search import SearchConfig, SearchAgent, OracleBelief, _SUPPORT
from .search_env import GpuSearchEnv, GpuCandidateEnv
from .jit import pointer
import ctypes as ct

MAX_STAGES = 22


def public(env, rows, fields=OBS_FIELDS):
    if hasattr(env,'public') and fields==OBS_FIELDS:return env.public(rows,fields)
    return {k: getattr(env, k).index_select(0, rows.long()) for k in fields}


def retain_candidates(paths, lp, lr, valid, limit):
    """CPU dict's last duplicate wins; then (-log_pi, lexicographic path)."""
    n, samples, width = paths.shape
    same = (paths[:, :, None] == paths[:, None]).all(-1)
    later = torch.arange(samples, device=paths.device)
    keep = valid & ~(same & (later[None, :] > later[:, None]) & valid[:, None, :]).any(-1)
    order = torch.arange(samples, device=paths.device).expand(n, -1)
    for stage in range(width-1, -1, -1):
        keys = paths[:, :, stage].gather(1, order)
        order = order.gather(1, keys.argsort(dim=1, stable=True))
    scores = lp.masked_fill(~keep, -torch.inf).gather(1, order)
    order = order.gather(1, scores.argsort(dim=1, descending=True, stable=True))[:, :limit]
    return (paths.gather(1, order[:, :, None].expand(-1, -1, width)), lp.gather(1, order),
            lr.gather(1, order), keep.gather(1, order), order)


def distribution(q, lr, lp, valid, alpha, beta):
    """Float64 ragged normalization; invalid candidates have exactly zero mass."""
    lr, lp, q = lr.double(), lp.double(), q.double()
    def normalize(x):
        x = x.masked_fill(~valid, -torch.inf)
        return x-torch.logsumexp(x, -1, keepdim=True)
    logits = (q + alpha*normalize(lr) + beta*normalize(lp))/(alpha+beta)
    # 0 * -inf must not poison zero-regularization configurations.
    if alpha == 0:
        logits = (q+beta*normalize(lp))/beta
    if beta == 0:
        logits = (q+alpha*normalize(lr))/alpha
    good = valid.any(-1)
    logits = logits.masked_fill(~valid, -torch.inf)
    logits = torch.where(good[:, None], logits, torch.zeros_like(logits))
    return logits.softmax(-1).masked_fill(~valid, 0)


class DeviceActor:
    """Same active-row multinomial calls as _SearchActor, with swappable inference.

    Encode only fresh decisions, deduplicate world copies at the root, and
    retain only the 21 tokens read by policy/value heads. No host cache spill.
    """
    def __init__(self, model, env, cap=0, seconds=None, phase='policy', encode=None, heads=None):
        self.model, self.env, self.cap = model, env, cap
        self.encode = encode or model.encode
        self.heads = heads or (lambda encoded,prefix:model.heads(encoded,**prefix)[0].float())
        self.cache = None
        self.seconds, self.phase = seconds, phase

    def _mark(self, key, start):
        if self.seconds is not None:
            synchronize(self.env.device)
            self.seconds[self.phase+'_'+key] += perf_counter()-start
        return perf_counter()

    @torch.inference_mode()
    def act(self, rows, groups=None):
        e = self.env
        fresh = rows[e.substep[rows] == 0]
        if len(fresh):
            encode = fresh
            inverse = None
            if groups is not None:
                # groups are in monotonic root order; stable first occurrence.
                unique, inverse = torch.unique(groups[fresh], sorted=True, return_inverse=True)
                first = torch.full((len(unique),), len(fresh), device=e.device, dtype=torch.long)
                first.scatter_reduce_(0, inverse, torch.arange(len(fresh), device=e.device), reduce='amin')
                encode = fresh[first]
            start = perf_counter()
            parts = []
            cap = self.cap or len(encode)
            for lo in range(0, len(encode), cap):
                parts.append(self.encode(public(e, encode[lo:lo+cap]))[:, :21])
            encoded = torch.cat(parts)
            if inverse is not None:
                encoded = encoded[inverse]
            if self.cache is None:
                self.cache = encoded.new_empty((e.num_envs, 21, encoded.shape[-1]))
            self.cache[fresh] = encoded
            self._mark('encoder', start)
        start = perf_counter()
        cap = self.cap or max(len(rows), 1)
        parts = [self.heads(self.cache[rr], public(e, rr, PREFIX_FIELDS))
                 for lo in range(0,len(rows),cap) for rr in (rows[lo:lo+cap],)]
        lp = torch.cat(parts)
        action = torch.multinomial(lp.exp(), 1).flatten()
        choice = torch.zeros(e.num_envs, device=e.device, dtype=torch.int32)
        choice[rows] = action.int()
        self._mark('heads', start)
        return choice, lp.gather(1, action[:, None]).flatten()


class GpuSearchAgent(SearchAgent):
    """Both-seat CUDA search. CPU SearchAgent remains unchanged.

    env_factory/device='cpu' is used exclusively by the generated-source native
    differential tests. The production default requires CUDA.
    """
    def __init__(self, model, belief, device='cuda', *, config=SearchConfig(), seed=0,
                 profile=False, record=False, env_factory=GpuSearchEnv, dtype='fp32', packed_record=False):
        if isinstance(belief, OracleBelief):
            raise ValueError('oracle belief is forbidden in GPU search')
        from ..search_belief import HybridBelief, MixedBelief
        if isinstance(belief, (HybridBelief, MixedBelief)):
            raise ValueError('GPU search currently supports AR, independent and baseline beliefs without mixture/hybrid')
        if config.search_cache_device == 'host':
            raise ValueError('GPU search requires a device cache')
        self.model, self.belief = model.to(device).eval(), belief
        self.device, self.config = torch.device(device), config
        self.rng = np.random.default_rng(seed)
        self.profile, self.record, self.env_factory = profile, record, env_factory
        self.counts, self.seconds, self.world_load_counts = Counter(), Counter(), Counter()
        self.records, self.last_search, self.last_candidates, self.last_policy_paths = [], [], {}, {}
        self.last_candidate_diagnostics = {}
        self.env, self.actor = None, None
        self._workspaces = {}
        self.packed_record=packed_record
        self.record_blocks=[]
        self._packed_count=0
        self._record_decisions = None
        if dtype not in ('fp32','bf16'):raise ValueError('search dtype must be fp32 or bf16')
        self.dtype=dtype
        self.fast_heads=None
        self.belief_encoder=None
        self._belief_graphs={}
        self.packed_encoder = None
        if (self.device.type == 'cuda' and dtype=='bf16'
                and self.model.config.d//self.model.config.heads >= 32):
            from .search_inference import PackedSearchEncoder
            self.packed_encoder = PackedSearchEncoder(self.model,torch.bfloat16 if dtype=='bf16' else torch.float32)
            if dtype=='bf16':
                from .search_inference import Bf16HeadGraphs
                self.fast_heads=Bf16HeadGraphs(self.model)
                if not isinstance(belief,(Baseline,IndependentBelief)):
                    belief_model=getattr(belief,'model',belief)
                    self.belief_encoder=PackedSearchEncoder(belief_model,torch.bfloat16,full=True)

    @property
    def record_count(self):
        return self._packed_count if self.packed_record else len(self.records)

    def _time(self):
        synchronize(self.device)
        return perf_counter()

    def close(self):
        for env in self._workspaces.values():
            env.close()
        self._workspaces.clear()

    def statistics(self):
        out = super().statistics()
        out.update(backend='cuda', host_reads='public records, scalar scheduling/counters, NumPy selection probabilities')
        return out

    def _world(self, src, rows, assignments, seeds, depth, role, clone_indices=None):
        start = self._time()
        actual=len(rows)
        capacity=(((actual+1023)//1024)*1024 if actual>=1024 else 1 << (actual-1).bit_length()) if self.device.type=='cuda' else actual
        key=(role,capacity,depth,src.history_len)
        for old in list(self._workspaces):
            if old[0]==role:
                if (old[2:]==key[2:] and capacity<=old[1]<=4*capacity):key=old
                elif old!=key:self._workspaces.pop(old).close()
        if key not in self._workspaces:
            extra={}
            if self.device.type=='cuda':
                extra['arena_bytes']=int(os.environ.get('GFRL_GPU_SEARCH_ARENA_BYTES','524288'))
                needed=key[1]*(extra['arena_bytes']+65536)
                free,_=torch.cuda.mem_get_info(self.device)
                if free<needed:torch.cuda.empty_cache()
            self._workspaces[key]=self.env_factory(key[1],history_len=src.history_len,
                canonical=src.canonical,max_decisions=depth,device=self.device,pooled=True,**extra)
        env=self._workspaces[key]
        if clone_indices is None and env.num_envs>actual:
            padded_rows=rows.new_zeros(env.num_envs);padded_rows[:actual]=rows
            padded_assignments=assignments.new_zeros((env.num_envs,297));padded_assignments[:actual]=assignments
            padded_assignments[actual:,0]=1
            padded_seeds=torch.zeros(env.num_envs,device=self.device,dtype=torch.int64)
            padded_seeds[:actual]=seeds.view(torch.int64)
            rows,assignments,seeds=padded_rows,padded_assignments,padded_seeds.view(torch.uint64)
        self.seconds['world_workspace'] += self._time()-start
        start = self._time()
        if clone_indices is None:
            status=env.load_worlds(src,rows.int(),assignments.int(),seeds)[:actual].clone()
        else:
            indices=torch.zeros(env.num_envs,device=self.device,dtype=torch.int32)
            indices[:actual]=clone_indices.int()
            env._module.launch('gpu_search_clone',env._args()+[pointer(self._base_world._storage),pointer(self._base_world._pointers),
                pointer(indices),pointer(env.world_status),ct.c_int(actual)],n=env.num_envs)
            env._observe();env._actors()
            status=env.world_status[:actual].clone()
        self.seconds['world_load'] += self._time()-start
        self.counts['invalid_loads'] += int((status != 0).sum())
        names = ('ok', 'invalid_assignment', 'unsupported_root', 'empty_own_prior',
                 'observation_mismatch', 'decision_mismatch', 'legality_mismatch')
        for value, count in zip(*torch.unique(status, return_counts=True)):
            self.world_load_counts[role+':'+names[int(value)]] += int(count)
        return env, status

    @contextmanager
    def _autocast(self):
        if self.dtype=='bf16':
            from torch.nn.attention import sdpa_kernel,SDPBackend
            with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
                yield
        else:yield

    def _belief_samples(self, src, rows):
        with self._autocast():
            return self._belief_samples_impl(src,rows)

    def _belief_samples_impl(self, src, rows):
        b, cap = self.belief, self.config.max_inference_batch
        if isinstance(b, Baseline):
            pn, pk = b.count_probs.numpy(), b.card_probs.numpy()
            return torch.tensor(np.array([self.rng.multinomial(self.rng.choice(19,p=pn),pk)
                                for _ in rows],np.int32),device=self.device)
        cached=getattr(self,'_belief_memory',None)
        if not cap and cached is None:
            return b.sample(public(src, rows), 1)[:, 0].int()
        cap=cap or len(rows)
        # Same inference chunking and draws as SearchAgent; memory stays on CUDA.
        independent = isinstance(b, IndependentBelief)
        count_probs, card_probs, chunks = [], [], []
        owner_parts=[]
        for lo in range(0,len(rows),cap):
            obs = public(src, rows[lo:lo+cap]) if independent or cached is None else None
            if independent:
                pn, pk = b._log_probs(obs)
                count_probs.append(pn.exp());card_probs.append(pk.exp())
            else:
                if cached is None:
                    cls, memory, padding = b.memory(obs)
                else:
                    root_ids,root_cls,root_memory,root_padding=cached
                    owners=(rows[lo:lo+cap,None]==root_ids[None,:]).long().argmax(1)
                    cls=root_cls[owners]
                    if self.dtype=='bf16' and getattr(self,'_belief_cross',None) is not None:
                        memory,padding=root_memory,root_padding
                        owner_parts.append(owners)
                    else:
                        padding=root_padding[owners]
                        width=int((~padding).sum(1).max())
                        memory=root_memory[owners,:width];padding=padding[:,:width]
                count_probs.append(b.count_head(cls).float().softmax(-1))
                chunks.append((memory,padding))
        counts = torch.multinomial(torch.cat(count_probs),1,replacement=True).flatten()
        result = torch.zeros((len(rows),297),dtype=torch.long,device=self.device)
        if independent:
            cards = torch.multinomial(torch.cat(card_probs),18,replacement=True)+1
            result.scatter_add_(1,cards,(torch.arange(18,device=self.device)<counts[:,None]).long())
        else:
            steps=int(counts.max())
            if not steps:return result.int()
            from torch.nn.functional import pad
            width = max(m.shape[1] for m,_ in chunks)
            chunks = [(pad(m,(0,0,0,width-m.shape[1])),pad(p,(0,width-p.shape[1]),value=True)) for m,p in chunks]
            inputs = torch.zeros((len(rows),1),dtype=torch.long,device=self.device)
            incremental=None
            if self.device.type=='cuda' and os.environ.get('GFRL_GPU_SEARCH_BELIEF_KV','0')=='1':
                from .search_inference import IncrementalBelief
                incremental=[]
                for i,(memory,padding) in enumerate(chunks):
                    count=counts[i*cap:(i+1)*cap]
                    shared=(*self._belief_cross,owner_parts[i]) if owner_parts else None
                    if (self.dtype=='bf16' and shared is not None and len(count)>0
                            and memory.shape[1]<=256 and os.environ.get('GFRL_GPU_SEARCH_BELIEF_GRAPH','0')=='1'
                            and os.environ.get('GFRL_GPU_BELIEF_COMPACT','0')!='1'):
                        from .search_inference import Bf16BeliefGraph
                        key=(1<<(len(count)-1).bit_length(),1<<(len(memory)-1).bit_length(),
                             max(128,1<<(memory.shape[1]-1).bit_length()),i)
                        if key not in self._belief_graphs:
                            self._belief_graphs[key]=Bf16BeliefGraph(b,*key[:3])
                        decoder=self._belief_graphs[key].begin(count,shared)
                    else:decoder=IncrementalBelief(b,memory,padding,count,shared=shared)
                    incremental.append(decoder)
            for t in range(steps):
                if incremental is not None and self.dtype=='bf16' and os.environ.get('GFRL_GPU_BELIEF_COMPACT','0')=='1':
                    active=[];parts=[]
                    for i,decoder in enumerate(incremental):
                        local=torch.nonzero(counts[i*cap:(i+1)*cap]>t,as_tuple=True)[0]
                        if not len(local):continue
                        active.append(local+i*cap)
                        parts.append(decoder.step(inputs[local+i*cap,-1],rows=local))
                    # Every decoder keeps the shared causal position even after
                    # all its rows have finished; no row can become active again.
                    for decoder in incremental:decoder.position=t+1
                    ids=torch.cat(active)
                    card=torch.multinomial(torch.cat(parts).exp(),1)+1
                    result[ids,card[:,0]]+=1
                    full=torch.zeros((len(rows),1),device=self.device,dtype=torch.long);full[ids]=card
                    inputs=torch.cat((inputs,full),1)
                    continue
                if incremental is None:
                    lp = torch.cat([b.decode(m,p,inputs[i*cap:(i+1)*cap],counts[i*cap:(i+1)*cap])[:,-1]
                                    for i,(m,p) in enumerate(chunks)])
                else:
                    lp = torch.cat([decoder.step(inputs[i*cap:(i+1)*cap,-1]) for i,decoder in enumerate(incremental)])
                card = torch.multinomial(lp.exp(),1)+1
                result.scatter_add_(1,card,(counts>t).long()[:,None]);inputs=torch.cat((inputs,card),1)
        return result.int()

    def _belief_memory_for(self,src,roots):
        obs=public(src,roots)
        if self.belief_encoder is None:
            with self._autocast():return self.belief.memory(obs)
        encoded=self.belief_encoder.encode(obs)
        valid=obs['obs_valid'].bool()
        width=int(valid.sum(1).max())
        slots=torch.argsort((~valid).int(),dim=1,stable=True)[:,:width]
        return encoded[:,0],encoded.gather(1,slots[...,None].expand(-1,-1,encoded.shape[-1])),~valid.gather(1,slots)

    def _sample_worlds(self, src, roots):
        c = self.config
        self._belief_memory=None
        self._belief_cross=None
        if (self.device.type=='cuda' and os.environ.get('GFRL_GPU_SEARCH_BELIEF_CACHE','0')=='1'
                and not isinstance(self.belief,(Baseline,IndependentBelief))):
            self._belief_memory=(roots,*self._belief_memory_for(src,roots))
            if self.dtype=='bf16':
                from .search_inference import IncrementalBelief
                model=getattr(self.belief,'model',self.belief)
                memory,padding=self._belief_memory[2:]
                pad=(-memory.shape[1])%8
                if pad:
                    memory=torch.nn.functional.pad(memory,(0,0,0,pad))
                    padding=torch.nn.functional.pad(padding,(0,pad),value=True)
                mask=memory.new_zeros((len(memory),1,1,memory.shape[1])).masked_fill(padding[:,None,None,:],-torch.inf)
                kvs=[]
                with self._autocast():
                    for layer in model.decoder.layers:
                        attention=layer.multihead_attn;d=attention.embed_dim
                        kv=torch.nn.functional.linear(memory,attention.in_proj_weight[d:],attention.in_proj_bias[d:])
                        k,v=kv.chunk(2,-1)
                        kvs.append((IncrementalBelief.heads(k,attention),IncrementalBelief.heads(v,attention)))
                self._belief_cross=(kvs,mask)
        rr = roots.repeat_interleave(c.worlds)
        assignments = torch.zeros((len(rr),297),dtype=torch.int32,device=self.device)
        valid = torch.zeros(len(rr),dtype=torch.bool,device=self.device)
        attempts = torch.zeros(len(rr),dtype=torch.int32,device=self.device)
        for _ in range(c.max_attempts):
            todo = torch.nonzero(~valid,as_tuple=True)[0]
            if not len(todo):break
            samples = self._belief_samples(src,rr[todo])
            ok = src.validate_assignments(rr[todo].int(),samples)
            attempts[todo] += 1
            self.counts['invalid_samples'] += int((~ok).sum())
            assignments[todo[ok]] = samples[ok];valid[todo[ok]]=True
        self.counts['resamples'] += int((attempts-1).clamp_min(0).sum())
        self.counts['exhausted_worlds'] += int((~valid).sum())
        # Keep the established NumPy world-chance stream, including rejected slots.
        seeds = torch.from_numpy(self.rng.integers(0,np.iinfo(np.uint64).max,len(rr),dtype=np.uint64)).to(self.device)
        return rr[valid], assignments[valid], seeds.view(torch.int64)[valid].view(torch.uint64)

    def _heads(self,encoded,prefix):
        if self.fast_heads is not None:return self.fast_heads(encoded,prefix)
        with self._autocast():return self.model.heads(encoded,**prefix)[0].float()

    def _values(self,encoded):
        if self.fast_heads is not None:return self.fast_heads.values(encoded)
        with self._autocast():return self.model.value_head(encoded[:,0]).float()

    def _encode(self, obs):
        if self.packed_encoder is not None:return self.packed_encoder.encode(obs)
        with self._autocast():return self.model.encode(obs)

    def _actor(self, env, phase):
        return DeviceActor(self.model, env, self.config.max_inference_batch,
                           self.seconds if self.profile else None, phase, self._encode, self._heads)

    @torch.inference_mode()
    def search(self, src, roots, *, fixed_candidates=None, policy_paths=None, report_values=False):
        if self.record and fixed_candidates is not None:
            raise ValueError('recording requires candidate generation')
        roots = torch.as_tensor(roots,device=self.device,dtype=torch.long)
        support = src.determinize_support(roots.int())
        roots = roots[support==0]
        phase = src.obs_schema()['cat']['phase']['index']
        response = src.obs_cat[roots,0,phase]==2
        selected = {}
        reports, candidates, policies = [], {}, {}
        for kind, take in (('main',~response),('response',response)):
            if kind=='response' and not self.config.search_response:continue
            rs = roots[take];cap=self.config.root_batch_size or max(len(rs),1)
            start=self._time()
            for lo in range(0,len(rs),cap):
                selected.update(self._search(src,rs[lo:lo+cap],fixed_candidates,policy_paths,report_values))
                reports.extend(self.last_search);candidates.update(self.last_candidates);policies.update(self.last_policy_paths)
            self.seconds['total_'+kind]+=self._time()-start
            self.counts['searched_'+kind]+=len(rs)
        self.last_search,self.last_candidates,self.last_policy_paths=reports,candidates,policies
        return selected

    def _search(self,src,roots,fixed_candidates,policy_paths,report_values):
        c=self.config;dev=self.device;n=len(roots)
        self.counts['search_batches']+=1;self.counts['batch_roots']+=n
        self.counts['max_batch_roots']=max(n,self.counts['max_batch_roots'])
        self.last_search=[];self.last_candidates={};self.last_policy_paths={}
        start=self._time()
        rr,assignments,seeds=self._sample_worlds(src,roots)
        self.seconds['sample']+=self._time()-start
        if not len(rr):return {}
        root_owners=(rr[:,None]==roots[None,:]).long().argmax(1)
        sizes=torch.bincount(root_owners,minlength=n).cpu().tolist()
        offsets=np.cumsum([0]+sizes)
        groups=[torch.arange(int(offsets[i]),int(offsets[i+1]),device=dev) for i in range(n)]
        self._base_world=None
        if self.device.type=='cuda':
            self._base_world,base_status=self._world(src,rr,assignments,seeds,1,'base')
            if self.record:
                good=base_status==0
                for k in OBS_FIELDS:
                    if bool((getattr(self._base_world,k)[:len(rr)][good]!=getattr(src,k)[rr[good]]).any()):
                        raise RuntimeError('determinized base observation differs from public root')
        start=self._time()
        root_obs=public(src,roots) if self.record else None
        prefixes=None
        if fixed_candidates is None:
            present=torch.tensor([i for i,g in enumerate(groups) if len(g)],device=dev,dtype=torch.long)
            prefix_roots={int(i):j for j,i in enumerate(present.cpu().tolist())}
            indices=torch.cat([g[torch.arange(c.candidate_samples,device=dev)%len(g)] for g in groups if len(g)])
            if self._base_world is None:
                env,status=self._world(src,rr[indices],assignments[indices],seeds.view(torch.int64)[indices].view(torch.uint64),1,'candidate')
            else:
                env=GpuCandidateEnv(self._base_world,indices,src,rr[indices])
                status=self._base_world.world_status[indices].clone()
            valid=(status==0).reshape(len(present),-1)
            if not bool(valid.any()):
                if isinstance(env,GpuCandidateEnv):env.close()
                self.last_candidates={int(r):[] for r in roots.cpu().tolist()}
                self.seconds['candidate']+=self._time()-start
                return {}
            if self.record:
                mismatch=torch.zeros((),dtype=torch.bool,device=dev)
                good=status==0
                observed=public(env,torch.nonzero(good,as_tuple=True)[0])
                for k in OBS_FIELDS:
                    mismatch |= (observed[k]!=getattr(src,k)[rr[indices][good]]).any()
                if bool(mismatch):
                    raise RuntimeError('determinized candidate observation differs from public root')
            actor=self._actor(env,'candidate')
            paths=torch.full((len(indices),MAX_STAGES),-1,dtype=torch.int32,device=dev)
            lengths=torch.zeros(len(indices),dtype=torch.long,device=dev)
            lp=torch.zeros(len(indices),dtype=torch.float64,device=dev);lr=torch.zeros_like(lp)
            prefixes={k:[] for k in PREFIX_FIELDS} if self.record else None
            first=True
            while True:
                active=torch.nonzero(env.stage>=0,as_tuple=True)[0]
                if not len(active):break
                if self.record:
                    # Prefix row dimension remains in sample order; gather retained paths once.
                    for k in PREFIX_FIELDS:prefixes[k].append(getattr(env,k).clone())
                choices,picked=actor.act(active,rr[indices] if first else None);first=False
                paths[active,lengths[active]]=choices[active];lengths[active]+=1
                lp[active]+=picked.double();lr[active]-=env.mask[active].sum(1).double().log()
                env.step(choices)
            sampled_paths=paths.reshape(len(present),-1,MAX_STAGES)
            sampled_valid=valid.clone()
            paths,lp,lr,valid,order=retain_candidates(sampled_paths,lp.reshape(len(present),-1),lr.reshape(len(present),-1),valid,c.candidates)
            first_indices=sampled_valid.int().argmax(1)
            first_paths=sampled_paths[torch.arange(len(present),device=dev),first_indices].cpu().tolist()
            live_roots=sampled_valid.any(1).cpu().tolist()
            self.last_policy_paths={int(r):tuple(a for a in p if a>=0)
                for r,p,live in zip(roots[present].cpu().tolist(),first_paths,live_roots) if live}
            if len(present)!=n:
                def padded(x,fill=0):
                    out=x.new_full((n,*x.shape[1:]),fill);out[present]=x;return out
                paths,lp,lr,valid,order=padded(paths,-1),padded(lp),padded(lr),padded(valid,False),padded(order)
            if self.record:
                prefixes={k:torch.stack(v,1) for k,v in prefixes.items()}
            if isinstance(env,GpuCandidateEnv):env.close()
        else:
            width=min(c.candidates,max((len(fixed_candidates.get(int(r),[])) for r in roots),default=0))
            if not width:return {}
            paths=torch.full((n,width,MAX_STAGES),-1,dtype=torch.int32,device=dev)
            lp=torch.zeros((n,width),dtype=torch.float64,device=dev);lr=torch.zeros_like(lp)
            valid=torch.zeros((n,width),dtype=torch.bool,device=dev)
            for i,r in enumerate(roots.tolist()):
                for j,(p,(a,b)) in enumerate(fixed_candidates.get(r,[])[:width]):
                    paths[i,j,:len(p)]=torch.tensor(p,device=dev);lp[i,j]=a;lr[i,j]=b;valid[i,j]=True
            order=None
        self.seconds['candidate']+=self._time()-start
        width=paths.shape[1]
        self.counts['candidate_paths']+=int(valid.sum())
        singleton=valid.sum(1)==1 if c.skip_single_candidate and not report_values else torch.zeros(n,dtype=torch.bool,device=dev)
        self.counts['singleton_roots']+=int(singleton.sum())
        self.counts['rollout_roots']+=int((valid.any(1)&~singleton).sum())
        q=torch.zeros((n,width),dtype=torch.float64,device=dev)
        count=torch.zeros((n,width),dtype=torch.long,device=dev)
        world_values={}
        world_counts=torch.tensor([len(g) for g in groups],device=dev)
        world_offsets=world_counts.cumsum(0)-world_counts
        for fast in (True,False):
            job_ids=torch.nonzero((valid & (singleton[:,None]==fast) & (world_counts[:,None]>0)).flatten(),as_tuple=True)[0]
            if not len(job_ids):continue
            repeats=world_counts[job_ids//width]
            owner=torch.repeat_interleave(job_ids,repeats)
            starts=torch.repeat_interleave(repeats.cumsum(0)-repeats,repeats)
            wi=world_offsets[owner//width]+torch.arange(len(owner),device=dev)-starts
            forced=paths.reshape(-1,MAX_STAGES)[owner]
            values,ok=self._rollouts(src,rr,assignments,seeds,wi,forced,fast)
            count.view(-1).index_add_(0,owner,ok.long())
            q.view(-1).index_add_(0,owner,values.double().masked_fill(~ok,0))
            if report_values:
                vv,oo,kk=values.cpu().tolist(),owner.cpu().tolist(),ok.cpu().tolist()
                for v,o,k in zip(vv,oo,kk):
                    if k:world_values.setdefault(o,[]).append(v)
        q/=count.clamp_min(1)
        surviving=valid&(count>0)
        probabilities=distribution(q,lr,lp,surviving,c.alpha,c.beta)
        start=self._time()
        # Public report transfer is also the record flush boundary. No state or
        # hidden inventory is copied to the host for candidate/rollout simulation.
        hp,hlp,hlr,hq,hc,hv,probs,hvalid,hsingle=[x.cpu().numpy() for x in (paths,lp,lr,q,count,surviving,probabilities,valid,singleton)]
        root_ids=roots.cpu().tolist();selected={};reports=[]
        if self.record and self.packed_record:
            from .search_records import record_block
            block=record_block(src,roots,root_obs,prefixes,prefix_roots,order,paths,lp,lr,q,count,surviving,singleton,
                               self._record_decisions,c.candidate_samples)
            if block is not None:
                block['probabilities']=probabilities[surviving].cpu()
                self.record_blocks.append(block);self._packed_count+=len(block['game'])
        hprefix={k:v.cpu().numpy() for k,v in prefixes.items()} if prefixes is not None and not self.packed_record else None
        horder=order.cpu().numpy() if order is not None else None
        hobs={k:v.cpu().numpy() for k,v in root_obs.items()} if root_obs is not None and not self.packed_record else None
        if self.record and not self.packed_record:
            hgames=src.episode_id.view(torch.int64)[roots].cpu().tolist()
            hseats=src.actor_seat[roots].cpu().tolist()
            hdecisions=self._record_decisions[roots].cpu().tolist() if self._record_decisions is not None else [0]*n
            phase_index=src.obs_schema()['cat']['phase']['index']
        for i,r in enumerate(root_ids):
            items=[];kept=np.flatnonzero(hv[i])
            if not self.packed_record or report_values:
                self.last_candidates[r]=[(tuple(int(a) for a in hp[i,j] if a>=0),(float(hlp[i,j]),float(hlr[i,j]))) for j in np.flatnonzero(hvalid[i])]
            if not len(kept):continue
            p=probs[i,kept];p=p/p.sum()
            pick=int(p.argmax()) if c.select=='argmax' else int(self.rng.choice(len(kept),p=p))
            if self.packed_record and not report_values:
                selected[r]=tuple(int(a) for a in hp[i,kept[pick]] if a>=0)
                self.counts['selected']+=1
                if policy_paths is not None:self.last_policy_paths[r]=policy_paths.get(r,())
                self.counts['policy_disagreements']+=selected[r]!=self.last_policy_paths.get(r,())
                continue
            for j in kept:
                path=[int(a) for a in hp[i,j] if a>=0]
                item=dict(path=path,log_pi=float(hlp[i,j]),log_rho=float(hlr[i,j]),q=None if hsingle[i] else float(hq[i,j]),valid_worlds=int(hc[i,j]))
                if report_values:
                    v=world_values[i*width+int(j)];item.update(world_values=v,q_se=float(np.std(v,ddof=1)/np.sqrt(len(v))) if len(v)>1 else None)
                items.append(item)
            selected[r]=tuple(items[pick]['path'])
            reports.append(dict(row=r,candidates=items,probabilities=p.tolist(),selected=list(selected[r])))
            self.counts['selected']+=1
            if policy_paths is not None:
                self.last_policy_paths[r]=policy_paths.get(r,())
            self.counts['policy_disagreements']+=selected[r]!=self.last_policy_paths.get(r)
            if report_values:reports[-1]['policy_path']=list(self.last_policy_paths.get(r,()))
            if hsingle[i]:reports[-1]['rollout_skipped']='single_candidate'
            if self.record and not self.packed_record:
                assert hprefix is not None
                candidates=[]
                for j,item in zip(kept,items):
                    sample=prefix_roots[i]*c.candidate_samples+int(horder[i,j])
                    candidates.append(dict(item,prefixes=[{k:v[sample,t].copy() for k,v in hprefix.items()} for t in range(len(item['path']))]))
                self.records.append(dict(obs={k:v[i].copy() for k,v in hobs.items()},game=hgames[i],seat=hseats[i],
                    phase='response' if hobs['obs_cat'][i,0,phase_index]==2 else 'main',
                    decision=hdecisions[i],
                    probabilities=p.tolist(),candidates=candidates))
        self.last_search=reports
        self.seconds['records_and_selection']+=self._time()-start
        return selected

    def _rollouts(self,src,rr,assignments,seeds,wi,paths,singleton):
        c=self.config;start=self._time()
        env,status=self._world(src,rr[wi],assignments[wi],seeds.view(torch.int64)[wi].view(torch.uint64),1 if singleton else c.depth+1,'singleton' if singleton else 'rollout',clone_indices=wi if self._base_world is not None else None)
        actor=self._actor(env,'rollout');valid=status==0
        cursor=torch.zeros(len(wi),dtype=torch.long,device=self.device);lengths=(paths>=0).sum(1)
        first=True
        while True:
            active=torch.nonzero((env.stage[:len(wi)]>=0)&valid,as_tuple=True)[0]
            if not len(active):break
            forcing=active[env.decisions.view(torch.int64)[active]==0]
            infer=active[env.decisions.view(torch.int64)[active]!=0] if c.skip_forced_inference or singleton else active
            self.counts['forced_inference_rows_skipped']+=len(active)-len(infer)
            if len(infer):choices,_=actor.act(infer,rr[wi] if first and not c.skip_forced_inference and not singleton else None)
            else:choices=torch.zeros(env.num_envs,dtype=torch.int32,device=self.device)
            first=False
            chosen=paths[forcing,cursor[forcing].clamp_max(MAX_STAGES-1)].long()
            ok=(cursor[forcing]<lengths[forcing])&env.mask[forcing,chosen.clamp_min(0)]
            valid[forcing[~ok]]=False;self.counts['invalid_candidate_worlds']+=int((~ok).sum())
            accepted=forcing[ok];choices[accepted]=chosen[ok].int();cursor[accepted]+=1
            invalid=torch.nonzero((env.stage[:len(wi)]>=0)&~valid,as_tuple=True)[0]
            choices[invalid]=env.mask[invalid].int().argmax(1).int()
            before=env.decisions.view(torch.int64).clone();env.step(choices)
            invalid=forcing[valid[forcing]&(env.decisions.view(torch.int64)[forcing]>before[forcing])&(cursor[forcing]!=lengths[forcing])]
            valid[invalid]=False;self.counts['invalid_candidate_worlds']+=len(invalid)
        self.seconds['rollout']+=self._time()-start
        if self.device.type=='cuda' and env._pool is not None:
            used,capacity=env._pool[:16].view(torch.int64).tolist()
            self.counts['arena_peak_bytes_per_world']=max(self.counts['arena_peak_bytes_per_world'],(used+env.num_envs-1)//env.num_envs)
        if singleton:
            self.counts['singleton_validated_worlds']+=int(valid.sum())
            return torch.zeros(len(wi),device=self.device,dtype=torch.float64),valid
        self.counts['rollout_worlds']+=len(wi)
        start=self._time();env._actors();values=env.result[:len(wi)].double().clone()
        rows=torch.nonzero(valid,as_tuple=True)[0];acting=env.next_actor[rows]
        cutoff=acting>=0;selected=rows[cutoff];own=acting[cutoff]==env.searcher_seat[selected]
        self.counts['depth_leaves']+=len(selected);self.counts['depth_leaves_searcher']+=int(own.sum())
        self.counts['depth_leaves_opponent']+=int((~own).sum());self.counts['terminal_leaves']+=int((~cutoff).sum())
        seats=acting[cutoff] if c.leaf_view=='actor' else env.searcher_seat[selected]
        cap=c.max_inference_batch or max(len(selected),1)
        for lo in range(0,len(selected),cap):
            rs=selected[lo:lo+cap];obs=env._observe_rows(rs.int(),seats[lo:lo+cap].int())
            p=self._values(self._encode(obs)).softmax(-1);v=p[:,0]-p[:,1]
            if c.leaf_view=='actor':v=torch.where(own[lo:lo+cap],v,-v)
            values[rs]=v.double()
        self.seconds['leaf']+=self._time()-start
        return values,valid

    @torch.inference_mode()
    def __call__(self,env,rows=None):
        if env is not self.env:
            self.env=env;self.actor=self._actor(env,'real_policy')
            self.pending=torch.full((env.num_envs,MAX_STAGES),-1,dtype=torch.int32,device=self.device)
            self.pending_cursor=torch.zeros(env.num_envs,dtype=torch.long,device=self.device)
            self._record_decisions=torch.full((env.num_envs,),-1,dtype=torch.long,device=self.device)
            self._record_games=torch.full_like(self._record_decisions,-1)
        rows=torch.nonzero(env.stage>=0,as_tuple=True)[0] if rows is None else torch.as_tensor(rows,device=self.device,dtype=torch.long)
        if not len(rows):return torch.zeros(env.num_envs,dtype=torch.int32,device=self.device)
        start=self._time()
        choices,_=self.actor.act(rows)
        self.seconds['real_policy']+=self._time()-start
        fresh=rows[env.substep[rows]==0];self.counts['decisions']+=len(fresh)
        games=env.episode_id.view(torch.int64)[fresh]
        self._record_decisions[fresh]=torch.where(self._record_games[fresh]==games,self._record_decisions[fresh]+1,0)
        self._record_games[fresh]=games;self.pending[fresh]=-1;self.pending_cursor[fresh]=0
        if len(fresh):
            support=env.determinize_support(fresh.int());phase=env.obs_schema()['cat']['phase']['index']
            response=env.obs_cat[fresh,0,phase]==2
            if not self.config.search_response:
                support=torch.where(response&(support!=-1),3,support)
                self.counts['skip_response_disabled']+=int(response.sum())
            for reason,count in zip(*torch.unique(support[support!=0],return_counts=True)):
                self.counts['skip_'+_SUPPORT.get(int(reason),str(int(reason)))]+=int(count)
            roots=fresh[support==0]
            if len(roots):
                self.counts['searched']+=len(roots);self.counts['search_calls']+=1;start=self._time()
                with _policy_rng(int(self.rng.integers(2**31)),(self,)):
                    paths=self.search(env,roots)
                self.seconds['total']+=self._time()-start
                self.counts['fallback_no_candidates']+=len(roots)-len(paths)
                if paths:
                    matrix=np.full((len(paths),MAX_STAGES),-1,np.int32)
                    for i,path in enumerate(paths.values()):matrix[i,:len(path)]=path
                    ids=torch.tensor(list(paths),device=self.device,dtype=torch.long)
                    self.pending[ids]=torch.from_numpy(matrix).to(self.device)
        chosen=self.pending[rows,self.pending_cursor[rows].clamp_max(MAX_STAGES-1)]
        use=chosen>=0
        good=use&env.mask[rows,chosen.clamp_min(0).long()]
        choices[rows[good]]=chosen[good];self.pending_cursor[rows[good]]+=1
        self.counts['real_mismatches']+=int((use&~good).sum());self.pending[rows[use&~good]]=-1
        return choices
