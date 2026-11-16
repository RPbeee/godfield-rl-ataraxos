"""CUDA SearchEnv and public-history-tracked finite self-play roots.

No state import from CPU, no host simulation, and no privileged inference inputs.
Only explicit diagnostic snapshots and completed distillation records leave CUDA.
"""
import ctypes as ct
import os
from functools import lru_cache

import torch

from . import FIELDS, GpuEnv
from .jit import Module, pointer, _check
from .source import build_source


@lru_cache(maxsize=1)
def _search_module(device):
    import os
    torch.cuda.init()
    _check(ct.CDLL('libcudart.so').cudaDeviceSetLimit(
        0, ct.c_size_t(int(os.environ.get('GFRL_GPU_STACK_BYTES', '32768')))), 'search stack limit')
    module = Module(build_source(search=True))
    sizes = torch.zeros(3, dtype=torch.int64, device=device)
    module.launch('gpu_sizes', [pointer(sizes)], block=1)
    state_bytes, c, f = sizes.tolist()
    module.launch('gpu_tables', block=1)
    module.ready = torch.cuda.Event()
    module.ready.record()
    return module, state_bytes, c, f


class GpuSearchEnv(GpuEnv):
    """Search worlds stop at external-decision depth or true terminal.

    Set games for finite, paired-deal, both-seat root self-play instead. Each
    lane plays one game; finished lanes freeze. This bounds retained records.
    """
    _load_module = staticmethod(_search_module)

    def __init__(self, num_envs, *, max_decisions=512, games=None, eval_seed=0, **kwargs):
        if not isinstance(max_decisions, int) or max_decisions <= 0:
            raise ValueError('max_decisions must be positive')
        if games is not None and (games < num_envs or games % 2):
            raise ValueError('finite CUDA roots require even games >= num_envs')
        if not 0 <= eval_seed < 2**64:
            raise ValueError('eval_seed must fit uint64')
        self.search = games is None
        self.max_decisions, self.games, self.eval_seed = max_decisions, games, eval_seed
        self.track_projection = not self.search
        super().__init__(num_envs, **kwargs)
        self.world_status = torch.full((num_envs,), 2, dtype=torch.int32, device=self.device)
        self.searcher_seat = torch.full_like(self.world_status, -1)
        self.next_actor = torch.full_like(self.world_status, -1)
        self.result = torch.zeros_like(self.world_status)

    def _initialize(self):
        self._module.launch('gpu_search_init', self._args()+[
            pointer(self._pool) if self._pool is not None else ct.c_void_p(), ct.c_bool(self.search),
            ct.c_int(self.max_decisions), ct.c_longlong(-1 if self.games is None else self.games),
            ct.c_ulonglong(self.eval_seed)], n=self.num_envs)

    def _rows(self, rows):
        rows = torch.as_tensor(rows, device=self.device, dtype=torch.int32).contiguous()
        if rows.ndim != 1:
            raise ValueError('rows must be a vector')
        # Public API validates indices. Internal generated rows already satisfy this.
        if rows.numel() and (bool((rows < 0).any()) or bool((rows >= self.num_envs).any())):
            raise IndexError('row outside environment')
        return rows

    def determinize_support(self, rows):
        rows = self._rows(rows)
        out = torch.empty_like(rows)
        self._module.launch('gpu_search_support', [pointer(self._storage), pointer(rows),
                            ct.c_int(len(rows)), pointer(out)], n=len(rows))
        return out

    def validate_assignments(self, rows, assignments):
        rows = self._rows(rows)
        if assignments.shape != (len(rows), 297) or assignments.dtype != torch.int32 or assignments.device != self.device:
            raise ValueError('assignments must be device int32 [len(rows),297]')
        assignments = assignments.contiguous()
        out = torch.empty(len(rows), dtype=torch.bool, device=self.device)
        self._module.launch('gpu_search_unused', [pointer(self._storage), pointer(rows), pointer(assignments),
                            ct.c_int(len(rows)), pointer(out)], n=len(rows))
        return out

    def load_worlds(self, src, rows, assignments, seeds):
        if not self.search or not isinstance(src, GpuSearchEnv) or src is self or not src.track_projection:
            raise ValueError('load_worlds requires separate projection-tracked CUDA roots')
        if (src.history_len, src.canonical, src.device) != (self.history_len, self.canonical, self.device):
            raise ValueError('history_len, canonical or device mismatch')
        rows = src._rows(rows)
        if rows.shape != (self.num_envs,) or assignments.shape != (self.num_envs,297) or seeds.shape != (self.num_envs,):
            raise ValueError('world shapes mismatch')
        if assignments.dtype != torch.int32 or seeds.dtype != torch.uint64 or any(t.device != self.device for t in (assignments,seeds)):
            raise ValueError('device int32 assignments and uint64 seeds required')
        assignments, seeds = assignments.contiguous(), seeds.contiguous()
        self._module.launch('gpu_search_load', self._args()+[pointer(src._storage), pointer(src._pointers),
                            pointer(rows), pointer(assignments), pointer(seeds), pointer(self.world_status),
                            ct.c_bool(os.environ.get('GFRL_GPU_SEARCH_LEAN','1')=='1')], n=self.num_envs)
        self._observe()
        self._actors()
        return self.world_status

    def _actors(self):
        self._module.launch('gpu_search_actors', [pointer(self._storage), ct.c_int(self.num_envs),
                            pointer(self.next_actor), pointer(self.searcher_seat), pointer(self.result)], n=self.num_envs)

    def next_actor_seats(self, rows):
        self._actors()
        return self.next_actor[self._rows(rows).long()]

    def observe(self, rows, seats):
        rows = self._rows(rows)
        seats = torch.as_tensor(seats, device=self.device, dtype=torch.int32).expand(len(rows)).contiguous()
        if bool(((seats < 0) | (seats > 1)).any()):
            raise ValueError('seat must be 0 or 1')
        if self.search and bool((self.world_status[rows.long()] != 0).any()):
            raise ValueError('observe invalid world')
        if len(torch.unique(rows)) != len(rows):
            raise ValueError('leaf rows must be unique')
        return self._observe_rows(rows,seats)

    def _observe_rows(self,rows,seats):
        """Trusted unique device rows used by the search leaf loop."""
        from ..model import OBS_FIELDS
        # The rule kernel temporarily encodes a leaf using that lane's allocator.
        # One row per lane prevents concurrent access to its private arena.
        leaf = {k: torch.empty((len(rows),*getattr(self,k).shape[1:]),device=self.device,dtype=getattr(self,k).dtype) for k in OBS_FIELDS}
        pointers = self._pointers.clone()
        pointers[:6] = torch.tensor([leaf[k].data_ptr() for k in FIELDS[:6]], device=self.device)
        self._module.launch('gpu_search_observe', [pointer(self._storage), pointer(pointers), ct.c_int(len(rows)),
                            ct.c_int(self.history_len), pointer(self._heaps), ct.c_uint(self._heap_stride),
                            pointer(rows), pointer(seats)], n=len(rows))
        return {k:v[:len(rows)] for k,v in leaf.items()}


class GpuCandidateEnv:
    """Read-only Match references plus independent selection cursors.

    Candidate generation needs root prefixes only: the completed command is
    executed later by each actual candidate/world rollout, never by this cursor.
    """
    def __init__(self,base,indices,src,root_rows):
        self.base,self.src,self.root_rows=base,src,root_rows.long()
        self.device=base.device;self.num_envs=len(indices);self._module=base._module
        self.canonical=base.canonical;self._closed=False
        size=torch.empty(1,device=self.device,dtype=torch.int64)
        self._module.launch('gpu_candidate_sizes',[pointer(size)],block=1)
        self._storage=torch.empty(self.num_envs*int(size[0]),device=self.device,dtype=torch.uint8)
        self.arena_bytes=8192
        self._heaps=torch.empty(self.num_envs*self.arena_bytes,device=self.device,dtype=torch.uint8)
        from ..actor import PREFIX_FIELDS
        fields=(*PREFIX_FIELDS,'actor_seat','substep','episode_done')
        for k in fields:setattr(self,k,torch.zeros_like(getattr(base,k)[:1]).expand(self.num_envs,*getattr(base,k).shape[1:]).clone())
        pointers=[getattr(self,k).data_ptr() if hasattr(self,k) else 0 for k in FIELDS]+[0,0]
        self._pointers=torch.tensor(pointers,device=self.device,dtype=torch.int64)
        indices=indices.int().contiguous()
        self._module.launch('gpu_candidate_init',[pointer(self._storage),pointer(self._pointers),ct.c_int(self.num_envs),
            pointer(base._storage),pointer(indices),pointer(self._heaps),ct.c_uint(self.arena_bytes)],n=self.num_envs)

    def public(self,rows,fields):
        return {k:getattr(self.src,k)[self.root_rows[rows.long()]] for k in fields}

    def step(self,choices):
        self._module.launch('gpu_candidate_step',[pointer(self._storage),pointer(self._pointers),ct.c_int(self.num_envs),
            pointer(self.base._storage),pointer(choices),ct.c_bool(self.canonical),pointer(self._heaps),ct.c_uint(self.arena_bytes)],n=self.num_envs)

    def close(self):
        if not self._closed:
            self._module.launch('gpu_candidate_destroy',[pointer(self._storage),ct.c_int(self.num_envs),pointer(self._heaps),ct.c_uint(self.arena_bytes)],n=self.num_envs)
            torch.cuda.synchronize(self.device);self._closed=True
