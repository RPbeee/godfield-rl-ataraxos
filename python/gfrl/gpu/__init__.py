"""CUDA-resident training environment, imported only when explicitly requested."""
import ctypes as ct
import os
from functools import lru_cache
from pathlib import Path
import tempfile

os.environ.setdefault('TRITON_DEFAULT_FP_FUSION','0')
os.environ.setdefault('TRITON_CACHE_DIR',str(Path(tempfile.gettempdir())/'gfrl-gpu-triton'))

import torch

from .jit import Module, pointer, _check
from .source import build_source

FIELDS = ('obs_type', 'obs_cat', 'obs_cards', 'obs_aux_card', 'obs_num', 'obs_valid',
          'label_opp_hand', 'label_opp_hand_count', 'stage', 'actor_seat', 'chosen_kind',
          'chosen_cards', 'chosen_count', 'chosen_target', 'chosen_alloc_hp', 'outcome',
          'mask', 'substep', 'decision_done', 'episode_done', 'truncated', 'episode_id', 'decisions')


@lru_cache(maxsize=1)
def _module(device):
    torch.cuda.init()
    runtime = ct.CDLL('libcudart.so')
    stack_bytes = int(os.environ.get('GFRL_GPU_STACK_BYTES','16384'))
    _check(runtime.cudaDeviceSetLimit(0, ct.c_size_t(stack_bytes)), 'cudaDeviceSetLimit(stack)')
    module = Module(build_source(private_rules=os.environ.get('GFRL_GPU_PRIVATE_RULES','0')=='1'))
    sizes = torch.zeros(3, dtype=torch.int64, device=device)
    module.launch('gpu_sizes', [pointer(sizes)], block=1)
    state_bytes, c, f = sizes.tolist()
    module.launch('gpu_tables', block=1)
    module.ready = torch.cuda.Event()
    module.ready.record()
    return module, state_bytes, c, f


class GpuEnv:
    """Both external seats, default MatchConfig, canonical selection, GPU outputs.

    step() is asynchronous on the current PyTorch CUDA stream. Call close() while
    the context is alive to release private device containers. Evaluation bots,
    search worlds and public-root restrictions continue to use the CPU VecEnv.
    """
    def __init__(self, num_envs, base_seed=0, history_len=64, canonical=True, device='cuda', arena_bytes=None, pooled=False):
        if not isinstance(num_envs, int) or not 1 <= num_envs <= 1_000_000:
            raise ValueError('num_envs must be in 1..1000000')
        if not isinstance(history_len, int) or not 0 <= history_len <= 4096:
            raise ValueError('history_len must be in 0..4096')
        if not 0 <= base_seed < 2**64:
            raise ValueError('base_seed must fit uint64')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('GpuEnv requires CUDA')
        self.device = torch.device('cuda', torch.cuda.current_device() if self.device.index is None else self.device.index)
        self.num_envs, self.history_len = num_envs, history_len
        self.tokens = t = 55 + history_len
        self.base_seed, self.canonical = base_seed, canonical
        self.group_actions = num_envs >= 16384
        self.split_step = os.environ.get('GFRL_GPU_SPLIT_STEP','1') == '1'
        self.compact_step = os.environ.get('GFRL_GPU_COMPACT_STEP',str(int(num_envs>=16384))) == '1'
        self.arena_bytes = arena_bytes or max(512*1024, 128*1024+history_len*2048)
        if self.arena_bytes < 4096 or self.arena_bytes % 16:
            raise ValueError('arena_bytes must be >=4096 and divisible by 16')
        self._closed = True
        with torch.cuda.device(self.device):
            self._module, state_bytes, c, f = _module(str(self.device))
            self._state_bytes=state_bytes
            self._parent=None
            torch.cuda.current_stream().wait_event(self._module.ready)
            shapes = {'obs_type': ((t,), torch.int8), 'obs_cat': ((t,c), torch.int8),
                      'obs_cards': ((t,8), torch.int16), 'obs_aux_card': ((t,), torch.int16),
                      'obs_num': ((t,f), torch.float32), 'obs_valid': ((t,), torch.bool),
                      'label_opp_hand': ((297,), torch.uint8), 'label_opp_hand_count': ((), torch.uint8),
                      'chosen_cards': ((18,), torch.int8), 'outcome': ((2,), torch.int8),
                      'mask': ((21,), torch.bool), 'substep': ((), torch.int16)}
            shapes.update({k:((), torch.bool) for k in ('decision_done','episode_done','truncated')})
            shapes.update({k:((), torch.uint64) for k in ('episode_id','decisions')})
            for name in FIELDS:
                shape, dtype = shapes.get(name, ((), torch.int8))
                setattr(self, name, torch.zeros((num_envs,*shape), dtype=dtype, device=self.device))
            self._stats = torch.zeros((num_envs,12), dtype=torch.uint64, device=self.device)
            self._ready = torch.full((num_envs,),127,dtype=torch.int8,device=self.device)
            self._heap_stride=256 if pooled else self.arena_bytes
            self._heaps = torch.empty(num_envs*self._heap_stride, dtype=torch.uint8, device=self.device)
            self._pool = None
            if pooled:
                self._pool=torch.empty(num_envs*self.arena_bytes+16,dtype=torch.uint8,device=self.device)
                self._module.launch('gpu_pool_init',[pointer(self._pool),ct.c_ulonglong(num_envs*self.arena_bytes)],block=1)
            self._high_water = torch.empty(num_envs, dtype=torch.uint32, device=self.device)
            self._storage = torch.empty(num_envs*state_bytes, dtype=torch.uint8, device=self.device)
            self._pointers = torch.tensor([getattr(self,k).data_ptr() for k in FIELDS] + [self._stats.data_ptr(),self._ready.data_ptr()],
                                          dtype=torch.int64, device=self.device)
            self._module.launch('gpu_init', self._args()+[pointer(self._pool) if self._pool is not None else ct.c_void_p()], n=num_envs)
            self._observe()
            self._closed = False

    def _slice(self,start,stop):
        """Disjoint device view for parallel streams; parent owns its lifetime."""
        if not 0<=start<stop<=self.num_envs:
            raise ValueError('invalid environment slice')
        view=object.__new__(GpuEnv)
        for name in ('device','history_len','tokens','base_seed','canonical','arena_bytes',
                     '_heap_stride','_state_bytes','_module','_pool','split_step','compact_step'):
            setattr(view,name,getattr(self,name))
        view.num_envs=stop-start
        view.group_actions=view.num_envs>=16384
        view._closed=False;view._parent=self
        for name in (*FIELDS,'_stats','_high_water','_ready'):
            setattr(view,name,getattr(self,name)[start:stop])
        view._storage=self._storage[start*self._state_bytes:stop*self._state_bytes]
        view._heaps=self._heaps[start*self._heap_stride:stop*self._heap_stride]
        view._pointers=torch.tensor([getattr(view,k).data_ptr() for k in FIELDS]+[view._stats.data_ptr(),view._ready.data_ptr()],dtype=torch.int64,device=self.device)
        return view

    def _args(self):
        return [pointer(self._storage), pointer(self._pointers), ct.c_int(self.num_envs),
                ct.c_int(self.history_len), ct.c_ulonglong(self.base_seed), ct.c_bool(self.canonical),
                pointer(self._heaps), ct.c_uint(self._heap_stride)]

    def _observe(self):
        self._module.launch('gpu_observe', [pointer(self._storage),pointer(self._pointers),ct.c_int(self.num_envs),ct.c_int(self.history_len)],n=self.num_envs*128,block=128)

    def step(self, choices, *, _active=None):
        if self._closed or (self._parent is not None and self._parent._closed):
            raise RuntimeError('GpuEnv is closed')
        if (not isinstance(choices, torch.Tensor) or choices.device != self.device
                or choices.dtype != torch.int32 or choices.shape != (self.num_envs,)
                or not choices.is_contiguous()):
            raise ValueError('choices must be a contiguous CUDA int32 tensor with shape [num_envs] on the environment device')
        with torch.cuda.device(self.device):
            indices = count = None
            if _active is not None:
                if (_active.shape != (self.num_envs,) or _active.dtype != torch.bool
                        or _active.device != self.device or not _active.is_contiguous()):
                    raise ValueError('_active must be a CUDA bool vector on the environment device')
                if self.group_actions:
                    # Similar cursor stages/kinds share warps. Every row retains
                    # its own state and RNG; execution order has no semantics.
                    kind=torch.where(self.stage==0,choices,self.chosen_kind.to(torch.int32))
                    key=torch.where(_active,self.stage.to(torch.int32)*10+kind,127)
                    indices=torch.argsort(key).to(torch.int32)
                    count=_active.sum(dtype=torch.int32).reshape(1)
                else:
                    from .packed import _scatter_index
                    ranks = torch.cumsum(_active.to(torch.int32),0,dtype=torch.int32)
                    indices = torch.empty_like(ranks)
                    _scatter_index[((self.num_envs+255)//256,)](_active,ranks,indices,self.num_envs,256)
                    count = ranks[-1:]
            args = self._args() + [pointer(choices), pointer(indices) if indices is not None else ct.c_void_p(),
                                    pointer(count) if count is not None else ct.c_void_p()]
            if self.split_step:
                self._module.launch('gpu_choose', args, n=self.num_envs)
                if self.compact_step:
                    ready_indices=torch.argsort(self._ready).to(torch.int32)
                    ready_count=(self._ready!=127).sum(dtype=torch.int32).reshape(1)
                    advance_args=self._args()+[pointer(choices),pointer(ready_indices),pointer(ready_count)]
                else:
                    advance_args=args
                self._module.launch('gpu_advance', advance_args, n=self.num_envs)
                self._module.launch('gpu_clear_unencoded', [pointer(self._storage),ct.c_int(self.num_envs)],n=self.num_envs*128,block=128)
                self._module.launch('gpu_encode_view', [pointer(self._storage),pointer(self._pointers),ct.c_int(self.num_envs),ct.c_int(self.history_len),pointer(self._heaps),ct.c_uint(self._heap_stride)],n=self.num_envs)
            else:
                self._module.launch('gpu_clear_tokens', [pointer(self._storage),ct.c_int(self.num_envs),pointer(indices) if indices is not None else ct.c_void_p(),pointer(count) if count is not None else ct.c_void_p(),pointer(choices)],n=self.num_envs*128,block=128)
                self._module.launch('gpu_step', args, n=self.num_envs)
            self._observe()
            choices.record_stream(torch.cuda.current_stream())

    def obs_schema(self):
        from .. import _core
        return _core.obs_schema(self.history_len)

    def stats(self):
        """Explicit host snapshot; never called by the stepping hot path."""
        if self._closed or (self._parent is not None and self._parent._closed):
            raise RuntimeError('GpuEnv is closed')
        with torch.cuda.device(self.device):
            self._module.launch('gpu_stats', [pointer(self._storage), pointer(self._pointers),
                                ct.c_int(self.num_envs), pointer(self._high_water),
                                pointer(self._heaps), ct.c_uint(self._heap_stride)], n=self.num_envs)
            values = self._stats.to(torch.int64).sum(0).cpu().tolist()
        names = ('episodes','decisions','policy_stages','automatic_decisions','truncations',
                 'confusion_decisions','grant_decisions','history_dropped','card_overflow',
                 'used_overflow','token_overflow')
        values[8] += values[11]
        return dict(zip(names, values))

    def close(self):
        if self._parent is not None:
            self._closed=True
            return
        if not self._closed:
            with torch.cuda.device(self.device):
                self._module.launch('gpu_destroy', [pointer(self._storage), ct.c_int(self.num_envs), pointer(self._heaps), ct.c_uint(self._heap_stride)], n=self.num_envs)
                # Storage cannot be recycled before private destructors finish.
                torch.cuda.current_stream().synchronize()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
