"""Execute the generated CUDA translation unit as serial native C++ on a Mac.

This checks generated rules/storage logic, not CUDA compilation, scheduling,
allocator concurrency, or floating point behavior on an actual GPU.
"""
import ctypes as ct
import hashlib
from pathlib import Path
import subprocess
import tempfile

import torch

from gfrl import _core
from gfrl.gpu import FIELDS
from gfrl.gpu.jit import pointer
from gfrl.gpu.search_env import GpuSearchEnv
from gfrl.gpu.source import ROOT, build_source


class NativeModule:
    def __init__(self):
        source = '#define GFRL_GPU_HOST_TEST\n'+build_source(search=True)
        source += (ROOT/'tests/cpp/gpu_search_fixture.cuh').read_text()
        source += '\nextern "C" void gpu_test_index(int i){blockIdx.x=i;blockDim.x=1;threadIdx.x=0;}\n'
        key = hashlib.sha256(source.encode()).hexdigest()[:16]
        directory = Path(tempfile.gettempdir())/'gfrl-search-native'
        directory.mkdir(exist_ok=True)
        cpp, binary = directory/(key+'.cpp'), directory/(key+'.so')
        if not binary.exists():
            cpp.write_text(source)
            proc=subprocess.run(['c++','-std=c++20','-O1','-ffp-contract=off','-shared','-fPIC',str(cpp),'-o',str(binary)],capture_output=True,text=True)
            if proc.returncode:raise RuntimeError(proc.stderr)
        self.lib=ct.CDLL(str(binary))
        self.launch('gpu_tables',block=1)
        sizes=torch.empty(3,dtype=torch.int64)
        self.launch('gpu_sizes',[pointer(sizes)],block=1)
        self.sizes=sizes.tolist()

    def launch(self,name,args=(),*,n=1,block=32):
        count=(n+block-1)//block if name in ('gpu_observe','gpu_clear_tokens','gpu_clear_unencoded') else n
        for i in range(count):
            self.lib.gpu_test_index(ct.c_int(i))
            getattr(self.lib,name)(*args)


class NativeEnv(GpuSearchEnv):
    module=None
    def __init__(self,num_envs,*,history_len=64,max_decisions=512,games=None,eval_seed=0,
                 canonical=True,base_seed=0,device='cpu',pooled=False):
        assert torch.device(device).type=='cpu'
        if NativeEnv.module is None:NativeEnv.module=NativeModule()
        self._module=NativeEnv.module
        self.device=torch.device('cpu');self.num_envs=num_envs;self.history_len=history_len
        self.tokens=55+history_len;self.max_decisions=max_decisions;self.search=games is None
        self.games=games;self.eval_seed=eval_seed;self.base_seed=base_seed;self.canonical=canonical
        self.track_projection=not self.search;self._closed=False;self._parent=None
        self.arena_bytes=self._heap_stride=256;self._pool=None
        self._state_bytes,c,f=self._module.sizes
        self._storage=torch.empty(num_envs*self._state_bytes,dtype=torch.uint8)
        self._heaps=torch.empty(num_envs*self._heap_stride,dtype=torch.uint8)
        t=self.tokens
        shapes={'obs_type':((t,),torch.int8),'obs_cat':((t,c),torch.int8),'obs_cards':((t,8),torch.int16),
                'obs_aux_card':((t,),torch.int16),'obs_num':((t,f),torch.float32),'obs_valid':((t,),torch.bool),
                'label_opp_hand':((297,),torch.uint8),'label_opp_hand_count':((),torch.uint8),
                'label_opp_unused_hand':((297,),torch.uint8),'label_opp_unused_hand_count':((),torch.uint8),
                'chosen_cards':((18,),torch.int8),'outcome':((2,),torch.int8),'mask':((21,),torch.bool),
                'substep':((),torch.int16)}
        shapes.update({k:((),torch.bool) for k in ('decision_done','episode_done','truncated')})
        shapes.update({k:((),torch.uint64) for k in ('episode_id','decisions')})
        for k in FIELDS:
            shape,dtype=shapes.get(k,((),torch.int8));setattr(self,k,torch.zeros((num_envs,*shape),dtype=dtype))
        self._stats=torch.zeros((num_envs,12),dtype=torch.uint64)
        self._ready=torch.zeros(num_envs,dtype=torch.int8)
        self._pointers=torch.tensor([getattr(self,k).data_ptr() for k in (*FIELDS,'_stats','_ready')])
        self.world_status=torch.full((num_envs,),2,dtype=torch.int32)
        self.searcher_seat=torch.full_like(self.world_status,-1);self.next_actor=torch.full_like(self.world_status,-1)
        self.result=torch.zeros_like(self.world_status)
        self._initialize();self._observe()

    def step(self,choices,**kwargs):
        choices=torch.as_tensor(choices,dtype=torch.int32).contiguous()
        self._module.launch('gpu_clear_tokens',[pointer(self._storage),ct.c_int(self.num_envs),ct.c_void_p(),ct.c_void_p(),pointer(choices)],n=self.num_envs,block=1)
        self._module.launch('gpu_step',self._args()+[pointer(choices),ct.c_void_p(),ct.c_void_p()],n=self.num_envs)
        self._observe()

    def fixture(self,kind=0,model=6,fog=False,seed=19):
        self._module.launch('gpu_search_fixture',self._args()+[ct.c_int(kind),ct.c_int(model),ct.c_bool(fog),ct.c_ulonglong(seed)],n=1)
        self._observe()

    def close(self):
        if not self._closed:
            self._module.launch('gpu_destroy',[pointer(self._storage),ct.c_int(self.num_envs),pointer(self._heaps),ct.c_uint(self._heap_stride)],n=self.num_envs)
            self._closed=True
