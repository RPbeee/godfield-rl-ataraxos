"""The GPU contract is checked against the independent native CPU environment."""
import numpy as np
import pytest
import torch

from gfrl import VecEnv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_forced_512th_decision_truncation_and_reset(tmp_path):
    import os
    import struct
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    from gfrl.gpu import FIELDS,GpuEnv
    from gfrl.gpu.jit import Module
    from gfrl.gpu.source import ROOT,build_source
    executable=os.environ.get('GFRL_GPU_CPU_ORACLE')
    if executable is None:
        build=Path(tempfile.gettempdir())/'gfrl-gpu-cpu-oracle-build'
        cmake=Path(sys.executable).with_name('cmake')
        cmake=str(cmake) if cmake.is_file() else 'cmake'
        subprocess.run([cmake,'-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                        f'-DPython_EXECUTABLE={sys.executable}'],check=True)
        subprocess.run([cmake,'--build',str(build),'--target','gfrl_gpu_oracle','-j','4'],check=True)
        executable=str(build/'gfrl_gpu_oracle')
    witness=tmp_path/'truncation.bin'
    subprocess.run([executable,str(witness)],check=True,capture_output=True)
    raw=witness.read_bytes();frames=struct.unpack_from('<I',raw)[0];offset=4
    with GpuEnv(1,base_seed=42) as gpu:
        fixture=Module(build_source()+(ROOT/'tests/cpp/gpu_fixture.cuh').read_text())
        fixture.launch('gpu_tables',block=1)
        fixture.launch('gpu_test_counter',gpu._args(),n=1)
        gpu._observe()
        for tick in range(frames):
            choice=struct.unpack_from('<i',raw,offset)[0];offset+=4
            if tick:gpu.step(torch.tensor([choice],device=gpu.device,dtype=torch.int32))
            for name in FIELDS:
                actual=getattr(gpu,name).cpu().numpy()
                expected=np.frombuffer(raw,dtype=actual.dtype,count=actual.size,offset=offset).reshape(actual.shape)
                offset+=actual.nbytes
                np.testing.assert_array_equal(actual,expected,err_msg=f'truncation step={tick}, field={name}')
        expected_stats=struct.unpack_from('<11Q',raw,offset);offset+=88
        assert tuple(gpu.stats().values())==expected_stats
        assert offset==len(raw)
        assert bool(gpu.truncated[0]) and bool(gpu.episode_done[0])
        assert int(gpu.episode_id[0])==1


@pytest.mark.parametrize('pooled',[False,True])
@pytest.mark.parametrize('execution',['monolithic','split','compact'])
@pytest.mark.parametrize('seed,history_len,canonical', [(0, 0, True), (42, 64, True),
                                                       (2**64-1, 1, False)])
def test_exact_replay(seed, history_len, canonical, pooled, execution, monkeypatch):
    monkeypatch.setenv('GFRL_GPU_SPLIT_STEP',str(int(execution!='monolithic')))
    monkeypatch.setenv('GFRL_GPU_COMPACT_STEP',str(int(execution=='compact')))
    from gfrl.gpu import FIELDS, GpuEnv
    cpu = VecEnv(8, num_threads=2, base_seed=seed, history_len=history_len, canonical=canonical)
    rng = np.random.default_rng(seed)
    with GpuEnv(8, base_seed=seed, history_len=history_len, canonical=canonical, pooled=pooled) as gpu:
        from gfrl.gpu.storage import NumericStorage
        numeric=NumericStorage(gpu.device,history_len)
        for step in range(257):
            if step%17==0:
                torch.testing.assert_close(numeric.expand(numeric.pack(gpu.obs_num,gpu.obs_type),gpu.obs_type),gpu.obs_num.half(),atol=0,rtol=0)
            for name in FIELDS:
                np.testing.assert_array_equal(getattr(cpu,name), getattr(gpu,name).cpu().numpy(),
                                              err_msg=f'step={step}, field={name}')
            choices = np.array([rng.choice(np.flatnonzero(row)) for row in cpu.mask], dtype=np.int32)
            if step % 31 == 0:
                choices[np.asarray(cpu.substep) == 0] = -1
            cpu.step(choices)
            gpu.group_actions = step % 4 == 1
            selected = torch.from_numpy(choices).to(gpu.device)
            gpu.step(selected, _active=(selected != -1) if step % 2 else None)

        assert cpu.stats() == gpu.stats()


@pytest.mark.parametrize('short_attention',[False,True])
@pytest.mark.parametrize('tma',[False,True])
def test_packed_encoder_dynamic_counts_and_weight_refresh(short_attention,tma):
    from types import SimpleNamespace
    from gfrl.model import Model, ModelConfig, OBS_FIELDS
    from gfrl.gpu.packed import PackedEncoder
    from gfrl.runtime import encoder_fastpath_disabled
    cpu = VecEnv(8, num_threads=2, base_seed=72, history_len=8)
    rng = np.random.default_rng(72)
    model = Model(ModelConfig(d=128, depth=2, heads=4, ff=512)).cuda().eval()
    obs = {k:torch.tensor(getattr(cpu,k).copy(), device='cuda') for k in OBS_FIELDS}
    env = SimpleNamespace(**obs, device=torch.device('cuda'))
    packed = PackedEncoder(model, env)
    packed.short_attention = short_attention
    packed.tma = tma and torch.cuda.get_device_capability()[0]>=9
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16), encoder_fastpath_disabled():
        packed()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = packed()
        for tick in range(4):
            for _ in range(11):
                cpu.step(np.array([rng.choice(np.flatnonzero(m)) for m in cpu.mask],dtype=np.int32))
            for key in OBS_FIELDS:
                obs[key].copy_(torch.from_numpy(getattr(cpu,key).copy()))
            if tick == 2:
                model.encoder.layers[0].linear1.weight[:,0].add_(.1)
                packed.refresh()
            graph.replay()
            expected = model.encode(obs,pack=False)[:,:21]
            torch.testing.assert_close(actual,expected,atol=.05,rtol=.03)



def test_long_rollout_resets_exact():
    from gfrl.gpu import FIELDS, GpuEnv
    cpu = VecEnv(2, num_threads=1, base_seed=19)
    with GpuEnv(2,base_seed=19) as gpu:
        for tick in range(1600):
            for field in FIELDS:
                np.testing.assert_array_equal(getattr(cpu,field),getattr(gpu,field).cpu().numpy(),err_msg=f'{tick}: {field}')
            choice = []
            for stage,mask in zip(cpu.stage,cpu.mask):
                legal = np.flatnonzero(mask)
                if stage == 0:
                    choice.append(next((k for k in (2,3,9) if mask[k]),int(legal[0])))
                else:
                    choice.append(18 if stage==1 and mask[18] else int(legal[0]))
            actions=np.array(choice,dtype=np.int32)
            cpu.step(actions);gpu.step(torch.from_numpy(actions).cuda())
        assert cpu.stats()['episodes'] >= 2
        assert cpu.stats() == gpu.stats()


def test_sparse_heads_match_public_model():
    from types import SimpleNamespace
    from gfrl.actor import PREFIX_FIELDS
    from gfrl.model import Model, ModelConfig, OBS_FIELDS
    from gfrl.gpu.heads import SparseHeads
    cpu=VecEnv(16,num_threads=2,base_seed=314)
    rng=np.random.default_rng(314)
    for _ in range(61):cpu.step(np.array([rng.choice(np.flatnonzero(m)) for m in cpu.mask],dtype=np.int32))
    fields={k:torch.tensor(getattr(cpu,k).copy(),device='cuda') for k in (*OBS_FIELDS,*PREFIX_FIELDS)}
    env=SimpleNamespace(**fields,num_envs=16,device=torch.device('cuda'))
    model=Model(ModelConfig(d=128,depth=2,heads=4,ff=512)).cuda().eval()
    heads=SparseHeads(model,env)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        encoded=model.encode({k:fields[k] for k in OBS_FIELDS},pack=False)[:,:21].contiguous()
        expected,value=model.heads(encoded,**{k:fields[k] for k in PREFIX_FIELDS})
        actual=heads(encoded)
        torch.testing.assert_close(actual,expected,atol=.04,rtol=.02)
        torch.testing.assert_close(heads.values(encoded),value,atol=.02,rtol=.02)
        active=torch.arange(16,device='cuda')%3==0
        sparse=heads(encoded,active)
        torch.testing.assert_close(sparse[active],expected[active],atol=.04,rtol=.02)


def test_disjoint_stream_slices_preserve_global_seeds():
    from gfrl.gpu import GpuEnv,FIELDS
    cpu=VecEnv(8,num_threads=2,base_seed=17,history_len=8)
    rng=np.random.default_rng(17)
    with GpuEnv(8,base_seed=17,history_len=8) as gpu:
        views=[gpu._slice(0,3),gpu._slice(3,8)]
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        for tick in range(501):
            for field in FIELDS:
                np.testing.assert_array_equal(getattr(cpu,field),getattr(gpu,field).cpu().numpy(),err_msg=f'{tick}: {field}')
            actions=np.array([rng.choice(np.flatnonzero(m)) for m in cpu.mask],dtype=np.int32)
            cpu.step(actions)
            for view,stream,start,stop in zip(views,streams,(0,3),(3,8)):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):view.step(torch.tensor(actions[start:stop],device='cuda'))
            for stream in streams:torch.cuda.current_stream().wait_stream(stream)
        assert cpu.stats()==gpu.stats()


@pytest.mark.parametrize('pooled',[False,True])
def test_parallel_round_replay_and_graph(pooled):
    from gfrl.gpu import GpuEnv,FIELDS
    from gfrl.gpu.actor import ParallelGpuActor
    from gfrl.model import Model,ModelConfig
    cpu=VecEnv(8,num_threads=2,base_seed=27,history_len=8)
    with GpuEnv(8,base_seed=27,history_len=8,pooled=pooled) as gpu:
        actor=ParallelGpuActor(Model(ModelConfig(d=128,depth=2,heads=4,ff=512)).cuda(),gpu,streams=2)
        for _ in range(3):
            finished,data=actor.round(record=True)
            assert finished.all()
            for stage in data['stages']:
                choice=torch.where(stage['valid'],stage['choice'],-1).cpu().numpy().astype(np.int32)
                cpu.step(choice)
            for field in FIELDS:
                np.testing.assert_array_equal(getattr(cpu,field),getattr(gpu,field).cpu().numpy(),err_msg=field)
        actor.capture(record=True)
        before=gpu.stats()
        for _ in range(3):
            finished,data=actor.round(record=True)
            assert finished.all()
            assert data['obs_type'].shape==(8,63)
        after=gpu.stats()
        assert after['decisions']-before['decisions']-after['automatic_decisions']+before['automatic_decisions']==24


@pytest.mark.parametrize('history_len',[0,1,64,128])
def test_numeric_storage_exact_roundtrip(history_len):
    from gfrl.gpu.storage import NumericStorage
    from gfrl import _core
    schema=_core.obs_schema(history_len)
    storage=NumericStorage(torch.device('cuda'),history_len)
    n=32;t=schema['T'];f=schema['F']
    # Every permitted channel, including zero/negative/large values, is covered.
    typ=torch.zeros((n,t),dtype=torch.int8,device='cuda')
    typ[:,:3]=torch.tensor([1,2,2],device='cuda',dtype=torch.int8)
    typ[:,3:21]=3;typ[:,21]=4;typ[:,22]=5;typ[:,23:55]=6
    if history_len:typ[:,55:]=torch.arange(n*history_len,device='cuda').reshape(n,history_len)%12+7
    present=storage.inverse[typ.long()]>=0
    numeric=torch.randn((n,t,f),device='cuda')*17
    numeric.masked_fill_(~present,0)
    torch.testing.assert_close(storage.expand(storage.pack(numeric,typ),typ),numeric.half(),atol=0,rtol=0)


def test_collector_pending_carry_and_returned_batch_ownership():
    from gfrl.gpu.collector import GpuCollector
    from gfrl.model import Model,ModelConfig
    collector=GpuCollector(Model(ModelConfig(d=128,depth=2,heads=4,ff=512)).cuda(),num_envs=16,seed=194,streams=2)
    try:
        first=collector.collect(4)
        saved={key:value.clone() for key,value in first.columns.items()}
        saved_stages={key:value.clone() for key,value in first.stages.items()}
        pending=collector.columns.size
        second=collector.collect(4)
        assert second.timings['carried_decisions']==pending>0
        for key,value in saved.items():torch.testing.assert_close(first.columns[key],value,atol=0,rtol=0)
        for key,value in saved_stages.items():torch.testing.assert_close(first.stages[key],value,atol=0,rtol=0)
        assert (second.outcome!=2).all()
        assert (collector.columns.data['outcome'][:collector.columns.size]==2).all()
        assert second.obs_num.shape[1:]==(119,81)
        assert second.obs_num.dtype==torch.float16
        stages=second.stages
        assert stages['mask'].gather(1,stages['choice'].long()[:,None]).all()
        assert int(second.stage_count.sum())==len(stages['choice'])
    finally:collector.close()


def test_forward_chunk_compaction_handles_overlapping_sources():
    from gfrl.gpu.collector import _compact_pending
    source=torch.arange(50000*17,device='cuda',dtype=torch.float32).reshape(50000,17)
    indices=torch.arange(1,50000,2,device='cuda')
    expected=source.index_select(0,indices)
    _compact_pending(source,indices)
    torch.testing.assert_close(source[:len(indices)],expected,atol=0,rtol=0)


def test_numeric_expansion_crosses_int32_element_boundary():
    from gfrl.gpu.storage import NumericStorage
    if torch.cuda.mem_get_info()[0]<6*1024**3:pytest.skip('needs 6 GiB free for the int32-boundary regression')
    storage=NumericStorage(torch.device('cuda'),0)
    n=2**31//(storage.tokens*storage.features)+1
    packed=torch.empty((n,storage.width),device='cuda',dtype=torch.float16)
    types=torch.zeros((n,storage.tokens),device='cuda',dtype=torch.int8)
    expanded=storage.expand(packed,types)
    assert expanded.numel()>2**31
    assert not expanded[-1].count_nonzero()


def test_paged_column_growth_gather_and_overlap_compaction():
    from gfrl.gpu.buffers import PagedColumn
    source=torch.arange(50000*17,device='cuda',dtype=torch.float32).reshape(50000,17)
    column=PagedColumn(source,1024)
    column.append(source[:23456],0);column.append(source[23456:],23456)
    indices=torch.arange(1,50000,2,device='cuda')
    expected=source.index_select(0,indices)
    torch.testing.assert_close(column.gather(indices),expected,atol=0,rtol=0)
    column.compact(indices)
    torch.testing.assert_close(column.gather(torch.arange(len(indices),device='cuda')),expected,atol=0,rtol=0)
    column.append(source[:5000],len(indices))
    expected=torch.cat((expected,source[:5000]))
    torch.testing.assert_close(column.gather(torch.arange(len(expected),device='cuda')),expected,atol=0,rtol=0)
