"""Independent CPU oracle vs generated search rules, on native host and CUDA.

Native execution is deliberately reported separately: it does not validate
NVRTC, GPU allocation/concurrency or H100 inference/throughput.
"""
import ctypes as ct
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest
import torch

from gfrl import VecEnv, SearchEnv, _core
from gfrl.belief import Belief, BeliefConfig
from gfrl.belief_metrics import Baseline, gift_probabilities
from gfrl.distill import pack_records, save_shard, load_shard, prepare_batch, objective, outcome_target
from gfrl.evaluation import _policy_rng
from gfrl.gpu import FIELDS
from gfrl.gpu.jit import Module, pointer
from gfrl.gpu.search import GpuSearchAgent, retain_candidates, distribution
from gfrl.gpu.search_env import GpuSearchEnv
from gfrl.gpu.source import ROOT, build_source
from gfrl.model import OBS_FIELDS, Model, ModelConfig
from gfrl.search import SearchAgent, SearchConfig, search_distribution
from helpers.gpu_search_native import NativeEnv


@pytest.mark.parametrize('config', [SearchConfig(rollouts_per_world=4), SearchConfig(rollout_crn=True)])
def test_device_search_rejects_unimplemented_rollout_crn(config):
    with pytest.raises(ValueError, match='rollout CRN requires SearchAgent/SearchEnv'):
        GpuSearchAgent(None, None, 'cpu', config=config)


@lru_cache(None)
def cuda_fixture_module(device):
    module = Module(build_source(search=True)+(ROOT/'tests/cpp/gpu_search_fixture.cuh').read_text())
    module.launch('gpu_tables',block=1)
    return module


class CudaFixtureEnv(GpuSearchEnv):
    def fixture(self,kind=0,model=6,fog=False,seed=19):
        module=cuda_fixture_module(str(self.device))
        module.launch('gpu_search_fixture',self._args()+[ct.c_int(kind),ct.c_int(model),ct.c_bool(fog),ct.c_ulonglong(seed)],n=1)
        self._observe()


@pytest.fixture(params=['native','cuda'])
def backend(request):
    if request.param=='cuda':
        if not torch.cuda.is_available():pytest.skip('CUDA hardware required; native is a separate check')
        return CudaFixtureEnv,'cuda'
    return NativeEnv,'cpu'


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def equal(cpu,gpu,fields=FIELDS,context=''):
    for k in fields:
        np.testing.assert_array_equal(getattr(cpu,k),getattr(gpu,k).cpu().numpy(),err_msg=f'{context}: {k}')


def choices(env,rng):
    return np.array([rng.choice(np.flatnonzero(m)) if s>=0 else -1 for m,s in zip(env.mask,env.stage)],np.int32)


def load_and_roll(cpu,gpu,factory,device,rng,depth=4,worlds=16):
    rows=np.flatnonzero(cpu.determinize_support(np.arange(cpu.num_envs,dtype=np.int32))==0).astype(np.int32)
    if not len(rows):return 0,0
    rr=np.resize(rows,worlds);assignments=cpu.label_opp_hand[rr].astype(np.int32)
    seeds=rng.integers(np.iinfo(np.uint64).max,size=worlds,dtype=np.uint64)
    # Deliberately inject invalid assignments, then reload valid ones after comparison.
    for invalid in (True,False):
        aa=assignments.copy()
        if invalid:aa[0,0]=1;aa[1,1]=-1;aa[2,1]=19
        with factory(worlds,max_decisions=depth,device=device,history_len=gpu.history_len) as dst:
            ref=SearchEnv(worlds,1,history_len=gpu.history_len,max_decisions=depth)
            expected=ref.load_worlds(cpu,rr,aa,seeds)
            actual=dst.load_worlds(gpu,torch.tensor(rr,device=device),torch.tensor(aa,device=device),torch.tensor(seeds,device=device))
            np.testing.assert_array_equal(expected,actual.cpu().numpy())
            if invalid:
                equal(ref,dst,(*FIELDS,'world_status','searcher_seat'),'invalid loads')
                continue
            for tick in range(3000):
                dst._actors()
                equal(ref,dst,(*FIELDS,'world_status','searcher_seat','result'),f'rollout {tick}')
                if ref.episode_done.all():break
                action=choices(ref,rng);ref.step(action);dst.step(torch.tensor(action,device=device))
            else:pytest.fail('rollouts did not stop')
            for seat in (0,1):
                rr2=np.arange(worlds,dtype=np.int32)
                a=ref.observe(rr2,seat);b=dst.observe(torch.tensor(rr2,device=device),seat)
                for k in OBS_FIELDS:np.testing.assert_array_equal(a[k],b[k].cpu().numpy(),err_msg='leaf '+k)
            terminals=int((ref.next_actor_seats(np.arange(worlds,dtype=np.int32))<0).sum())
    return len(rows),terminals


@pytest.mark.parametrize('history_len',[0,1,64])
def test_many_roots_worlds_histories_legal_terminal_and_depth(backend,history_len):
    factory,device=backend;rng=np.random.default_rng(481)
    cpu=VecEnv(8,1,eval_pairs=4,opponent='external',eval_seed=42,history_len=history_len,track_projection=True)
    searched=terminals=0
    with factory(8,games=8,eval_seed=42,device=device,history_len=history_len) as gpu:
        for tick in range(160):
            equal(cpu,gpu,context=f'root {tick}')
            a=cpu.determinize_support(np.arange(8,dtype=np.int32));b=gpu.determinize_support(torch.arange(8,device=device,dtype=torch.int32))
            np.testing.assert_array_equal(a,b.cpu().numpy())
            if tick%9==0:
                nr,nt=load_and_roll(cpu,gpu,factory,device,rng,depth=512 if tick%18==0 else 4)
                searched+=nr;terminals+=nt
            action=choices(cpu,rng);cpu.step(action);gpu.step(torch.tensor(action,device=device))
    assert searched>=20 and terminals>=16


@pytest.mark.parametrize('kind,model,fog',[(0,6,False),(0,8,False),(0,7,False),
    (1,6,False),(1,8,False),(1,7,False),(2,0,False),(2,1,False),
    (3,6,False),(3,6,True),(3,202,False),(3,202,True),(3,43,False)])
def test_dream_response_chance_and_danger_roots(backend,kind,model,fog):
    factory,device=backend
    if kind==3 and model==43:
        model=next(k for k,c in _core.play_catalog().items() if c['hit_rate'] and c['category']=='weapons')
    cpu=VecEnv(2,1,eval_pairs=1,opponent='external',eval_seed=19,track_projection=True)
    with factory(2,games=2,eval_seed=19,device=device) as gpu:
        if kind<2:_core._test_dream_root(cpu,bool(kind),model)
        elif kind==2:_core._test_danger_root(cpu,bool(model))
        else:_core._test_response_root(cpu,model,fog,19)
        gpu.fixture(kind,model,fog,19)
        equal(cpu,gpu,context='fixture root')
        load_and_roll(cpu,gpu,factory,device,np.random.default_rng(93),depth=6)


def small_model():
    torch.manual_seed(15)
    return Model(ModelConfig(d=16,heads=2,depth=1,ff=32)).eval()


def baseline():
    return Baseline(torch.full((19,),1/19,dtype=torch.float64),gift_probabilities())


def compare_reports(a,b,atol=2e-6):
    assert len(a.last_search)==len(b.last_search)
    for x,y in zip(a.last_search,b.last_search):
        assert (x['row'],x['selected'])==(y['row'],y['selected'])
        np.testing.assert_allclose(x['probabilities'],y['probabilities'],atol=atol,rtol=2e-5)
        assert len(x['candidates'])==len(y['candidates'])
        for c,d in zip(x['candidates'],y['candidates']):
            assert c['path']==d['path'] and c['valid_worlds']==d['valid_worlds']
            for k in ('log_pi','log_rho','q','world_values'):
                if k in c:
                    if c[k] is None:assert d[k] is None
                    else:np.testing.assert_allclose(c[k],d[k],atol=atol,rtol=2e-5,err_msg=k)


@pytest.mark.parametrize('ar,skip,cap',[(False,False,0),(False,True,3),(True,False,0),(True,True,3)])
def test_search_candidates_q_pi_and_record_format(backend,ar,skip,cap,tmp_path):
    factory,device=backend
    model=small_model().to(device)
    belief=Belief(model.config,BeliefConfig(depth=1,heads=2)).eval().to(device) if ar else baseline()
    config=SearchConfig(worlds=3,candidates=4,candidate_samples=8,depth=3,search_response=True,
        skip_forced_inference=skip,skip_single_candidate=skip,max_inference_batch=cap,max_attempts=3)
    cpu=VecEnv(4,1,eval_pairs=2,opponent='external',eval_seed=12,track_projection=True)
    a=SearchAgent(model,belief,device,config=config,seed=55,record=True)
    b=GpuSearchAgent(model,belief,device,config=config,seed=55,record=True,env_factory=factory)
    try:
        with factory(4,games=4,eval_seed=12,device=device) as gpu:
            for ref,src in ((a,cpu),(b,gpu)):
                with _policy_rng(42,(ref,)):ref.search(src,np.arange(4,dtype=np.int32),report_values=True)
            compare_reports(a,b)
            assert a.last_candidates.keys()==b.last_candidates.keys()
            for ref in (a,b):
                for r in ref.records:r['outcome']=outcome_target(1,r['seat'])
            da,db=pack_records(a.records,{}),pack_records(b.records,{})
            for k in ('obs','prefix'):
                for field in da[k]:torch.testing.assert_close(da[k][field],db[k][field],atol=0,rtol=0)
            for k in ('candidate_offsets','stage_offsets','choice','valid_worlds','game','seat','phase'):
                torch.testing.assert_close(da[k],db[k],atol=0,rtol=0)
            save_shard(tmp_path/'gpu.pt.gz',db);loaded=load_shard(tmp_path/'gpu.pt.gz')
            batch=prepare_batch(loaded,np.arange(len(loaded['game'])),device)
            loss,_=objective(model,model,batch);assert torch.isfinite(loss)
    finally:b.close()


@pytest.mark.parametrize('response',[False,True])
@pytest.mark.parametrize('dtype',['fp32','bf16'])
def test_gpu_search_does_not_read_real_hidden_hand(backend,response,dtype):
    factory,device=backend;results=[]
    if dtype=='bf16' and device!='cuda':pytest.skip('production bf16 requires CUDA')
    if dtype=='bf16':
        import os
        from gfrl.evaluation import load_policy_model
        path=os.environ.get('GFRL_GPU_SEARCH_CHECKPOINT')
        torch.manual_seed(17)
        model=load_policy_model(path,device) if path else Model(ModelConfig(d=128,heads=4,depth=1,ff=128)).to(device).eval()
    else:model=small_model()
    config=SearchConfig(worlds=3,candidates=4,candidate_samples=8,depth=3,search_response=True,max_attempts=3)
    for changed in (False,True):
        with factory(2,games=2,eval_seed=12,device=device) as env:
            if response:env.fixture(3,6,False,19)
            if changed:env.fixture(4,6,False,1234)
            agent=GpuSearchAgent(model,baseline(),device,config=config,seed=55,env_factory=factory,dtype=dtype)
            try:
                with _policy_rng(42,(agent,)):agent.search(env,[0],report_values=True)
                results.append(agent.last_search)
            finally:agent.close()
    assert results[0]==results[1]


def test_device_candidate_dedup_last_wins_and_distribution():
    paths=torch.tensor([[[1,2,-1],[0,-1,-1],[1,2,-1],[1,-1,-1],[9,-1,-1]]],dtype=torch.int32)
    lp=torch.tensor([[2.,1.,3.,3.,100.]],dtype=torch.float64)
    lr=torch.arange(5,dtype=torch.float64)[None];valid=torch.tensor([[True,True,True,True,False]])
    p,a,b,v,idx=retain_candidates(paths,lp,lr,valid,4)
    assert idx[0,:3].tolist()==[3,2,1]
    assert v.tolist()==[[True,True,True,False]]
    q=torch.tensor([[.1,.2,.3,0.]])
    for alpha,beta in ((.002,.1),(0.,.1),(.1,0.)):
        actual=distribution(q,b,a,v,alpha,beta)
        expected=search_distribution(q[0,:3],b[0,:3],a[0,:3],alpha,beta)
        np.testing.assert_allclose(actual[0,:3],expected,atol=1e-14,rtol=1e-14)
        assert actual[0,3]==0


@pytest.mark.parametrize("rounds",[False,True])
def test_both_seat_selfplay_actions_response_records_and_outcomes(backend,rounds):
    factory,device=backend
    config=SearchConfig(worlds=2,candidates=3,candidate_samples=6,depth=2,search_response=True,
        skip_forced_inference=True,skip_single_candidate=True,max_attempts=3)
    traces=[];agents=[]
    cpu=VecEnv(4,1,eval_pairs=2,opponent='external',eval_seed=12,track_projection=True)
    with factory(4,games=4,eval_seed=12,device=device) as gpu:
        a=SearchAgent(small_model(),baseline(),device,config=config,seed=55,record=True)
        b=GpuSearchAgent(small_model(),baseline(),device,config=config,seed=55,record=True,env_factory=factory)
        try:
            for agent,env in ((a,cpu),(b,gpu)):
                trace=[]
                remaining=None
                with _policy_rng(42,(agent,)):
                    for tick in range(1500):
                        if rounds:
                            if remaining is None or not bool(remaining.any()):remaining=(env.stage>=0).copy() if env is cpu else (env.stage>=0)
                            active=np.flatnonzero(remaining) if env is cpu else torch.nonzero(remaining,as_tuple=True)[0]
                        else:active=np.flatnonzero(env.stage>=0) if env is cpu else torch.nonzero(env.stage>=0,as_tuple=True)[0]
                        if not len(active):break
                        action=agent(env,active)
                        if rounds:
                            if env is cpu:action[~remaining]=-1
                            else:action=torch.where(remaining,action,-1)
                        env.step(action)
                        if rounds:remaining &= ~env.decision_done & (env.stage>=0)
                        trace.append((action.copy() if env is cpu else action.cpu().numpy(),
                            {k:getattr(env,k).copy() if env is cpu else getattr(env,k).cpu().numpy().copy()
                             for k in (*OBS_FIELDS,'mask','stage','actor_seat','episode_done','outcome')}))
                    else:pytest.fail('selfplay did not terminate')
                traces.append(trace);agents.append(agent)
            assert len(traces[0])==len(traces[1])
            for tick,(x,y) in enumerate(zip(*traces)):
                # Idle choices are ignored by both engines and need not match.
                live=(traces[0][tick-1][1]['stage']>=0) if tick else np.ones(4,bool)
                np.testing.assert_array_equal(x[0][live],y[0][live],err_msg=f'actions {tick}')
                for k in x[1]:np.testing.assert_array_equal(x[1][k],y[1][k],err_msg=f'{tick}: {k}')
            assert a.counts['searched_response']>0
            assert b.counts['searched_response']==a.counts['searched_response']
            assert b.counts['searched_main']==a.counts['searched_main']
            assert b.counts['real_mismatches']==0
            assert len(a.records)==len(b.records)
            for x,y in zip(a.records,b.records):
                for k in ('game','seat','phase','decision'):assert x[k]==y[k]
                np.testing.assert_allclose(x['probabilities'],y['probabilities'],atol=2e-6,rtol=2e-5)
                assert [c['path'] for c in x['candidates']]==[c['path'] for c in y['candidates']]
        finally:b.close()


def test_refused_hidden_information_roots_are_not_sampled(backend):
    factory,device=backend
    with factory(2,games=2,eval_seed=19,device=device) as env:
        env.fixture(5,6,True,19)
        roots=torch.tensor([0],dtype=torch.int32,device=device)
        assert env.determinize_support(roots).tolist()==[8]  # viewer Fog hides enemy status
        class NeverSample:
            def sample(self,*args):raise AssertionError('belief used for a refused root')
        agent=GpuSearchAgent(small_model(),NeverSample(),device,env_factory=factory)
        try:
            assert agent.search(env,roots)=={}
            with factory(1,max_decisions=4,device=device) as dst:
                assignments=torch.zeros((1,297),dtype=torch.int32,device=device)
                seeds=torch.zeros(1,dtype=torch.uint64,device=device)
                assert dst.load_worlds(env,roots,assignments,seeds).tolist()==[2]
                assert int(dst.stage[0])==-1 and bool(dst.episode_done[0])
                for k in OBS_FIELDS:assert not bool(getattr(dst,k).any())
        finally:agent.close()


def test_reload_clears_old_world_outputs_and_reuses_history_ring(backend):
    factory,device=backend;rng=np.random.default_rng(7)
    cpu=VecEnv(2,1,eval_pairs=1,opponent='external',eval_seed=19,track_projection=True)
    with factory(2,games=2,eval_seed=19,device=device) as gpu,factory(2,max_decisions=2,device=device) as dst:
        ref=SearchEnv(2,1,max_decisions=2)
        for cycle in range(8):
            rr=np.array([0,1],np.int32);assignment=cpu.label_opp_hand.astype(np.int32).copy()
            if cycle%3==0:assignment[0,0]=1
            seeds=rng.integers(2**64-1,size=2,dtype=np.uint64)
            status=ref.load_worlds(cpu,rr,assignment,seeds)
            actual=dst.load_worlds(gpu,torch.tensor(rr,device=device),torch.tensor(assignment,device=device),torch.tensor(seeds,device=device))
            np.testing.assert_array_equal(status,actual.cpu().numpy())
            while True:
                dst._actors();equal(ref,dst,(*FIELDS,'world_status','searcher_seat','result'))
                if ref.episode_done.all():break
                action=choices(ref,rng);ref.step(action);dst.step(torch.tensor(action,device=device))


@pytest.mark.parametrize('response',[False,True])
@pytest.mark.parametrize('tf32',[False,True])
def test_model_m_fixed_world_search_cuda(response,tf32):
    """Real production weights, explicit common worlds, distill-1 search budget."""
    import os
    from gfrl.evaluation import load_policy_model
    from gfrl.search import load_search_belief
    checkpoint=os.environ.get('GFRL_GPU_SEARCH_CHECKPOINT')
    belief_path=os.environ.get('GFRL_GPU_SEARCH_BELIEF')
    if not torch.cuda.is_available() or not checkpoint or not belief_path:
        pytest.skip('CUDA and production checkpoints required')
    previous_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=tf32
    model=load_policy_model(checkpoint,'cuda')
    belief=load_search_belief(belief_path,device='cuda')
    config=SearchConfig(worlds=16,candidates=16,candidate_samples=64,depth=3,beta=.1,
                        search_response=True,skip_forced_inference=True,skip_single_candidate=True)
    cpu=VecEnv(4,1,eval_pairs=2,opponent='external',eval_seed=12,track_projection=True)
    with CudaFixtureEnv(4,games=4,eval_seed=12,device='cuda') as gpu:
        if response:
            _core._test_response_root(cpu,6,False,19);gpu.fixture(3,6,False,19)
        roots=np.array([0],np.int32) if response else np.arange(4,dtype=np.int32)
        a=SearchAgent(model,belief,'cuda',config=config,seed=55)
        b=GpuSearchAgent(model,belief,'cuda',config=config,seed=55,env_factory=CudaFixtureEnv)
        try:
            with _policy_rng(81,(a,)):rr,assignments,seeds=a._sample_worlds(cpu,roots)
            a._sample_worlds=lambda src,rs:(rr,assignments,seeds)
            # Root selection has its independent, identical NumPy stream.
            a.rng=np.random.default_rng(92);b.rng=np.random.default_rng(92)
            fixed=(torch.tensor(rr,device='cuda').long(),torch.tensor(assignments,device='cuda'),torch.tensor(seeds,device='cuda'))
            b._sample_worlds=lambda src,rs:fixed
            for agent,env in ((a,cpu),(b,gpu)):
                with _policy_rng(42,(agent,)):agent.search(env,roots,report_values=True)
            compare_reports(a,b,atol=3e-5)
            assert len(a.last_search)==len(roots)
        finally:
            b.close()
            torch.backends.cuda.matmul.allow_tf32=previous_tf32


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_incremental_belief_decode(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    import os
    from gfrl.gpu.search_inference import IncrementalBelief
    from gfrl.search import load_search_belief
    path=os.environ.get('GFRL_GPU_SEARCH_BELIEF')
    belief=(load_search_belief(path,device=device) if path else
            Belief(ModelConfig(d=32,heads=4,depth=2,ff=64),BeliefConfig(depth=2,heads=4)).eval().to(device))
    env=VecEnv(8,1,42)
    obs={k:torch.tensor(getattr(env,k),device=device) for k in OBS_FIELDS}
    with torch.inference_mode():
        _,memory,padding=belief.memory(obs)
        counts=torch.tensor([0,1,3,7,9,12,17,18],device=device)
        decoder=IncrementalBelief(belief,memory,padding,counts)
        inputs=torch.zeros((8,1),device=device,dtype=torch.long)
        for t in range(18):
            expected=belief.decode(memory,padding,inputs,counts)[:,-1]
            actual=decoder.step(inputs[:,-1])
            torch.testing.assert_close(actual,expected,atol=3e-5,rtol=2e-5)
            card=expected.argmax(-1)+1
            inputs=torch.cat((inputs,card[:,None]),1)


def test_fp32_packed_encoder_cuda():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import os
    from gfrl.gpu.search_inference import PackedSearchEncoder
    from gfrl.evaluation import load_policy_model
    path=os.environ.get('GFRL_GPU_SEARCH_CHECKPOINT')
    model=load_policy_model(path,'cuda') if path else Model(ModelConfig(d=128,heads=4,depth=2,ff=512)).cuda().eval()
    env=VecEnv(8,1,42);rng=np.random.default_rng(133)
    packed=PackedSearchEncoder(model)
    with torch.inference_mode():
        for tick in range(4):
            if tick:
                for _ in range(27):env.step(choices(env,rng))
            obs={k:torch.tensor(getattr(env,k),device='cuda') for k in OBS_FIELDS}
            expected=model.encode(obs)[:,:21]
            actual=packed.encode(obs)
            torch.testing.assert_close(actual,expected,atol=3e-5,rtol=2e-5)


def test_bf16_full_belief_encoder_cuda():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import os
    from gfrl.gpu.search_inference import PackedSearchEncoder
    from gfrl.search import load_search_belief
    path=os.environ.get('GFRL_GPU_SEARCH_BELIEF')
    belief=load_search_belief(path,device='cuda') if path else Belief(ModelConfig(d=128,heads=4,depth=2,ff=512),BeliefConfig(heads=4)).cuda().eval()
    env=VecEnv(8,1,42);rng=np.random.default_rng(931)
    packed=PackedSearchEncoder(belief,torch.bfloat16,full=True)
    with torch.inference_mode():
        for tick in range(3):
            if tick:
                for _ in range(39):env.step(choices(env,rng))
            obs={k:torch.tensor(getattr(env,k),device='cuda') for k in OBS_FIELDS}
            with torch.autocast('cuda',dtype=torch.bfloat16):expected=belief.encode(obs,pack=False)
            actual=packed.encode(obs)
            torch.testing.assert_close(actual,expected,atol=.05,rtol=.03)


def test_finite_refill_replay_and_episode_ids(backend):
    factory,device=backend;rng=np.random.default_rng(23)
    cpu=VecEnv(4,1,eval_pairs=8,opponent='external',eval_seed=12,track_projection=True)
    with factory(4,games=16,eval_seed=12,device=device) as gpu:
        for tick in range(3000):
            equal(cpu,gpu,context=f'refill {tick}')
            if cpu.eval_done():break
            actions=choices(cpu,rng)
            if tick%31==0:actions[(cpu.substep==0)&(np.arange(4)%2==0)]=-1
            cpu.step(actions);gpu.step(torch.tensor(actions,device=device))
        else:pytest.fail('refill games did not finish')
        assert len(cpu.eval_results())==16


def test_bf16_incremental_compacted_decoder_cuda():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from gfrl.gpu.search_inference import IncrementalBelief
    from torch.nn.attention import sdpa_kernel,SDPBackend
    belief=Belief(ModelConfig(d=128,heads=4,depth=2,ff=512),BeliefConfig(heads=4)).cuda().eval()
    env=VecEnv(8,1,42);obs={k:torch.tensor(getattr(env,k),device='cuda') for k in OBS_FIELDS}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
        _,memory,padding=belief.memory(obs)
        counts=torch.tensor([0,1,3,7,9,12,17,18],device='cuda')
        decoder=IncrementalBelief(belief,memory,padding,counts)
        inputs=torch.zeros((8,1),device='cuda',dtype=torch.long)
        for t in range(18):
            active=torch.nonzero(counts>t,as_tuple=True)[0]
            expected=belief.decode(memory,padding,inputs,counts)[:,-1]
            actual=decoder.step(inputs[active,-1],rows=active)
            torch.testing.assert_close(actual,expected[active],atol=.04,rtol=.02)
            card=expected.argmax(-1)+1;inputs=torch.cat((inputs,card[:,None]),1)


def test_bf16_shared_belief_memory_cuda():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from gfrl.gpu.search_inference import IncrementalBelief
    from torch.nn.attention import sdpa_kernel,SDPBackend
    belief=Belief(ModelConfig(d=128,heads=4,depth=2,ff=512),BeliefConfig(heads=4)).cuda().eval()
    env=VecEnv(3,1,42);obs={k:torch.tensor(getattr(env,k),device='cuda') for k in OBS_FIELDS}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
        _,memory,padding=belief.memory(obs);owners=torch.arange(3,device='cuda').repeat_interleave(16)
        counts=torch.arange(48,device='cuda')%19;inputs=torch.zeros((48,1),device='cuda',dtype=torch.long)
        kvs=[]
        for layer in belief.decoder.layers:
            att=layer.multihead_attn;d=att.embed_dim
            k,v=torch.nn.functional.linear(memory,att.in_proj_weight[d:],att.in_proj_bias[d:]).chunk(2,-1)
            kvs.append((IncrementalBelief.heads(k,att),IncrementalBelief.heads(v,att)))
        mask=memory.new_zeros((3,1,1,memory.shape[1])).masked_fill(padding[:,None,None,:],-torch.inf)
        decoder=IncrementalBelief(belief,memory,padding,counts,shared=(kvs,mask,owners))
        for t in range(18):
            expected=belief.decode(memory[owners],padding[owners],inputs,counts)[:,-1]
            actual=decoder.step(inputs[:,-1])
            torch.testing.assert_close(actual,expected,atol=.04,rtol=.02)
            inputs=torch.cat((inputs,(expected.argmax(-1)+1)[:,None]),1)

@pytest.mark.parametrize('response',[False,True])
def test_shared_world_clone_and_candidate_cursor_cuda(response):
    """Read-only root sharing preserves prefixes; cloned rollouts replay exactly."""
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from gfrl.gpu.search_env import GpuCandidateEnv
    from gfrl.actor import PREFIX_FIELDS
    rng=np.random.default_rng(671)
    cpu=VecEnv(2,1,eval_pairs=1,opponent='external',eval_seed=19,track_projection=True)
    _core._test_dream_root(cpu,response,6)
    with CudaFixtureEnv(2,games=2,eval_seed=19,device='cuda') as src, \
         GpuSearchEnv(16,max_decisions=1,device='cuda') as base, \
         GpuSearchEnv(32,max_decisions=4,device='cuda') as clone:
        src.fixture(int(response),6)
        rr=np.zeros(16,np.int32)
        aa=cpu.label_opp_hand[rr].astype(np.int32)
        seeds=rng.integers(2**64-1,size=16,dtype=np.uint64)
        base.load_worlds(src,torch.tensor(rr,device='cuda'),torch.tensor(aa,device='cuda'),torch.tensor(seeds,device='cuda'))
        ii=np.arange(32,dtype=np.int32)%16
        index=torch.tensor(ii,device='cuda')
        ref=SearchEnv(32,1,max_decisions=4)
        ref.load_worlds(cpu,rr[ii],aa[ii],seeds[ii])
        clone._module.launch('gpu_search_clone',clone._args()+[pointer(base._storage),pointer(base._pointers),pointer(index),pointer(clone.world_status),ct.c_int(32)],n=32)
        clone._observe();clone._actors()
        for tick in range(1000):
            clone._actors();equal(ref,clone,(*FIELDS,'world_status','searcher_seat','result'),f'clone {tick}')
            if ref.episode_done.all():break
            action=choices(ref,rng);ref.step(action);clone.step(torch.tensor(action,device='cuda'))
        else:pytest.fail('clones did not terminate')
        # Cursor copies remain independent even when several rows share a base.
        cursor=GpuCandidateEnv(base,index,src,torch.zeros(32,device='cuda',dtype=torch.long))
        short=SearchEnv(32,1,max_decisions=1)
        short.load_worlds(cpu,rr[ii],aa[ii],seeds[ii])
        try:
            for tick in range(100):
                np.testing.assert_array_equal(short.stage,cursor.stage.cpu().numpy())
                active=short.stage>=0
                for k in (*PREFIX_FIELDS,'actor_seat','substep'):
                    np.testing.assert_array_equal(getattr(short,k)[active],getattr(cursor,k).cpu().numpy()[active],err_msg=f'cursor {tick}: {k}')
                if not active.any():break
                action=choices(short,rng);short.step(action);cursor.step(torch.tensor(action,device='cuda'))
            else:pytest.fail('candidate cursor did not complete')
        finally:cursor.close()


@pytest.mark.parametrize('self_kernel',[False,True])
@pytest.mark.parametrize('worlds',[1,7,509])
def test_bf16_belief_graph_cuda(self_kernel,worlds,monkeypatch):
    monkeypatch.setenv('GFRL_GPU_BELIEF_SELF',str(int(self_kernel)))
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from gfrl.gpu.search_inference import IncrementalBelief,Bf16BeliefGraph
    from gfrl.search import load_search_belief
    import os
    path=os.environ.get('GFRL_GPU_SEARCH_BELIEF')
    belief=load_search_belief(path,device='cuda') if path else Belief(ModelConfig(d=128,heads=4,depth=2,ff=512),BeliefConfig(heads=4)).cuda().eval()
    env=VecEnv(3,1,42);obs={k:torch.tensor(getattr(env,k),device='cuda') for k in OBS_FIELDS}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
        _,memory,padding=belief.memory(obs)
        owners=torch.arange(worlds,device='cuda')%3;counts=torch.arange(worlds,device='cuda')%19
        kvs=[]
        for layer in belief.decoder.layers:
            a=layer.multihead_attn;d=a.embed_dim
            k,v=torch.nn.functional.linear(memory,a.in_proj_weight[d:],a.in_proj_bias[d:]).chunk(2,-1)
            kvs.append((IncrementalBelief.heads(k,a),IncrementalBelief.heads(v,a)))
        mask=memory.new_zeros((3,1,1,len(memory[0]))).masked_fill(padding[:,None,None,:],-torch.inf)
        shared=(kvs,mask,owners)
        graph=Bf16BeliefGraph(belief,1<<(worlds-1).bit_length(),4,128)
        for reset in range(2):
            graph.begin(counts,shared)
            ref=IncrementalBelief(belief,memory,padding,counts,shared=shared)
            previous=torch.zeros(worlds,device='cuda',dtype=torch.long)
            for _ in range(18):
                expected=ref.step(previous);actual=graph.step(previous)
                torch.testing.assert_close(actual,expected,atol=.04,rtol=.02)
                previous=expected.argmax(-1)+1


def test_device_position_incremental_decoder_cpu():
    """The graph's full-cache prefix mask matches the original causal prefix."""
    from gfrl.gpu.search_inference import IncrementalBelief
    torch.manual_seed(184)
    belief=Belief(ModelConfig(d=16,heads=2,depth=1,ff=32),BeliefConfig(heads=2,depth=1,ff=32)).eval()
    env=VecEnv(8,1,42);obs={k:torch.tensor(getattr(env,k)) for k in OBS_FIELDS}
    with torch.inference_mode():
        _,memory,padding=belief.memory(obs);counts=torch.arange(8)
        ref=IncrementalBelief(belief,memory,padding,counts)
        actual=IncrementalBelief(belief,memory,padding,counts)
        actual.self_kv=[tuple(k.new_zeros((8,k.shape[1],32,k.shape[3])) for _ in range(2)) for k,_ in actual.self_kv]
        previous=torch.zeros(8,dtype=torch.long)
        for t in range(18):
            expected=ref.step(previous);found=actual.step(previous,position=torch.tensor(t))
            torch.testing.assert_close(found,expected,atol=3e-5,rtol=2e-5)
            previous=expected.argmax(-1)+1


@pytest.mark.parametrize('response',[False,True])
def test_bf16_ar_graph_does_not_read_hidden_hand_cuda(response,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import os
    from gfrl.evaluation import load_policy_model
    from gfrl.search import load_search_belief
    for key in ('GFRL_GPU_SEARCH_BELIEF_CACHE','GFRL_GPU_SEARCH_BELIEF_KV','GFRL_GPU_SEARCH_BELIEF_GRAPH'):
        monkeypatch.setenv(key,'1')
    path=os.environ.get('GFRL_GPU_SEARCH_CHECKPOINT');bp=os.environ.get('GFRL_GPU_SEARCH_BELIEF')
    torch.manual_seed(87)
    model=load_policy_model(path,'cuda') if path else Model(ModelConfig(d=128,heads=4,depth=1,ff=128)).cuda().eval()
    belief=load_search_belief(bp,device='cuda') if bp else Belief(ModelConfig(d=128,heads=4,depth=1,ff=128),BeliefConfig(heads=4,depth=1,ff=128)).cuda().eval()
    config=SearchConfig(worlds=16,candidates=4,candidate_samples=8,depth=2,search_response=True,
                        skip_forced_inference=True,root_batch_size=64,max_inference_batch=8192)
    results=[]
    for changed in (False,True):
        with CudaFixtureEnv(64,games=64,eval_seed=12,device='cuda',pooled=True) as env:
            if response:env.fixture(3,6,False,19)
            if changed:env.fixture(4,6,False,1234)
            agent=GpuSearchAgent(model,belief,'cuda',config=config,seed=55,dtype='bf16')
            try:
                with _policy_rng(42,(agent,)):agent.search(env,torch.arange(64,device='cuda'),report_values=True)
                assert agent._belief_graphs,'AR graph path was not exercised'
                results.append(agent.last_search)
            finally:agent.close()
    assert results[0]==results[1]


@pytest.mark.parametrize('response',[False,True])
@pytest.mark.parametrize('report',[False,True])
def test_packed_record_blocks_match_legacy_and_learner(backend,response,report):
    from gfrl.gpu.search_records import finish_outcomes,pack_blocks
    factory,device=backend
    model=small_model()
    config=SearchConfig(worlds=3,candidates=4,candidate_samples=8,depth=2,search_response=True)
    with factory(2,games=2,eval_seed=12,device=device) as env:
        if response:env.fixture(3,6,False,19)
        a=GpuSearchAgent(model,baseline(),device,config=config,seed=55,record=True,env_factory=factory)
        b=GpuSearchAgent(model,baseline(),device,config=config,seed=55,record=True,env_factory=factory,packed_record=True)
        try:
            selected=[]
            for agent in (a,b):
                with _policy_rng(42,(agent,)):selected.append(agent.search(env,torch.arange(2,device=device),report_values=report))
            assert selected[0]==selected[1]
            if report:assert a.last_search==b.last_search
            for record in a.records:record['outcome']=np.array([0,0,1],np.uint8)
            finish_outcomes(b.record_blocks,np.zeros((2,2),np.int8),np.zeros(2,bool))
            x=pack_records(a.records,{});y=pack_blocks(b.record_blocks,{})
            assert b.record_count==len(a.records)
            for group in ('obs','prefix'):
                for k in x[group]:
                    np.testing.assert_array_equal(x[group][k].numpy(),y[group][k].numpy(),err_msg=k)
            for k in ('candidate_offsets','stage_offsets','choice','valid_worlds','game','seat','decision','phase','outcome'):
                np.testing.assert_array_equal(x[k],y[k],err_msg=k)
            for k in ('q','probabilities','log_pi','log_rho'):
                torch.testing.assert_close(x[k],y[k],atol=1e-14,rtol=1e-14,equal_nan=True)
            rows=np.arange(b.record_count)
            lx,_=objective(model,model,prepare_batch(x,rows,device))
            ly,_=objective(model,model,prepare_batch(y,rows,device))
            torch.testing.assert_close(lx,ly,atol=0,rtol=0)
        finally:a.close();b.close()
