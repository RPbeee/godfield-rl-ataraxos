#!/usr/bin/env python3
"""Paired FP32/bf16 searches: common public roots, worlds, candidates and RNG.

BF16 scores the common candidate prefixes with the production packed encoder /
SparseHeads. Rollouts use the respective policy/value precision. Belief-sample
variance is excluded by using the FP32-sampled worlds for both searches.
"""
import argparse,json,os,sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import numpy as np
import torch
from gfrl.distill import pack_records,prepare_batch
from gfrl.evaluation import load_policy_model,_policy_rng
from gfrl.model import OBS_FIELDS
from gfrl.gpu.search import GpuSearchAgent,public
from gfrl.gpu.search_env import GpuSearchEnv
from gfrl.search import SearchConfig,load_search_belief
from gfrl.runtime import configure_precision


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--belief',required=True)
    p.add_argument('--roots',type=int,default=3000);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);configure_precision('cuda',True)
    os.environ['GFRL_GPU_SEARCH_PACKED']='0';os.environ['GFRL_GPU_SEARCH_GRAPH']='0'
    os.environ['GFRL_GPU_SEARCH_BELIEF_CACHE']='1';os.environ['GFRL_GPU_SEARCH_BELIEF_KV']='1'
    model=load_policy_model(args.checkpoint,'cuda');belief=load_search_belief(args.belief,device='cuda')
    config=SearchConfig(worlds=16,candidates=16,candidate_samples=64,depth=3,beta=.1,search_response=True,
        skip_forced_inference=True,skip_single_candidate=True,max_inference_batch=8192,root_batch_size=64)
    main_agent=GpuSearchAgent(model,belief,config=config,seed=923000,dtype='bf16')
    ref=GpuSearchAgent(model,belief,config=replace(config,select='argmax'),seed=55,record=True,dtype='fp32')
    fast=GpuSearchAgent(model,belief,config=replace(config,select='argmax'),seed=55,dtype='bf16')
    sampler=ref._sample_worlds;rollouts=ref._rollouts
    samples=[];start=perf_counter();round_index=0
    try:
        with GpuSearchEnv(128,games=512,eval_seed=923000,device='cuda',pooled=True) as env,_policy_rng(923000,(main_agent,)):
            phase_column=env.obs_schema()['cat']['phase']['index']
            while bool((env.stage>=0).any()) and len(samples)<args.roots:
                alive=env.stage>=0;fresh=torch.nonzero(alive,as_tuple=True)[0]
                action=main_agent(env,fresh)
                supported=fresh[env.determinize_support(fresh.int())==0]
                # Spread across the game, including both response and main roots.
                if round_index>=2:
                    selected=supported[::2][:min(64,args.roots-len(samples))]
                    for phase in (1,2):
                        roots=selected[env.obs_cat[selected,0,phase_column]==phase]
                        if not len(roots):continue
                        seed=951000+len(samples)
                        with _policy_rng(seed,(ref,)):common=sampler(env,roots)
                        ref._sample_worlds=lambda src,rr:common
                        fast._sample_worlds=lambda src,rr:common
                        state=[]
                        def capture(*a,**kw):
                            if not state:state.append(torch.cuda.get_rng_state())
                            return rollouts(*a,**kw)
                        ref._rollouts=capture;ref.records.clear()
                        with _policy_rng(seed,(ref,)):ref.search(env,roots,report_values=True)
                        if not ref.last_search:continue
                        for record in ref.records:record['outcome']=np.array([0,0,1],np.uint8)
                        data=pack_records(ref.records,{})
                        batch=prepare_batch(data,np.arange(len(ref.records)),'cuda')
                        with torch.inference_mode():
                            encoded=fast._encode(batch['obs'])
                            stage_lp=[]
                            prefix=batch['prefix']
                            for lo in range(0,len(batch['owners']),8192):
                                rr=slice(lo,lo+8192)
                                lp=fast._heads(encoded[batch['owners'][rr]],{k:v[rr] for k,v in prefix.items()})
                                stage_lp.append(lp.gather(1,batch['choice'][rr,None]).flatten())
                            lp=torch.cat(stage_lp)
                            path_lp=torch.zeros(len(batch['probabilities']),device=lp.device,dtype=torch.float64).index_add_(0,batch['stage_owners'],lp.double()).cpu().tolist()
                        priors={};index=0
                        for report,record in zip(ref.last_search,ref.records):
                            priors[report['row']]=[
                                (tuple(c['path']),(path_lp[index+j],c['log_rho'])) for j,c in enumerate(record['candidates'])]
                            index+=len(record['candidates'])
                        with _policy_rng(seed,(fast,)):
                            torch.cuda.set_rng_state(state[0]);fast.search(env,roots,fixed_candidates=priors,report_values=True)
                        by_row={r['row']:r for r in fast.last_search}
                        for a in ref.last_search:
                            b=by_row[a['row']]
                            assert [c['path'] for c in a['candidates']]==[c['path'] for c in b['candidates']]
                            qa=np.array([c['q'] for c in a['candidates']]);qb=np.array([c['q'] for c in b['candidates']])
                            pa=np.array(a['probabilities']);pb=np.array(b['probabilities'])
                            samples.append(dict(phase=phase,mean_abs_q=float(abs(qa-qb).mean()),max_abs_q=float(abs(qa-qb).max()),
                                q_argmax=int(qa.argmax()==qb.argmax()),pi_argmax=int(pa.argmax()==pb.argmax()),
                                kl=float((pa*(np.log(pa.clip(1e-300))-np.log(pb.clip(1e-300)))).sum()),
                                reverse_kl=float((pb*(np.log(pb.clip(1e-300))-np.log(pa.clip(1e-300)))).sum()),
                                mean_abs_pi=float(abs(pa-pb).mean()),candidates=len(pa)))
                        print(f'compared={len(samples)} elapsed={perf_counter()-start:.1f}s',flush=True)
                remaining=alive
                while bool(remaining.any()):
                    env.step(torch.where(remaining,action,-1))
                    remaining &= ~env.decision_done & (env.stage>=0)
                    if bool(remaining.any()):action=main_agent(env,torch.nonzero(remaining,as_tuple=True)[0])
                round_index+=1
    finally:
        for a in (main_agent,ref,fast):a.close()
    if len(samples)<args.roots:raise RuntimeError(f'only {len(samples)} roots; increase source games')
    keys=('mean_abs_q','max_abs_q','q_argmax','pi_argmax','kl','reverse_kl','mean_abs_pi','candidates')
    result=dict(roots=len(samples),response_roots=sum(s['phase']==2 for s in samples),elapsed_seconds=perf_counter()-start,
        metrics={k:float(np.mean([s[k] for s in samples])) for k in keys},
        absolute_max_q=max(s['max_abs_q'] for s in samples),
        protocol='common FP32 sampled worlds, candidates and initial rollout RNG; BF16 priors, rollout policy and leaf value; independent sample variance excluded')
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
