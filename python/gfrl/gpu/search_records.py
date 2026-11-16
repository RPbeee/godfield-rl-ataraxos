"""Batched public records in the existing root/candidate/stage format v1."""
import torch

from ..actor import PREFIX_FIELDS
from ..model import OBS_FIELDS
from ..distill import FORMAT_VERSION


def record_block(src,roots,obs,prefixes,prefix_roots,order,paths,lp,lr,q,count,valid,singleton,decisions,samples):
    """Gather retained prefixes on device; transfer only the retained records."""
    keep=torch.nonzero(valid.any(1),as_tuple=True)[0]
    if not len(keep):return None
    owners,columns=torch.nonzero(valid,as_tuple=True)
    retained=paths[owners,columns]
    lengths=(retained>=0).sum(1)
    stage_owners=torch.repeat_interleave(torch.arange(len(owners),device=roots.device),lengths)
    steps=torch.arange(len(stage_owners),device=roots.device)-torch.repeat_interleave(lengths.cumsum(0)-lengths,lengths)
    sample_roots=torch.tensor([prefix_roots.get(i,-1) for i in range(len(roots))],device=roots.device)
    sample=sample_roots[owners]*samples+order[owners,columns]
    fields={k:v[sample[stage_owners],steps].cpu() for k,v in prefixes.items()}
    zero=torch.zeros(1,dtype=torch.long,device=roots.device)
    phase_index=src.obs_schema()['cat']['phase']['index']
    return dict(format_version=FORMAT_VERSION,
        obs={k:v[keep].cpu() for k,v in obs.items()},prefix=fields,
        candidate_offsets=torch.cat((zero,valid[keep].sum(1).cumsum(0))).cpu(),
        stage_offsets=torch.cat((zero,lengths.cumsum(0))).cpu(),
        choice=retained[stage_owners,steps].cpu(),
        probabilities=None,  # caller supplies the normalized search distribution
        q=torch.where(singleton[owners],torch.nan,q[owners,columns]).cpu(),
        log_pi=lp[owners,columns].cpu(),log_rho=lr[owners,columns].cpu(),valid_worlds=count[owners,columns].cpu(),
        game=src.episode_id.view(torch.int64)[roots[keep]].cpu(),seat=src.actor_seat[roots[keep]].cpu(),
        decision=(decisions[roots[keep]] if decisions is not None else torch.zeros_like(keep)).cpu(),
        phase=(obs['obs_cat'][keep,0,phase_index]==2).to(torch.uint8).cpu())


def finish_outcomes(blocks,outcomes,truncated):
    outcomes=torch.as_tensor(outcomes);truncated=torch.as_tensor(truncated)
    for block in blocks:
        game,seat=block['game'].long(),block['seat'].long()
        result=outcomes[game,seat]
        result=torch.where(truncated[game],0,result)
        target=torch.where(result>0,0,torch.where(result<0,1,2)).long()
        block['outcome']=torch.nn.functional.one_hot(target,3).to(torch.uint8)


def pack_blocks(blocks,metadata):
    if not blocks:raise ValueError('no recorded blocks')
    co=[torch.zeros(1,dtype=torch.long)];so=[torch.zeros(1,dtype=torch.long)]
    candidates=stages=0
    for block in blocks:
        co.append(block['candidate_offsets'][1:]+candidates)
        so.append(block['stage_offsets'][1:]+stages)
        candidates+=len(block['q']);stages+=len(block['choice'])
    data=dict(format_version=FORMAT_VERSION,metadata=metadata,
              obs={k:torch.cat([b['obs'][k] for b in blocks]) for k in OBS_FIELDS},
              prefix={k:torch.cat([b['prefix'][k] for b in blocks]) for k in PREFIX_FIELDS},
              candidate_offsets=torch.cat(co),stage_offsets=torch.cat(so))
    for k in ('choice','probabilities','q','log_pi','log_rho','valid_worlds','game','seat','decision','phase','outcome'):
        data[k]=torch.cat([b[k] for b in blocks])
    return data
