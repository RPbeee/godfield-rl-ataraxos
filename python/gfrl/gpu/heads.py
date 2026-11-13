"""Device-counted policy heads; finished lanes do not gather full hand tensors."""
import torch
import triton as tr
import triton.language as tl
from .packed import _linear, _scatter_index


@tr.jit
def _query_input(Encoded,Ids,Count,Kind,Stage,Target,Hp,Chosen,ChosenCount,KindEmb,StageEmb,TargetEmb,HpEmb,Out,D:tl.constexpr,SM:tl.constexpr):
    count=tl.load(Count);d=tl.arange(0,D)
    for r in range(tl.program_id(0),count,SM):
        row=tl.load(Ids+r);n=tl.load(ChosenCount+row).to(tl.int32)
        selected=tl.full((D,),0,tl.float32)
        for j in tl.static_range(18):
            card=tl.maximum(tl.load(Chosen+row*18+j).to(tl.int32),0)
            selected+=tl.load(Encoded+(row*21+3+card)*D+d,j<n,0)
        kind=tl.load(Kind+row).to(tl.int32)+1;stage=tl.load(Stage+row).to(tl.int32)
        target=tl.load(Target+row).to(tl.int32)+1;hp=tl.load(Hp+row).to(tl.int32)+1
        base=Out+r*6*D+d
        tl.store(base,tl.load(Encoded+row*21*D+d))
        tl.store(base+D,tl.load(KindEmb+kind*D+d));tl.store(base+2*D,selected)
        tl.store(base+3*D,tl.load(StageEmb+stage*D+d))
        tl.store(base+4*D,tl.load(TargetEmb+target*D+d));tl.store(base+5*D,tl.load(HpEmb+hp*D+d))


@tr.jit
def _log_probs(Encoded,CardQuery,Small,Ids,Count,Stage,Mask,Out,D:tl.constexpr,SM:tl.constexpr):
    count=tl.load(Count);j=tl.arange(0,32);d=tl.arange(0,D)
    for r in range(tl.program_id(0),count,SM):
        row=tl.load(Ids+r);stage=tl.load(Stage+row).to(tl.int32)
        valid=tl.load(Mask+row*21+j,j<21,0)
        if stage==1:
            hand=tl.load(Encoded+(row*21+3+j[:,None])*D+d[None,:],((j<18)&valid)[:,None],0)
            query=tl.load(CardQuery+r*D+d).to(tl.float32)
            logits=tl.sum(hand*query[None,:],1)
            stop=tl.load(Small+r*56+10).to(tl.float32)
            logits=tl.where(j==18,stop,logits)
        else:
            offset=tl.where(stage==0,0,tl.where(stage==2,11,tl.where(stage==3,14,35)))
            width=tl.where(stage==0,10,tl.where(stage==2,3,21))
            logits=tl.load(Small+r*56+offset+j,j<width,0).to(tl.float32)
        logits=tl.where(valid,logits,-1.e9)
        shifted=logits-tl.max(logits,0)
        lp=shifted-tl.log(tl.sum(tl.exp(shifted),0))
        tl.store(Out+row*21+j,lp,j<21)


class SparseHeads:
    def __init__(self,model,env):
        self.model,self.env=model,env
        self.sm=torch.cuda.get_device_properties(env.device).multi_processor_count*2
        self.weights={}
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        m=self.model
        for key,module in [('q0',m.query[0]),('q2',m.query[2]),('card',m.card_query),('value',m.value_head)]:
            if key not in self.weights:
                self.weights[key]=(module.weight.detach().bfloat16().contiguous(),module.bias.detach().bfloat16().contiguous())
            else:
                w,b=self.weights[key];w.copy_(module.weight);b.copy_(module.bias)
        modules=[m.kind_head,m.stop_head,m.target_head,m.hp_head,m.mp_head]
        w=torch.cat([v.weight.detach() for v in modules]).bfloat16()
        b=torch.cat([v.bias.detach() for v in modules]).bfloat16()
        if 'small' not in self.weights:self.weights['small']=(w,b)
        else:self.weights['small'][0].copy_(w);self.weights['small'][1].copy_(b)

    def values(self,encoded):
        w,b=self.weights['value']
        return torch.nn.functional.linear(encoded[:,0].bfloat16(),w,b).float()

    def mm(self,x,key,ids,count,*,gelu=False):
        w,b=self.weights[key]
        out=torch.empty((self.env.num_envs,w.shape[0]),device=x.device,dtype=torch.bfloat16)
        _linear[(self.sm,)](x,w,b,out,ids,count,w.shape[1],w.shape[0],False,False,self.sm,
                            BM=64,BN=64,BK=32,GELU=gelu,num_warps=4)
        return out

    def __call__(self,encoded,active=None):
        e,m=self.env,self.model;n=e.num_envs;d=m.config.d
        if active is None:active=torch.ones(n,device=e.device,dtype=torch.bool)
        ranks=active.to(torch.int32).cumsum(0,dtype=torch.int32)
        ids=torch.empty_like(ranks);count=ranks[-1:]
        _scatter_index[(tr.cdiv(n,256),)](active,ranks,ids,n,256)
        features=torch.empty((n,6*d),device=e.device,dtype=torch.bfloat16)
        _query_input[(self.sm,)](encoded,ids,count,e.chosen_kind,e.stage,e.chosen_target,e.chosen_alloc_hp,e.chosen_cards,e.chosen_count,
                                 m.kind_embedding.weight,m.stage_embedding.weight,m.target_embedding.weight,m.hp_embedding.weight,features,d,self.sm)
        q=self.mm(self.mm(features,'q0',ids,count,gelu=True),'q2',ids,count)
        card=self.mm(q,'card',ids,count);small=self.mm(q,'small',ids,count)
        lp=torch.zeros((n,21),device=e.device,dtype=torch.float32)
        _log_probs[(self.sm,)](encoded,card,small,ids,count,e.stage,e.mask,lp,d,self.sm)
        return lp
