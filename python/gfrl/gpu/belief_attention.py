"""One-query, public-memory cross attention for incremental AR belief decoding."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _cross(Q,K,V,Mask,Rows,Out,T:tl.constexpr,D:tl.constexpr,H:tl.constexpr,
           QS0:tl.constexpr,QS1:tl.constexpr,KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,
           VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,MS0:tl.constexpr,BT:tl.constexpr,BD:tl.constexpr):
    r=tl.program_id(0);head=tl.program_id(1);source=tl.load(Rows+r)
    t=tl.arange(0,BT);d=tl.arange(0,BD)
    q=tl.load(Q+r*QS0+head*QS1+d,d<D,0).to(tl.float32)
    k=tl.load(K+source*KS0+head*KS1+t[:,None]*KS2+d[None,:],(t[:,None]<T)&(d[None,:]<D),0).to(tl.float32)
    score=tl.sum(k*q[None,:],1)*(D**-0.5)
    bias=tl.load(Mask+source*MS0+t,t<T,-float('inf'))
    score=tl.where(t<T,score+bias,-float('inf'))
    p=tl.exp(score-tl.max(score,0));den=tl.sum(p,0)
    v=tl.load(V+source*VS0+head*VS1+t[:,None]*VS2+d[None,:],(t[:,None]<T)&(d[None,:]<D),0).to(tl.float32)
    out=tl.sum(p[:,None]*v,0)/den
    tl.store(Out+(r*H+head)*D+d,out,d<D)


def cross_attention(q,k,v,mask,rows):
    n,h,_,d=q.shape;t=k.shape[2]
    out=torch.empty((n,h,1,d),device=q.device,dtype=q.dtype)
    _cross[(n,h)](q,k,v,mask,rows,out,t,d,h,q.stride(0),q.stride(1),
        k.stride(0),k.stride(1),k.stride(2),v.stride(0),v.stride(1),v.stride(2),mask.stride(0),
        tr.next_power_of_2(t),tr.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
    return out
