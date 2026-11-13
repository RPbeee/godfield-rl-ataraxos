"""Short-sequence, noncausal packed attention and its full backward."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _forward_tile(Q,K,V,CQ,CK,O,LSE,H:tl.constexpr,D:tl.constexpr,
             SQ:tl.constexpr,SK:tl.constexpr,SV:tl.constexpr, M:tl.constexpr=64,N:tl.constexpr=128):
    seq=tl.program_id(0);head=tl.program_id(1);tile=tl.program_id(2)
    qs=tl.load(CQ+seq);qe=tl.load(CQ+seq+1)
    ks=tl.load(CK+seq);ke=tl.load(CK+seq+1)
    r=tile*M+tl.arange(0,M);c=tl.arange(0,N);d=tl.arange(0,D)
    if tile*M < qe-qs:
        q=tl.load(Q+(qs+r[:,None])*SQ+head*D+d[None,:],r[:,None]<qe-qs,0)
        k=tl.load(K+(ks+c[None,:])*SK+head*D+d[:,None],c[None,:]<ke-ks,0)
        score=tl.dot(q,k)*(D**-.5)
        score=tl.where(c[None,:]<ke-ks,score,-float('inf'))
        mx=tl.max(score,1)
        p=tl.exp(score-mx[:,None]);den=tl.sum(p,1)
        p=p/den[:,None]
        v=tl.load(V+(ks+c[:,None])*SV+head*D+d[None,:],c[:,None]<ke-ks,0)
        out=tl.dot(p.to(q.dtype),v)
        tl.store(O+(qs+r[:,None])*H*D+head*D+d[None,:],out,r[:,None]<qe-qs)
        tl.store(LSE+(qs+r)*H+head,mx+tl.log(den),r<qe-qs)


@tr.jit
def _backward_tile(Q,K,V,CQ,CK,O,LSE,DO,DQ,DK,DV,H:tl.constexpr,D:tl.constexpr,
              SQ:tl.constexpr,SK:tl.constexpr,SV:tl.constexpr,M:tl.constexpr=32,N:tl.constexpr=128):
    seq=tl.program_id(0);head=tl.program_id(1)
    qs=tl.load(CQ+seq);qe=tl.load(CQ+seq+1)
    ks=tl.load(CK+seq);ke=tl.load(CK+seq+1)
    r=tl.arange(0,M);c=tl.arange(0,N);d=tl.arange(0,D)
    k=tl.load(K+(ks+c[:,None])*SK+head*D+d[None,:],c[:,None]<ke-ks,0)
    v=tl.load(V+(ks+c[:,None])*SV+head*D+d[None,:],c[:,None]<ke-ks,0)
    dk=tl.full((N,D),0,tl.float32);dv=tl.full((N,D),0,tl.float32)
    for tile in range(tl.cdiv(qe-qs,M)):
        rows=tile*M+r
        q=tl.load(Q+(qs+rows[:,None])*SQ+head*D+d[None,:],rows[:,None]<qe-qs,0)
        do=tl.load(DO+(qs+rows[:,None])*H*D+head*D+d[None,:],rows[:,None]<qe-qs,0)
        out=tl.load(O+(qs+rows[:,None])*H*D+head*D+d[None,:],rows[:,None]<qe-qs,0).to(tl.float32)
        lse=tl.load(LSE+(qs+rows)*H+head,rows<qe-qs,0)
        score=tl.dot(q,tl.trans(k))*(D**-.5)
        p=tl.exp(score-lse[:,None])
        p=tl.where((rows[:,None]<qe-qs)&(c[None,:]<ke-ks),p,0.)
        dv+=tl.dot(tl.trans(p.to(q.dtype)),do)
        dp=tl.dot(do,tl.trans(v))
        delta=tl.sum(do.to(tl.float32)*out,1)
        ds=(p*(dp-delta[:,None])*(D**-.5)).to(q.dtype)
        dq=tl.dot(ds,k)
        dk+=tl.dot(tl.trans(ds),q)
        tl.store(DQ+(qs+rows[:,None])*H*D+head*D+d[None,:],dq,rows[:,None]<qe-qs)
    tl.store(DK+(ks+c[:,None])*H*D+head*D+d[None,:],dk,c[:,None]<ke-ks)
    tl.store(DV+(ks+c[:,None])*H*D+head*D+d[None,:],dv,c[:,None]<ke-ks)


@tr.jit
def _forward(Q,K,V,CQ,CK,O,LSE,H:tl.constexpr,D:tl.constexpr,
             SQ:tl.constexpr,SK:tl.constexpr,SV:tl.constexpr,M:tl.constexpr,N:tl.constexpr):
    seq=tl.program_id(0)
    length=tl.load(CK+seq+1)-tl.load(CK+seq)
    if length<=32:
        _forward_tile(Q,K,V,CQ,CK,O,LSE,H,D,SQ,SK,SV,32,32)
    elif length<=64:
        _forward_tile(Q,K,V,CQ,CK,O,LSE,H,D,SQ,SK,SV,M,64)
    else:
        _forward_tile(Q,K,V,CQ,CK,O,LSE,H,D,SQ,SK,SV,M,128)


@tr.jit
def _backward(Q,K,V,CQ,CK,O,LSE,DO,DQ,DK,DV,H:tl.constexpr,D:tl.constexpr,
              SQ:tl.constexpr,SK:tl.constexpr,SV:tl.constexpr,M:tl.constexpr,N:tl.constexpr):
    seq=tl.program_id(0)
    length=tl.load(CK+seq+1)-tl.load(CK+seq)
    if length<=32:
        _backward_tile(Q,K,V,CQ,CK,O,LSE,DO,DQ,DK,DV,H,D,SQ,SK,SV,M,32)
    elif length<=64:
        _backward_tile(Q,K,V,CQ,CK,O,LSE,DO,DQ,DK,DV,H,D,SQ,SK,SV,M,64)
    else:
        _backward_tile(Q,K,V,CQ,CK,O,LSE,DO,DQ,DK,DV,H,D,SQ,SK,SV,M,128)


class ShortAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,cq,ck,max_q,max_k):
        if max_k>128:
            raise ValueError('short attention supports at most 128 tokens')
        out=torch.empty_like(q, memory_format=torch.contiguous_format)
        lse=torch.empty(q.shape[:2],device=q.device,dtype=torch.float32)
        m=32 if max_q<=32 else 64
        _forward[(len(cq)-1,q.shape[1],tr.cdiv(max_q,m))](q,k,v,cq,ck,out,lse,
            q.shape[1],q.shape[2],q.stride(0),k.stride(0),v.stride(0),M=m,N=128,num_warps=4)
        ctx.save_for_backward(q,k,v,cq,ck,out,lse)
        return out

    @staticmethod
    def backward(ctx,grad):
        q,k,v,cq,ck,out,lse=ctx.saved_tensors
        dq=torch.empty_like(q,memory_format=torch.contiguous_format)
        dk=torch.empty_like(k,memory_format=torch.contiguous_format)
        dv=torch.empty_like(v,memory_format=torch.contiguous_format)
        grad=grad.contiguous()
        _backward[(len(cq)-1,q.shape[1])](q,k,v,cq,ck,out,lse,grad,dq,dk,dv,
            q.shape[1],q.shape[2],q.stride(0),k.stride(0),v.stride(0),M=32,N=128,num_warps=4)
        return dq,dk,dv,None,None,None,None


short_attention=ShortAttention.apply
