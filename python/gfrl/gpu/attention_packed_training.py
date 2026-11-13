"""Grouped short attention with interleaved QKV gradients (full bidirectional)."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _fwd_tile(QKV, CQ, CK, O, LSE, H:tl.constexpr, D:tl.constexpr,
              G:tl.constexpr, M:tl.constexpr, N:tl.constexpr):
    seq=tl.program_id(0); hg=tl.program_id(1); tile=tl.program_id(2)
    qs=tl.load(CQ+seq); qe=tl.load(CQ+seq+1)
    ks=tl.load(CK+seq); ke=tl.load(CK+seq+1)
    hs=hg*G+tl.arange(0,G)
    r=tile*M+tl.arange(0,M); c=tl.arange(0,N); d=tl.arange(0,tr.next_power_of_2(D))
    if tile*M < qe-qs:
        q=tl.load(QKV+(ks+r[None,:,None])*3*H*D+hs[:,None,None]*D+d[None,None,:],(r[None,:,None]<qe-qs)&(d[None,None,:]<D),0)
        k=tl.load(QKV+(ks+c[None,None,:])*3*H*D+H*D+hs[:,None,None]*D+d[None,:,None],(c[None,None,:]<ke-ks)&(d[None,:,None]<D),0)
        s=tl.dot(q,k)*(D**-.5)
        s=tl.where(c[None,None,:]<ke-ks,s,-float('inf'))
        mx=tl.max(s,2); p=tl.exp(s-mx[:,:,None]); den=tl.sum(p,2)
        p=p/den[:,:,None]
        v=tl.load(QKV+(ks+c[None,:,None])*3*H*D+2*H*D+hs[:,None,None]*D+d[None,None,:],(c[None,:,None]<ke-ks)&(d[None,None,:]<D),0)
        out=tl.dot(p.to(q.dtype),v)
        tl.store(O+(qs+r[None,:,None])*H*D+hs[:,None,None]*D+d[None,None,:],out,(r[None,:,None]<qe-qs)&(d[None,None,:]<D))
        tl.store(LSE+(qs+r[None,:])*H+hs[:,None],mx+tl.log(den),r[None,:]<qe-qs)


@tr.jit
def _bwd_tile(QKV,CQ,CK,O,LSE,DO,DQKV,H:tl.constexpr,D:tl.constexpr,
              G:tl.constexpr,M:tl.constexpr,N:tl.constexpr):
    seq=tl.program_id(0); hg=tl.program_id(1)
    qs=tl.load(CQ+seq); qe=tl.load(CQ+seq+1)
    ks=tl.load(CK+seq); ke=tl.load(CK+seq+1)
    hs=hg*G+tl.arange(0,G); r=tl.arange(0,M); c=tl.arange(0,N); d=tl.arange(0,tr.next_power_of_2(D))
    k=tl.load(QKV+(ks+c[None,:,None])*3*H*D+H*D+hs[:,None,None]*D+d[None,None,:],(c[None,:,None]<ke-ks)&(d[None,None,:]<D),0)
    v=tl.load(QKV+(ks+c[None,:,None])*3*H*D+2*H*D+hs[:,None,None]*D+d[None,None,:],(c[None,:,None]<ke-ks)&(d[None,None,:]<D),0)
    dk=tl.full((G,N,tr.next_power_of_2(D)),0,tl.float32); dv=tl.full((G,N,tr.next_power_of_2(D)),0,tl.float32)
    for tile in range(tl.cdiv(qe-qs,M)):
        rows=tile*M+r
        q=tl.load(QKV+(ks+rows[None,:,None])*3*H*D+hs[:,None,None]*D+d[None,None,:],(rows[None,:,None]<qe-qs)&(d[None,None,:]<D),0)
        go=tl.load(DO+(qs+rows[None,:,None])*H*D+hs[:,None,None]*D+d[None,None,:],(rows[None,:,None]<qe-qs)&(d[None,None,:]<D),0)
        out=tl.load(O+(qs+rows[None,:,None])*H*D+hs[:,None,None]*D+d[None,None,:],(rows[None,:,None]<qe-qs)&(d[None,None,:]<D),0).to(tl.float32)
        lse=tl.load(LSE+(qs+rows[None,:])*H+hs[:,None],rows[None,:]<qe-qs,0)
        s=tl.dot(q,tl.trans(k,0,2,1))*(D**-.5)
        p=tl.exp(s-lse[:,:,None])
        p=tl.where((rows[None,:,None]<qe-qs)&(c[None,None,:]<ke-ks),p,0.)
        dv+=tl.dot(tl.trans(p.to(q.dtype),0,2,1),go)
        dp=tl.dot(go,tl.trans(v,0,2,1))
        delta=tl.sum(go.to(tl.float32)*out,2)
        ds=(p*(dp-delta[:,:,None])*(D**-.5)).to(q.dtype)
        dq=tl.dot(ds,k)
        dk+=tl.dot(tl.trans(ds,0,2,1),q)
        tl.store(DQKV+(ks+rows[None,:,None])*3*H*D+hs[:,None,None]*D+d[None,None,:],dq,(rows[None,:,None]<qe-qs)&(d[None,None,:]<D))
    # All unused final-layer queries have zero derivative.
    tl.store(DQKV+(ks+c[None,:,None])*3*H*D+hs[:,None,None]*D+d[None,None,:],0.,(c[None,:,None]<ke-ks)&(c[None,:,None]>=qe-qs)&(d[None,None,:]<D))
    tl.store(DQKV+(ks+c[None,:,None])*3*H*D+H*D+hs[:,None,None]*D+d[None,None,:],dk,(c[None,:,None]<ke-ks)&(d[None,None,:]<D))
    tl.store(DQKV+(ks+c[None,:,None])*3*H*D+2*H*D+hs[:,None,None]*D+d[None,None,:],dv,(c[None,:,None]<ke-ks)&(d[None,None,:]<D))


@tr.jit
def _fwd(QKV,CQ,CK,O,LSE,H:tl.constexpr,D:tl.constexpr,G:tl.constexpr,M:tl.constexpr):
    seq=tl.program_id(0); n=tl.load(CK+seq+1)-tl.load(CK+seq)
    if n<=32: _fwd_tile(QKV,CQ,CK,O,LSE,H,D,G,M,32)
    elif n<=64: _fwd_tile(QKV,CQ,CK,O,LSE,H,D,G,M,64)
    else: _fwd_tile(QKV,CQ,CK,O,LSE,H,D,G,M,128)


@tr.jit
def _bwd(QKV,CQ,CK,O,LSE,DO,DQKV,H:tl.constexpr,D:tl.constexpr,G:tl.constexpr,M:tl.constexpr):
    seq=tl.program_id(0); n=tl.load(CK+seq+1)-tl.load(CK+seq)
    if n<=32: _bwd_tile(QKV,CQ,CK,O,LSE,DO,DQKV,H,D,G,M,32)
    elif n<=64: _bwd_tile(QKV,CQ,CK,O,LSE,DO,DQKV,H,D,G,M,64)
    else: _bwd_tile(QKV,CQ,CK,O,LSE,DO,DQKV,H,D,G,M,128)


class PackedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cq, ck, query_count, max_q):
        n, _, h, d=qkv.shape
        out=qkv.new_empty((query_count,h,d))
        lse=torch.empty((query_count,h),device=qkv.device,dtype=torch.float32)
        _fwd[(len(cq)-1,h//2,tr.cdiv(max_q,32))](qkv,cq,ck,out,lse,h,d,2,32,num_warps=4)
        ctx.save_for_backward(qkv,cq,ck,out,lse)
        return out

    @staticmethod
    def backward(ctx,grad):
        qkv,cq,ck,out,lse=ctx.saved_tensors
        dqkv=torch.empty_like(qkv)
        h,d=qkv.shape[2:]
        _bwd[(len(cq)-1,h//2)](qkv,cq,ck,out,lse,grad.contiguous(),dqkv,h,d,2,32,num_warps=4)
        return dqkv,None,None,None,None


packed_attention=PackedAttention.apply
