"""Hopper TMA loads/stores with a GPU-resident active row count."""
import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Descriptor storage is allocated during capture and owned by its graph pool.
tr.set_allocator(lambda size,alignment,stream:torch.empty(size,device='cuda',dtype=torch.uint8))


@tr.jit
def _matmul(A,W,B,Y,Count,K:tl.constexpr,N:tl.constexpr,SM:tl.constexpr,RELU:tl.constexpr,
            BM:tl.constexpr=128,BN:tl.constexpr=128,BK:tl.constexpr=128):
    active=tl.load(Count);ntiles=tl.cdiv(N,BN)
    for tile in tl.range(tl.program_id(0),tl.cdiv(active,BM)*ntiles,SM,flatten=True):
        m=(tile//ntiles)*BM;n=(tile%ntiles)*BN
        accum=tl.full((BM,BN),0,tl.float32)
        for kb in range(tl.cdiv(K,BK)):
            a=A.load([m,kb*BK]);w=W.load([n,kb*BK])
            accum=tl.dot(a,w.T,accum)
        col=n+tl.arange(0,BN)
        result=accum+tl.load(B+col,col<N,0).to(tl.float32)[None,:]
        if RELU:result=tl.maximum(result,0)
        Y.store([m,n],result.to(tl.bfloat16))


def linear(x,w,b,out,count,sm,*,relu=False):
    k=w.shape[1];n=w.shape[0]
    a=TensorDescriptor(x,list(x.shape),list(x.stride()),[128,128])
    weights=TensorDescriptor(w,[n,k],[k,1],[128,128])
    dest=TensorDescriptor(out,list(out.shape),list(out.stride()),[128,128])
    _matmul[(sm,)](a,weights,b,dest,count,k,n,sm,relu,num_warps=4,num_stages=3)
    return out
