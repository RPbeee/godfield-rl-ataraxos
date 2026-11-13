"""Dynamically scaled E4M3 products; FP32 master parameters and Adam state.

Quantization also writes the transposed layout needed by backward. The maximum
is reduced without materializing FP32 copies of the large activation tensors.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _amax(X,Partial,Size,B:tl.constexpr):
    at=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+at,at<Size,0).to(tl.float32)
    tl.store(Partial+tl.program_id(0),tl.max(tl.abs(x),0))


@tr.jit
def _scale(Partial,Scale,N,B:tl.constexpr):
    at=tl.arange(0,B)
    accum=tl.full((B,),0,tl.float32)
    for start in range(tl.cdiv(N,B)):
        i=start*B+at
        accum=tl.maximum(accum,tl.load(Partial+i,i<N,0))
    tl.store(Scale,tl.maximum(tl.max(accum,0),1e-12)/448.)


@tr.jit
def _dual(X,Row,Col,Scale,M,K:tl.constexpr,MP, BM:tl.constexpr=32, BK:tl.constexpr=64):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    k=tl.program_id(1)*BK+tl.arange(0,BK)
    x=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0).to(tl.float32)
    x=(x/tl.load(Scale)).to(Row.dtype.element_ty)
    tl.store(Row+m[:,None]*K+k[None,:],x,(m[:,None]<MP)&(k[None,:]<K))
    tl.store(Col+k[None,:]*MP+m[:,None],x,(m[:,None]<MP)&(k[None,:]<K))


def quantize_dual(x):
    x=x.contiguous()
    m,k=x.shape
    mp=tr.cdiv(m,16)*16
    partial=torch.empty((tr.cdiv(m*k,4096),),device=x.device,dtype=torch.float32)
    scale=torch.empty((),device=x.device,dtype=torch.float32)
    row=torch.empty((mp,k),device=x.device,dtype=torch.float8_e4m3fn)
    col=torch.empty((k,mp),device=x.device,dtype=torch.float8_e4m3fn)
    _amax[(len(partial),)](x,partial,m*k,4096)
    _scale[(1,)](partial,scale,len(partial),1024)
    _dual[(tr.cdiv(mp,32),tr.cdiv(k,64))](x,row,col,scale,m,k,mp,32,64,num_warps=4)
    return row,col,scale


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias):
        a,at,sa=quantize_dual(x)
        w,wt,sw=quantize_dual(weight)
        out=torch._scaled_mm(a,w.T,sa,sw,bias=bias.to(torch.bfloat16),
                             out_dtype=torch.bfloat16,use_fast_accum=False)
        ctx.save_for_backward(at,wt,sa,sw)
        ctx.n=x.shape[0]
        ctx.input_dtype=x.dtype
        return out[:ctx.n]

    @staticmethod
    def backward(ctx,grad):
        at,wt,sa,sw=ctx.saved_tensors
        g,gt,sg=quantize_dual(grad)
        dx=torch._scaled_mm(g,wt.T,sg,sw,out_dtype=torch.bfloat16,use_fast_accum=False)
        dw=torch._scaled_mm(gt,at.T,sg,sa,out_dtype=torch.float32,use_fast_accum=False)
        return dx[:ctx.n].to(ctx.input_dtype),dw,grad.float().sum(0)


linear=Linear.apply
