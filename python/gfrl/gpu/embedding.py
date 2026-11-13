"""Fused ordered embedding sum with a tensor-core vocabulary gradient reduction.

The reduction rounds the upstream embedding gradient to bf16; accumulation and
master gradients stay fp32. No straight-through approximation is used.
"""
import torch
from torch.nn import functional as F
import triton as tr
import triton.language as tl


@tr.jit
def _sum(Indices, Table, Out, N, D:tl.constexpr, C:tl.constexpr, R:tl.constexpr):
    row = tl.program_id(0)*R+tl.arange(0, R)
    d = tl.arange(0, tr.next_power_of_2(D))
    i = tl.load(Indices+row*C, row<N, 0)
    x = tl.load(Table+i[:, None]*D+d[None, :], (row<N)[:, None]&(d<D)[None, :], 0)
    for j in tl.static_range(1, C):
        i = tl.load(Indices+row*C+j, row<N, 0)
        x += tl.load(Table+i[:, None]*D+d[None, :], (row<N)[:, None]&(d<D)[None, :], 0)
    tl.store(Out+row[:, None]*D+d[None, :], x, (row<N)[:, None]&(d<D)[None, :])


@tr.jit
def _counts(Indices, Counts, N, C:tl.constexpr, V:tl.constexpr, B:tl.constexpr):
    row = tl.program_id(0)*B+tl.arange(0, B)
    for j in tl.static_range(C):
        i = tl.load(Indices+row*C+j, row<N, 0)
        count = tl.full((B,), 1, tl.int32)
        first = tl.full((B,), True, tl.int1)
        # Only card slots can repeat an id; all other fields have disjoint tables.
        if 2 <= j < 10:
            count = tl.full((B,), 0, tl.int32)
            for k in tl.static_range(2, 10):
                other = tl.load(Indices+row*C+k, row<N, -1)
                count += (i == other).to(tl.int32)
                if k < j:
                    first &= i != other
        tl.store(Counts+row*V+i, count, (row<N)&first)


class EmbeddingSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, indices):
        n, c = indices.shape
        out = table.new_empty((n, table.shape[1]))
        _sum[(tr.cdiv(n, 4),)](indices, table, out, n, table.shape[1], c, 4)
        ctx.save_for_backward(indices)
        ctx.shape = table.shape
        return out

    @staticmethod
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        n, c = indices.shape
        v, d = ctx.shape
        width = tr.cdiv(v, 16)*16
        padded = tr.cdiv(n, 16)*16
        counts = torch.zeros((padded, width), device=grad.device, dtype=torch.bfloat16)
        _counts[(tr.cdiv(n, 128),)](indices, counts, n, c, width, 128)
        # FP32 output avoids an additional rounding of the master gradient.
        # The CUDA out_dtype GEMM requires a tensor-core aligned reduction
        # dimension; otherwise torch 2.15/cu132 can silently omit its tail.
        gradient = F.pad(grad.to(torch.bfloat16), (0, 0, 0, padded-n))
        result = torch.mm(counts.T, gradient, out_dtype=torch.float32)
        return result[:v], None
