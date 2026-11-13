"""CUDA kernel checks against PyTorch autograd and the original CPU targets."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('head_only', [False, True])
def test_short_attention_forward_backward(head_only):
    from gfrl.gpu.attention_training import short_attention
    from torch.nn.attention.varlen import varlen_attn
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(1)
    lengths = torch.tensor([1, 5, 21, 53, 119], device='cuda', dtype=torch.int32)
    qlengths = lengths.clamp_max(21) if head_only else lengths
    ck = torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    cq = torch.nn.functional.pad(qlengths.cumsum(0, dtype=torch.int32), (1, 0))
    tensors = [torch.randn(int(total), 8, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
               for total in [cq[-1], ck[-1], ck[-1]]]
    clones = [t.detach().clone().requires_grad_(True) for t in tensors]
    actual = short_attention(*tensors, cq, ck, 21 if head_only else 119, 119)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected = varlen_attn(*clones, cq, ck, 21 if head_only else 119, 119)
    torch.testing.assert_close(actual, expected, atol=.016, rtol=.02)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for a, b in zip(tensors, clones):
        assert float((a.grad.float()-b.grad.float()).norm()/b.grad.float().norm()) < .015


def test_embedding_vocabulary_reduction():
    from gfrl.gpu.embedding import EmbeddingSum, _counts
    torch.manual_seed(3)
    n, d = 3073, 32
    table = torch.randn(512, d, device='cuda', requires_grad=True)
    indices = torch.zeros(n, 25, device='cuda', dtype=torch.int64)
    indices[:, 0] = torch.randint(19, (n,), device='cuda')
    indices[:, 1] = torch.randint(9, (n,), device='cuda')+19
    indices[:, 2:10] = torch.randint(200, (n, 8), device='cuda')+28
    indices[:, 10:] = torch.arange(250, 265, device='cuda')
    counts = torch.zeros(n, 512, device='cuda', dtype=torch.bfloat16)
    _counts[((n+127)//128,)](indices, counts, n, 25, 512, 128)
    expected_counts = torch.zeros_like(counts).scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.bfloat16))
    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
    reference = table.detach().clone().requires_grad_(True)
    expected = reference[indices[:, 0]]
    for j in range(1, 25):
        expected = expected+reference[indices[:, j]]
    actual = EmbeddingSum.apply(table, indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert float((table.grad-reference.grad).norm()/reference.grad.norm()) < .004


@pytest.mark.parametrize('gamma', [1., .995, .5])
def test_gpu_targets_match_numpy(gamma):
    from gfrl.gpu.targets import compute_targets as gpu
    from gfrl.targets import compute_targets as cpu
    from gfrl.rollout import RolloutBatch
    rng = np.random.default_rng(42)
    n = 5003
    probs = torch.from_numpy(rng.dirichlet([1, 1, 1], n).astype(np.float32)).cuda()
    columns = dict(env=torch.from_numpy(rng.integers(0, 16, n)).cuda(),
                   seat=torch.from_numpy(rng.integers(0, 2, n)).cuda(),
                   game=torch.from_numpy(rng.integers(0, 3, n)).cuda(),
                   outcome=torch.from_numpy(rng.integers(-1, 2, n)).cuda(), old_value=probs)
    batch = RolloutBatch(columns, {})
    actual, expected = gpu(batch, gamma=gamma), cpu(batch, gamma=gamma)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a.cpu().numpy(), b)


def test_fp8_linear_backward_is_finite_and_close():
    from gfrl.gpu.fp8_training import linear
    torch.manual_seed(2)
    x = torch.randn(1003, 256, device='cuda', requires_grad=True)
    w = (torch.randn(256, 256, device='cuda')/16).requires_grad_(True)
    b = torch.randn(256, device='cuda', requires_grad=True)
    xx, ww, bb = (t.detach().clone().requires_grad_(True) for t in (x, w, b))
    actual = linear(x, w, b)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = torch.nn.functional.linear(xx, ww, bb)
    assert float(((actual.float()-expected.float()).norm()/expected.float().norm()).detach()) < .05
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for a, b in zip((x, w, b), (xx, ww, bb)):
        assert torch.isfinite(a.grad).all()
        assert float((a.grad-b.grad).norm()/b.grad.norm()) < .07


@pytest.mark.parametrize('head_only', [False, True])
def test_grouped_attention_interleaved_gradient(head_only):
    from gfrl.gpu.attention_packed_training import packed_attention
    from torch.nn.attention.varlen import varlen_attn
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(4)
    lengths=torch.tensor([1, 5, 21, 31, 32, 33, 53, 64, 65, 119],device='cuda',dtype=torch.int32)
    qlengths=lengths.clamp_max(21) if head_only else lengths
    ck=torch.nn.functional.pad(lengths.cumsum(0,dtype=torch.int32),(1,0))
    cq=torch.nn.functional.pad(qlengths.cumsum(0,dtype=torch.int32),(1,0))
    x=torch.randn(int(ck[-1]),3,8,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=x.detach().clone().requires_grad_(True)
    q,k,v=y.unbind(1)
    if head_only:
        indices=torch.cat([torch.arange(int(a),int(a+b),device='cuda') for a,b in zip(ck[:-1],qlengths)])
        q=q[indices]
    actual=packed_attention(x,cq,ck,int(cq[-1]),21 if head_only else 119)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected=varlen_attn(q,k,v,cq,ck,21 if head_only else 119,119)
    torch.testing.assert_close(actual,expected,atol=.016,rtol=.02)
    grad=torch.randn_like(actual)
    actual.backward(grad);expected.backward(grad)
    assert float((x.grad.float()-y.grad.float()).norm()/y.grad.float().norm()) < .015
