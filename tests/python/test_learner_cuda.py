"""CUDA kernel checks against PyTorch autograd and the original CPU targets."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('head_only', [False, True])
@pytest.mark.parametrize('head_dim', [32, 48])
def test_short_attention_forward_backward(head_only, head_dim):
    from gfrl.gpu.attention_training import short_attention
    from torch.nn.attention.varlen import varlen_attn
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(1)
    lengths = torch.tensor([1, 5, 21, 53, 119], device='cuda', dtype=torch.int32)
    qlengths = lengths.clamp_max(21) if head_only else lengths
    ck = torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    cq = torch.nn.functional.pad(qlengths.cumsum(0, dtype=torch.int32), (1, 0))
    tensors = [torch.randn(int(total), 8, head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
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
@pytest.mark.parametrize('head_dim', [32, 48])
def test_grouped_attention_interleaved_gradient(head_only, head_dim):
    from gfrl.gpu.attention_packed_training import packed_attention
    from torch.nn.attention.varlen import varlen_attn
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(4)
    lengths=torch.tensor([1, 5, 21, 31, 32, 33, 53, 64, 65, 119],device='cuda',dtype=torch.int32)
    qlengths=lengths.clamp_max(21) if head_only else lengths
    ck=torch.nn.functional.pad(lengths.cumsum(0,dtype=torch.int32),(1,0))
    cq=torch.nn.functional.pad(qlengths.cumsum(0,dtype=torch.int32),(1,0))
    x=torch.randn(int(ck[-1]),3,8,head_dim,device='cuda',dtype=torch.bfloat16,requires_grad=True)
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


def test_moving_magnet_packed_grouped_matches_eager():
    from copy import deepcopy
    from gfrl.learner import Learner, LearnerConfig
    from gfrl.model import Model, ModelConfig
    from gfrl.rollout import Collector, RolloutBatch
    from gfrl.runtime import configure_precision
    configure_precision('cuda', False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = Model(ModelConfig(d=256, heads=8, depth=2, ff=1024)).eval()
    batch = Collector(model, num_envs=16, threads=2, seed=7, pool_fraction=0).collect(2)[:64]
    batch = RolloutBatch({k:v.cuda() for k,v in batch.columns.items()},
                         {k:v.cuda() for k,v in batch.stages.items()})
    cfg = LearnerConfig(magnet=True, magnet_source='reg', alpha_schedule='constant',
                        alpha0=.5, reg_update_every=2, minibatch_size=64, microbatch_size=64)
    eager = Learner(deepcopy(model), deepcopy(cfg), 'cuda', 'bf16')
    cfg.learner_backend = 'packed-grouped'
    fast = Learner(deepcopy(model), cfg, 'cuda', 'bf16')
    inputs = eager._prepare(batch, torch.arange(len(batch)))
    # bf16 compiled grad/no-grad graphs can fuse/round differently. The zero
    # KL tolerance accounts for that bf16 rounding; CPU fp32 is tested at 1e-6.
    for learner in (eager, fast):
        terms, _ = learner.forward(inputs)
        torch.testing.assert_close(terms['magnet'], torch.zeros_like(terms['magnet']), atol=1e-4, rtol=0,
                                   msg=learner.config.learner_backend)
    with torch.no_grad():
        for p in eager.reg.parameters():
            p.add_(torch.randn_like(p)*.002)
    fast.reg.load_state_dict(eager.reg.state_dict())
    initial = torch.cat([p.detach().flatten() for p in eager.model.parameters()])
    results = []
    for learner in (eager, fast):
        terms, values = learner.forward(inputs)
        loss = terms['log_prob'].mean()+terms['kl'].mean()*.1+terms['magnet'].mean()*.5+values.square().mean()
        loss.backward()
        gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                              for p in learner.model.parameters()])
        results.append((terms, values, gradient))
        assert float(terms['magnet'].detach().mean()) > 1e-5
        assert all(p.grad is None for p in learner.reg.parameters())
    a, b = results
    for key in b[0]:
        torch.testing.assert_close(a[0][key], b[0][key], atol=.025, rtol=.02)
    assert float(((a[1]-b[1]).norm()/a[1].norm()).detach()) < .01
    assert float((a[2]-b[2]).norm()/a[2].norm()) < .06
    for iteration in (1, 2, 3):
        for learner in (eager, fast):
            np.random.seed(123+iteration)
            metrics = learner.update(batch, iteration)
            assert metrics['reg_updated'] == (iteration == 2)
            assert np.isfinite(metrics['reg_kl'])
            if iteration == 2:
                terms, _ = learner.forward(inputs)
                torch.testing.assert_close(terms['magnet'], torch.zeros_like(terms['magnet']), atol=1e-4, rtol=0,
                                           msg=learner.config.learner_backend)
    updates = [torch.cat([p.detach().flatten() for p in l.model.parameters()])-initial for l in (eager, fast)]
    assert float((updates[0]-updates[1]).norm()/updates[0].norm()) < .15
    # Fresh-run EMA uses ~0.1 decay, so its error tracks Adam's update error.
    # Compare cumulative updates; the production-checkpoint validation also
    # reports absolute EMA error with the mature 0.999 decay.
    emas = [torch.cat([p.detach().flatten() for p in l.ema.parameters()])-initial for l in (eager, fast)]
    assert float((emas[0]-emas[1]).norm()/emas[0].norm()) < .15
