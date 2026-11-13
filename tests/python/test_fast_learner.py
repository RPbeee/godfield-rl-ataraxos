"""Structural equivalence of the packed learner's embedding and stage heads."""
from copy import deepcopy
import numpy as np
import torch

from gfrl.actor import PREFIX_FIELDS
from gfrl.fast_learner import embed, encode, heads
from gfrl.learner import decision_terms
from gfrl.model import Model, ModelConfig, OBS_FIELDS
from gfrl.rollout import Collector


def test_packed_embedding_and_heads_match_original_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    m = Model(ModelConfig(d=32, depth=1, heads=4, ff=128)).eval()
    batch = Collector(m, num_envs=8, pool_fraction=0, seed=9).collect(4)
    rows = torch.arange(min(8, len(batch)))
    stages, owners = batch.stage_rows(rows)
    obs = {k: batch.columns[k][rows] for k in OBS_FIELDS}
    positions = torch.where(obs['obs_valid'].flatten())[0]
    # Capture the pre-encoder embeddings in the reference implementation.
    captured = []
    handle = m.encoder.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    m.encode(obs, pack=False)
    handle.remove()
    expected = captured[0].flatten(0, 1)[positions]
    torch.testing.assert_close(embed(m, obs, positions), expected, rtol=0, atol=0)
    inputs = {k: batch.stages[k][stages] for k in PREFIX_FIELDS}
    inputs.update(owners=owners, choice=batch.stages['choice'][stages].long(),
                  old_lp=batch.stages['old_log_probs'][stages].float())
    a = torch.randn(len(rows), 119, 32, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    other = deepcopy(m)
    lp, _ = m.heads(a[owners], **{k: inputs[k] for k in PREFIX_FIELDS})
    reference = decision_terms(lp, inputs['old_lp'], inputs['mask'].bool(), inputs['choice'], owners, len(rows))
    actual, values = heads(other, b[:, :21], inputs)
    ref_value = m.value_head(a[:, 0])
    for k in reference:
        torch.testing.assert_close(actual[k], reference[k], rtol=0, atol=0)
    torch.testing.assert_close(values, ref_value, rtol=0, atol=0)
    (reference['log_prob'].sum()+reference['kl'].sum()+ref_value.square().sum()).backward()
    (actual['log_prob'].sum()+actual['kl'].sum()+values.square().sum()).backward()
    torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
    for p, q in zip(m.parameters(), other.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5)


def test_packed_encoder_full_keys_final_head_queries_and_gradients(monkeypatch):
    from torch.nn import functional as F
    import torch.nn.attention.varlen as varlen

    def attention(q, k, v, cq, ck, *args):
        outputs = []
        for qa, qb, ka, kb in zip(cq[:-1], cq[1:], ck[:-1], ck[1:]):
            out = F.scaled_dot_product_attention(q[qa:qb].transpose(0, 1),
                k[ka:kb].transpose(0, 1), v[ka:kb].transpose(0, 1))
            outputs.append(out.transpose(0, 1))
        return torch.cat(outputs)

    monkeypatch.setattr(varlen, 'varlen_attn', attention)
    torch.manual_seed(3)
    m = Model(ModelConfig(d=32, depth=3, heads=4, ff=128)).eval()
    other = deepcopy(m)
    valid = torch.rand(3, 55) > .5
    valid[:, 0] = True
    positions = torch.where(valid.flatten())[0]
    cu = F.pad(valid.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
    head = valid[:, :21]
    hcu = F.pad(head.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
    hidx = torch.where((positions % 55) < 21)[0]
    a = torch.randn(3, 55, 32, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    expected = m.encoder(a, src_key_padding_mask=~valid)[:, :21][head]
    actual = encode(other, b.flatten(0, 1)[positions], cu, hidx, hcu, 55)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    weight = torch.randn_like(actual)
    (actual*weight).sum().backward()
    (expected*weight).sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-4)
    for p, q in zip(m.parameters(), other.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
