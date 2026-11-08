"""B2 multinomial normalization, exact marginalization, and sampling."""
import pytest
import torch
from gfrl.belief import IndependentBelief, BeliefConfig
from gfrl.belief_metrics import Baseline
from gfrl.belief_training import evaluate_belief
from gfrl.model import ModelConfig
from test_belief_model import observation, single_thread


def make_model():
    return IndependentBelief(ModelConfig(d=16, depth=1, heads=4, ff=32),
                             BeliefConfig(heads=4)).eval()


def test_all_multisets():
    model = make_model()
    pn, pk = model._log_probs(observation())
    pn, pk = pn.detach(), pk.detach()
    # Cache the same observation encoding while enumerating all 43,956 pairs.
    model._log_probs = lambda obs: (pn.expand(obs['size'], -1), pk.expand(obs['size'], -1))
    for n in (1, 2):
        codes = torch.arange(1, 297)[:, None] if n == 1 else (torch.triu_indices(296, 296)+1).T
        labels = torch.zeros(len(codes), 297)
        labels.scatter_add_(1, codes, torch.ones_like(codes, dtype=torch.float32))
        total, count, cards = model.log_prob({'size': len(codes)}, labels)
        torch.testing.assert_close(total, count+cards)
        assert total.double().exp().sum().item() == pytest.approx(pn[0, n].exp().item(), rel=2e-6)


def test_constant_logits_b1_and_presence():
    model = make_model()
    with torch.no_grad():
        model.count_head.weight.zero_()
        model.card_head.weight.zero_()
        model.card_head.bias.fill_(-20)
        model.card_head.bias[:3] = torch.tensor([0., 1., 2.])
    obs = observation(3)
    pn, pk = model._log_probs(obs)
    baseline = Baseline(pn[0].double().exp(), torch.cat((torch.zeros(1), pk[0].double().exp())))
    labels = torch.zeros(3, 297)
    labels[0, 1] = 2; labels[0, 3] = 1
    labels[1, 2] = 1
    torch.testing.assert_close(model.log_prob(obs, labels)[0].double(), baseline.log_prob(labels), atol=2e-6, rtol=1e-6)
    exact = model.presence_probs(obs)
    torch.testing.assert_close(exact.double(), baseline.presence_probs().expand(3, -1), atol=1e-6, rtol=1e-5)
    samples = model.sample(obs, 20000, generator=torch.Generator().manual_seed(72))
    assert not samples[..., 0].any()
    assert samples.sum(-1).max() <= 18
    assert ((samples > 0).float().mean(1)-exact).abs().max() < .02
    for n in (0, -1):
        with pytest.raises(ValueError):
            model.sample(obs, n)


def test_evaluate_independent_does_not_sample():
    model = make_model()
    obs = observation(2)
    data = dict(obs, decision=torch.arange(2), label_opp_hand=torch.zeros(2, 297))
    model.sample = lambda *a, **k: pytest.fail('B2 evaluation must use exact presence')
    a = evaluate_belief(model, data, 'cpu', 1, 1, 1)
    b = evaluate_belief(model, data, 'cpu', 1, 256, 32)
    assert a == b


def test_autoregressive_presence_chunks_and_mode():
    from gfrl.belief import Belief
    model = Belief(ModelConfig(d=16, depth=1, heads=4, ff=32), BeliefConfig(heads=4))
    obs = observation(2)
    data = dict(obs, decision=torch.arange(2), label_opp_hand=torch.zeros(2, 297))
    calls = []
    def sample(obs, size):
        calls.append(size)
        result = torch.zeros(2, size, 297, dtype=torch.long)
        result[:, :, 1] = 1
        return result
    model.sample = sample
    chunked = evaluate_belief(model, data, 'cpu', 2, 5, 2)
    assert calls == [2, 2, 1]
    assert model.training
    calls.clear()
    unchunked = evaluate_belief(model, data, 'cpu', 2, 5)
    assert calls == [5]
    assert chunked == unchunked
