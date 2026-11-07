"""Multiset likelihood baselines and model-independent belief metrics (nats)."""
from dataclasses import dataclass
import numpy as np
import torch
from . import _core


def gift_probabilities():
    """Catalog giftRate/100, indexed by card_code; row 0 is absent/padding."""
    column = _core.obs_schema()['card_features']['gift_rate']['index']
    rates = torch.tensor(_core.card_features()[:, column], dtype=torch.float64)
    return rates / rates.sum()


def validate_labels(labels, counts):
    if labels.ndim != 2 or labels.shape[1] != 297 or counts.shape != labels.shape[:1]:
        raise ValueError('expected labels [B,297] and counts [B]')
    if (labels < 0).any() or (labels != labels.floor()).any() or labels[:, 0].any():
        raise ValueError('invalid card counts (card code 0 is padding)')
    if not torch.equal(labels.sum(1).long(), counts.long()) or (counts > 18).any() or (counts < 0).any():
        raise ValueError('hand counts must equal label sums and lie in 0..18')


def multiset_log_prob(labels, count_probs, card_probs, floor=1e-12):
    """Per-position log probability; zero-probability observed cards use floor.

    Raw distributions remain unchanged, including exact giftRate zeros. The
    evaluator separately reports impossible observations before log clamping.
    Supports any number of card types for enumeration/unit tests.
    """
    c = torch.as_tensor(labels, dtype=torch.float64)
    pn = torch.as_tensor(count_probs, dtype=torch.float64)
    pk = torch.as_tensor(card_probs, dtype=torch.float64)
    n = c.sum(-1).long()
    return (pn[n].log() + torch.lgamma(n.double() + 1)
            - torch.lgamma(c + 1).sum(-1) + (c * pk.clamp_min(floor).log()).sum(-1))


@dataclass
class Baseline:
    count_probs: torch.Tensor
    card_probs: torch.Tensor

    def log_prob(self, labels):
        return multiset_log_prob(labels, self.count_probs, self.card_probs)

    def presence_probs(self):
        n = torch.arange(len(self.count_probs), dtype=torch.float64)
        # 0**0=1 correctly accounts for empty hands, including p=1.
        return 1 - (self.count_probs[:, None] * (1-self.card_probs)[None, :].pow(n[:, None])).sum(0)


def fit_baselines(data):
    labels, counts = data['label_opp_hand'], data['label_opp_hand_count']
    if not len(counts):
        raise ValueError('empty fitting data')
    totals = torch.zeros(297, dtype=torch.float64)
    for start in range(0, len(counts), 8192):
        c, n = labels[start:start+8192], counts[start:start+8192]
        validate_labels(c, n)
        totals += c.sum(0)
    pn = torch.bincount(counts.long(), minlength=19).double() + 1
    pn /= pn.sum()
    # code 0 is absent, not an actual card in the catalog.
    pk = totals + 1
    pk[0] = 0
    pk /= pk.sum()
    return {'B0': Baseline(pn, gift_probabilities()), 'B1': Baseline(pn, pk)}


def calibration_metrics(probabilities, labels, bins=15):
    """ECE pooled over every position/card, with reliability diagram bins.

    labels may be multiset counts or binary presence. Empty bins use null means.
    """
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    y = torch.as_tensor(labels)
    if p.ndim != 2 or p.shape != y.shape or p.shape[1] != 297 or bins <= 0:
        raise ValueError('expected matching [B,297] arrays and positive bins')
    if not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('probabilities must be finite in [0,1]')
    index = (p.flatten() * bins).long().clamp_max(bins-1)
    count = torch.bincount(index, minlength=bins)
    ps = torch.bincount(index, weights=p.flatten(), minlength=bins)
    ys = torch.bincount(index, weights=(y.flatten() > 0).double(), minlength=bins)
    return _calibration_summary(count, ps, ys)


def _calibration_summary(count, ps, ys):
    rows = [dict(lower=i/len(count), upper=(i+1)/len(count), count=int(c),
                 mean_prediction=float(ps[i]/c) if c else None,
                 actual_fraction=float(ys[i]/c) if c else None) for i, c in enumerate(count)]
    return dict(ece=float((ps-ys).abs().sum()/count.sum()) if count.sum() else None,
                reliability=rows)


def likelihood_metrics(log_prob, decisions, boundaries=None):
    """Accept per-position true-multiset log probabilities from ANY model.

    Default boundaries are validation decision-number tertiles. Ties stay in
    one bucket, so small/discrete datasets can have empty buckets (null NLL).
    Pass fixed boundaries to compare datasets using identical phase splits.
    """
    lp = np.asarray(log_prob, dtype=np.float64)
    d = np.asarray(decisions)
    if lp.ndim != 1 or d.shape != lp.shape or not len(lp) or np.isnan(lp).any():
        raise ValueError('require nonempty matching per-position arrays without NaN')
    edges = np.asarray(boundaries if boundaries is not None else np.quantile(d, [1/3, 2/3]))
    if edges.shape != (2,) or edges[0] > edges[1]:
        raise ValueError('require two ordered boundaries')
    bucket = np.searchsorted(edges, d, side='left')
    return dict(multiset_nll=float(-lp.mean()), phase_boundaries=edges.tolist(),
                phase_rule='early: d<=b0; middle: b0<d<=b1; late: d>b1',
                phase_nll={name: dict(positions=int((bucket == i).sum()),
                                      nll=float(-lp[bucket == i].mean()) if (bucket == i).any() else None)
                           for i, name in enumerate(('early', 'middle', 'late'))})


def evaluate_baseline(baseline, data, batch_size=4096):
    labels, counts = data['label_opp_hand'], data['label_opp_hand_count']
    lp = np.empty(len(counts), dtype=np.float64)
    presence = baseline.presence_probs().clamp(0, 1)
    bin_count = torch.zeros(15, dtype=torch.int64)
    ps = torch.zeros(15, dtype=torch.float64)
    ys = torch.zeros(15, dtype=torch.float64)
    zero_cards = baseline.card_probs == 0
    impossible_positions = impossible_copies = 0
    for start in range(0, len(counts), batch_size):
        c = labels[start:start+batch_size]
        validate_labels(c, counts[start:start+batch_size])
        lp[start:start+len(c)] = baseline.log_prob(c).numpy()
        impossible_positions += int((c[:, zero_cards].sum(1) > 0).sum())
        impossible_copies += int(c[:, zero_cards].sum())
        bins = (presence * 15).long().clamp_max(14).expand(len(c), -1).flatten()
        bin_count += torch.bincount(bins, minlength=15)
        ps += torch.bincount(bins, weights=presence.expand(len(c), -1).flatten(), minlength=15)
        ys += torch.bincount(bins, weights=(c.flatten() > 0).double(), minlength=15)
    result = likelihood_metrics(lp, data['decision'])
    result.update(count_nll=float(-baseline.count_probs[counts.long()].log().mean()),
                  count_accuracy=float((counts == baseline.count_probs.argmax()).double().mean()),
                  zero_probability_positions=impossible_positions,
                  zero_probability_card_copies=impossible_copies, log_probability_floor=1e-12,
                  **_calibration_summary(bin_count, ps, ys))
    return result
