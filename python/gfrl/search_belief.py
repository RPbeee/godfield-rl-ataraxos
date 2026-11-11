"""Search-only transforms; T=1/P=0 return the original model unchanged."""
import math
from .belief import Belief, IndependentBelief
import torch
from .belief_metrics import multiset_log_prob


class TemperedBelief:
    # Reuse the existing AR factorization/teacher forcing, changing only its
    # normalized conditional distributions. No model parameters are modified.
    sample = Belief.sample
    log_prob = Belief.log_prob
    count_probs = Belief.count_probs

    def __init__(self, model, temperature):
        if isinstance(model, IndependentBelief):
            raise ValueError('temperature requires an autoregressive checkpoint')
        self.model, self.temperature = model, temperature

    def memory(self, obs):
        return self.model.memory(obs)

    def count_head(self, cls):
        return self.model.count_head(cls) / self.temperature

    def decode(self, *args):
        return (self.model.decode(*args) / self.temperature).log_softmax(-1)


class MixedBelief:
    def __init__(self, ar, baseline, probability):
        self.ar, self.baseline, self.probability = ar, baseline, probability

    def choose(self, rng, size):
        return rng.random(size) < self.probability


class HybridBelief:
    """AR unused count and B1 conditional multinomial composition (public only)."""
    def __init__(self, ar, baseline, count_temperature=1.):
        self.ar, self.baseline = ar, baseline
        self.count_temperature = count_temperature

    def count_log_probs(self, obs):
        cls, _, _ = self.ar.memory(obs)
        return (self.ar.count_head(cls).float() / self.count_temperature).log_softmax(-1)

    def count_probs(self, obs):
        return self.count_log_probs(obs).exp()

    def log_prob(self, obs, labels):
        counts = labels.long().sum(1)
        count_part = self.count_log_probs(obs).gather(1, counts[:, None]).squeeze(1)
        # A unit count prior extracts B1's conditional composition, even for
        # sizes with zero B1 prior mass. Match B1's impossible-card floor.
        card_part = multiset_log_prob(labels.cpu(), torch.ones(19),
                                      self.baseline.card_probs).to(count_part.device)
        return count_part + card_part, count_part, card_part


def validate_variants(temperature=1., mix=0., hybrid=False, count_temperature=1.):
    if hybrid not in (False, True):
        raise ValueError('belief-hybrid must be 0 or 1')
    if not math.isfinite(count_temperature) or count_temperature <= 0:
        raise ValueError('belief-count-temperature must be finite and positive')
    if count_temperature != 1 and not hybrid:
        raise ValueError('belief-count-temperature requires belief-hybrid')
    if hybrid and (temperature != 1 or mix):
        raise ValueError('belief-hybrid cannot combine with belief-temperature or belief-mix')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('belief-temperature must be finite and positive')
    if not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError('belief-mix must be in [0,1]')


def wrap_belief(model, temperature=1., mix=0., baseline=None, *, hybrid=False, count_temperature=1.):
    validate_variants(temperature, mix, hybrid, count_temperature)
    if (temperature != 1 or mix or hybrid) and (not isinstance(model, Belief) or isinstance(model, IndependentBelief)):
        raise ValueError('belief variants require an autoregressive checkpoint')
    if hybrid:
        if baseline is None:
            raise ValueError('--fit is required for belief-hybrid')
        return HybridBelief(model, baseline, count_temperature)
    if temperature != 1:
        model = TemperedBelief(model, temperature)
    if mix:
        if baseline is None:
            raise ValueError('--fit is required for belief-mix')
        model = MixedBelief(model, baseline, mix)
    return model


def add_belief_arguments(parser, *, sweep=False, offline=False):
    import argparse
    # Absent options must not add fields to legacy evaluation JSON/configs.
    for name, default in (('belief-temperature', 1.), ('belief-mix', 0.), ('belief-count-temperature', 1.)):
        parser.add_argument('--'+name, type=float, default=argparse.SUPPRESS,
                            **({'nargs': '+'} if sweep else {}),
                            help=f'search-only AR transform (default {default})')
    parser.add_argument('--belief-hybrid', type=int, choices=(0, 1),
                        nargs='+' if sweep else '?', const=None if sweep else 1,
                        default=argparse.SUPPRESS, help='AR count + B1 composition (default off)')
    if offline:
        parser.add_argument('--diagnostic-belief', default=argparse.SUPPRESS,
                            help='AR checkpoint for diagnostics when --belief is b1/uniform/oracle')
        parser.add_argument('--diagnostic-true-hand-nll', action='store_true', default=argparse.SUPPRESS,
                            help='OFFLINE ONLY: score privileged true unused hands; requires AR checkpoint and --fit')


def variant_options(args):
    return dict(temperature=getattr(args, 'belief_temperature', 1.),
                mix=getattr(args, 'belief_mix', 0.),
                hybrid=getattr(args, 'belief_hybrid', False),
                count_temperature=getattr(args, 'belief_count_temperature', 1.))
