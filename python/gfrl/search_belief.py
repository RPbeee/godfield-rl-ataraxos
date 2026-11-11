"""Search-only transforms; T=1/P=0 return the original model unchanged."""
import math
from .belief import Belief, IndependentBelief


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


def validate_variants(temperature=1., mix=0.):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('belief-temperature must be finite and positive')
    if not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError('belief-mix must be in [0,1]')


def wrap_belief(model, temperature=1., mix=0., baseline=None):
    validate_variants(temperature, mix)
    if (temperature != 1 or mix) and (not isinstance(model, Belief) or isinstance(model, IndependentBelief)):
        raise ValueError('belief variants require an autoregressive checkpoint')
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
    for name, default in (('belief-temperature', 1.), ('belief-mix', 0.)):
        parser.add_argument('--'+name, type=float, default=argparse.SUPPRESS,
                            **({'nargs': '+'} if sweep else {}),
                            help=f'search-only AR transform (default {default})')
    if offline:
        parser.add_argument('--diagnostic-belief', default=argparse.SUPPRESS,
                            help='AR checkpoint for diagnostics when --belief is b1/uniform/oracle')
        parser.add_argument('--diagnostic-true-hand-nll', action='store_true', default=argparse.SUPPRESS,
                            help='OFFLINE ONLY: score privileged true unused hands; requires AR checkpoint and --fit')


def variant_options(args):
    return dict(temperature=getattr(args, 'belief_temperature', 1.),
                mix=getattr(args, 'belief_mix', 0.))
