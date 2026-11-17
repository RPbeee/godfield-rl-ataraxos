"""Training-only asymmetric value network. Never attached to a policy Model.

The public encoder has the policy's shape. A residual MLP conditions its W/L/D
logits on the true unused opponent-hand multiset and count. No future RNG,
opponent private history, or fog-hidden resources are exposed.
"""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .model import Model, OBS_FIELDS

PRIVILEGED_FIELDS = ('label_opp_unused_hand', 'label_opp_unused_hand_count')


class PrivilegedCritic(Model):
    def __init__(self, config):
        super().__init__(config)
        for name in ('kind_embedding', 'stage_embedding', 'target_embedding', 'hp_embedding',
                     'query', 'kind_head', 'card_query', 'stop_head', 'target_head',
                     'hp_head', 'mp_head'):
            delattr(self, name)
        d = config.d
        self.hand_projection = nn.Linear(297, d, bias=False)
        self.condition = nn.Sequential(nn.Linear(2*d+1, d), nn.GELU(), nn.Linear(d, 3))
        # A copied policy starts with its existing baseline, learning a hidden
        # hand correction immediately without perturbing the initial prediction.
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def initialize_from_policy(self, policy):
        if self.config == policy.config:
            weights = {k: v for k, v in policy.state_dict().items() if k in self.state_dict()}
            self.load_state_dict(weights, strict=False)

    def values(self, cls, hand, count):
        hidden = F.gelu(self.hand_projection(hand.float()))
        correction = self.condition(torch.cat((cls, hidden, count.float()[:, None]/18), -1))
        return self.value_head(cls).float() + correction.float()

    def forward(self, public, hand, count):
        encoded = self.encode({k: public[k] for k in OBS_FIELDS})
        return self.values(encoded[:, self.positions['cls']], hand, count)


def create_critic(policy, config):
    cfg = replace(policy.config, d=config.critic_d or policy.config.d,
                  depth=config.critic_depth or policy.config.depth,
                  ff=4*(config.critic_d or policy.config.d))
    if not config.critic_d:
        cfg = replace(cfg, ff=policy.config.ff)
    if cfg.depth <= 0 or cfg.d % cfg.heads:
        raise ValueError('critic width must be divisible by policy heads; depth must be positive')
    critic = PrivilegedCritic(cfg).to(next(policy.parameters()).device).eval()
    critic.initialize_from_policy(policy)
    return critic


class PackedCriticForward:
    """Reuse differentiable packed kernels; final layer needs only CLS queries."""
    def __init__(self, learner):
        from .fast_learner import PackedForward
        backend = learner.config.learner_backend
        low = backend in ('packed-bf16', 'packed-grouped', 'packed-fp8')
        self.packed = PackedForward(learner, compile=backend != 'packed-eager',
            reduced='reduce' in backend or low, short='short' in backend or low,
            fp8=backend == 'packed-fp8', low_precision=low,
            grouped=backend in ('packed-grouped', 'packed-fp8'))
        self.values = learner.critic.values
        if backend != 'packed-eager':
            self.values = torch.compile(self.values, dynamic=True)
        self.learner = learner

    def __call__(self, obs, hand, count):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        model = self.learner.critic
        valid = obs['obs_valid'].bool()
        b, t = valid.shape
        positions = torch.where(valid.flatten())[0]
        lengths = valid.sum(1, dtype=torch.int32)
        cu = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        indices = cu[:-1].long()  # CLS is always the first valid token.
        hcu = torch.arange(b+1, dtype=torch.int32, device=cu.device)
        with self.learner.autocast(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = self.packed.embed(model, obs, positions)
            cls = self.packed.encode(model, x, cu, indices, hcu, t)
            return self.values(cls, hand, count)


def prediction_metrics(critic, shared, outcomes, advantages):
    """Frozen, pre-update predictions against identical terminal labels.

    EV uses scalar W-L and population variance. Constant outcomes yield finite
    zero EV; Brier remains informative. Draws/truncations have outcome zero.
    """
    outcome = outcomes.float()
    target = F.one_hot(torch.where(outcome == 1, 0, torch.where(outcome == -1, 1, 2)), 3).float()
    result = {'advantage_std': advantages.float().std(correction=0)}
    for name, probs in (('critic', critic), ('shared_value', shared)):
        scalar = probs[:, 0]-probs[:, 1]
        variance = outcome.var(correction=0)
        ev = 1-(outcome-scalar).var(correction=0)/variance.clamp_min(1e-12)
        result[name+'_explained_variance'] = torch.where(variance > 1e-12, ev, 0.)
        result[name+'_brier'] = (probs-target).square().sum(-1).mean()
        result[name+'_accuracy'] = (probs.argmax(-1) == target.argmax(-1)).float().mean()
    return result
