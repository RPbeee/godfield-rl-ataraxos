"""Decision-level PPO with exact masked KL on sampled stage prefixes."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .actor import PREFIX_FIELDS
from .model import OBS_FIELDS
from .targets import compute_targets


@dataclass
class LearnerConfig:
    lr: float = 1e-4
    lr_decay: bool = False
    lr_c: float = 0.5
    lr_min: float = 5e-6
    lr_max: float = 1e-4
    magnet: bool = False
    alpha0: float = 0.05
    filtering: bool = False
    kl_coef: float = 0.1
    clip: float = 0.2
    lambda_v: float = 0.8
    lambda_a: float = 0.5
    ema: float = 0.999
    minibatch_size: int = 4096
    microbatch_size: int = 32


def schedules(config, iteration):
    if iteration < 1:
        raise ValueError('iteration is one-based')
    lr = (float(np.clip(config.lr_c / iteration**1.1, config.lr_min, config.lr_max))
          if config.lr_decay else config.lr)
    alpha = config.alpha0 / iteration**0.3 if config.magnet else 0.0
    return lr, alpha


def decision_terms(new_lp, old_lp, mask, choices, owners, count):
    """Sum stage log probabilities, KLs and entropy for each decision."""
    # Explicitly zero masked entries before arithmetic, including -inf inputs.
    new = new_lp.masked_fill(~mask, 0)
    old = old_lp.masked_fill(~mask, 0)
    p = new_lp.exp().masked_fill(~mask, 0)
    selected_new = new_lp.gather(1, choices[:, None]).squeeze(1)
    selected_old = old_lp.gather(1, choices[:, None]).squeeze(1)
    kl = (p * (new - old)).sum(-1)
    entropy = -(p * new).sum(-1)
    uniform_kl = mask.sum(-1).float().log() - entropy
    def reduce(values):
        return values.new_zeros(count).index_add(0, owners, values)
    log_new, log_old = reduce(selected_new), reduce(selected_old)
    return dict(log_prob=log_new, old_log_prob=log_old,
                ratio=(log_new-log_old).exp(), kl=reduce(kl),
                magnet=reduce(uniform_kl), entropy=reduce(entropy))


class Learner:
    def __init__(self, model, config=None, device='cpu', dtype='fp32'):
        self.config = config or LearnerConfig()
        if min(self.config.minibatch_size, self.config.microbatch_size) <= 0:
            raise ValueError('batch sizes must be positive')
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.to(self.device)
        self.ema = deepcopy(model).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def autocast(self):
        return (nullcontext() if self.dtype == 'fp32' else torch.autocast(
            self.device.type, dtype={'fp16': torch.float16, 'bf16': torch.bfloat16}[self.dtype]))

    def _tensor(self, array):
        return torch.as_tensor(np.asarray(array), device=self.device)

    def forward(self, decisions):
        obs = {k: self._tensor(np.stack([d.obs[k] for d in decisions])) for k in OBS_FIELDS}
        stages = [s for d in decisions for s in d.stages]
        owners = self._tensor([i for i, d in enumerate(decisions) for _ in d.stages]).long()
        prefix = {k: self._tensor(np.stack([s.prefix[k] for s in stages])) for k in PREFIX_FIELDS}
        with self.autocast():
            encoded = self.model.encode(obs)
            lp, _ = self.model.heads(encoded.index_select(0, owners), **prefix)
            value = self.model.value_head(encoded[:, self.model.positions['cls']]).float()
        terms = decision_terms(lp, self._tensor(np.stack([s.old_log_probs for s in stages])),
                               prefix['mask'].bool(), self._tensor([s.choice for s in stages]).long(),
                               owners, len(decisions))
        return terms, value

    def update(self, decisions, iteration):
        if not decisions:
            raise ValueError('no completed training decisions')
        cfg = self.config
        lr, alpha = schedules(cfg, iteration)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        xi, advantages = compute_targets(decisions, cfg.lambda_v, cfg.lambda_a)
        keep = (np.abs(advantages) >= max(float(np.quantile(np.abs(advantages), .75)), .01)
                if cfg.filtering else np.ones(len(decisions), dtype=bool))
        metrics = dict(policy_loss=0., kl_loss=0., magnet_loss=0., value_loss=0.,
                       entropy=0., ratio_mean=0., clip_fraction=0., value_accuracy=0.)
        # Dropout remains off to match the behavior policies (gradients stay enabled).
        self.model.eval()
        order = np.random.permutation(len(decisions))
        for start in range(0, len(order), cfg.minibatch_size):
            batch = order[start:start+cfg.minibatch_size]
            policy_count = max(int(keep[batch].sum()), 1)
            self.optimizer.zero_grad(set_to_none=True)
            # Accumulate microbatches for bounded activation memory; one Adam step per minibatch.
            for offset in range(0, len(batch), cfg.microbatch_size):
                rows = batch[offset:offset+cfg.microbatch_size]
                terms, value = self.forward([decisions[i] for i in rows])
                advantage = self._tensor(advantages[rows])
                selected = self._tensor(keep[rows]).float()
                ratio = terms['ratio']
                policy = -torch.minimum(ratio*advantage, ratio.clamp(1-cfg.clip, 1+cfg.clip)*advantage)
                ce = -(self._tensor(xi[rows])*F.log_softmax(value, -1)).sum(-1)
                total_policy = policy + cfg.kl_coef*terms['kl'] + alpha*terms['magnet']
                loss = (total_policy*selected).sum()/policy_count + ce.sum()/len(batch)
                loss.backward()
                values = dict(policy_loss=policy*selected, kl_loss=cfg.kl_coef*terms['kl']*selected,
                              magnet_loss=alpha*terms['magnet']*selected, value_loss=ce,
                              entropy=terms['entropy'], ratio_mean=ratio,
                              clip_fraction=((ratio-1).abs() > cfg.clip).float(),
                              value_accuracy=(value.argmax(-1) == self._tensor(xi[rows]).argmax(-1)).float())
                for key, val in values.items():
                    metrics[key] += float(val.detach().sum().cpu())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            with torch.no_grad():
                for ema, current in zip(self.ema.parameters(), self.model.parameters()):
                    ema.lerp_(current, 1-cfg.ema)
                for ema, current in zip(self.ema.buffers(), self.model.buffers()):
                    ema.copy_(current)
        for key in metrics:
            denom = max(int(keep.sum()), 1) if key in ('policy_loss', 'kl_loss', 'magnet_loss') else len(decisions)
            metrics[key] /= denom
        metrics.update(lr=lr, alpha=alpha, filtering_fraction=float(keep.mean()))
        return metrics

    def pool_entry(self, iteration):
        return dict(iteration=iteration, weights={k: v.detach().cpu().clone()
                                                for k, v in self.ema.state_dict().items()})
