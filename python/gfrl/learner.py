"""Decision-level PPO with exact masked KL on sampled stage prefixes."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .actor import PREFIX_FIELDS, synchronize
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
    magnet_source: str = 'uniform'
    alpha_schedule: str = 'decay'
    reg_update_every: int = 100
    reg_source: str = 'online'
    filtering: bool = False
    kl_coef: float = 0.1
    clip: float = 0.2
    gamma: float = 1.0
    lambda_v: float = 0.8
    lambda_a: float = 0.5
    ema: float = 0.999
    minibatch_size: int = 4096
    microbatch_size: int = 0  # automatic: CPU 32, CUDA full minibatch
    learner_backend: str = 'eager'  # CUDA packed-* execution; see docs/gpu_learner.md
    critic: str = 'shared'
    critic_lr: float = 1e-4
    critic_d: int = 0  # 0 inherits policy width/FF
    critic_depth: int = 0  # 0 inherits policy depth


def schedules(config, iteration):
    if iteration < 1:
        raise ValueError('iteration is one-based')
    lr = (float(np.clip(config.lr_c / iteration**1.1, config.lr_min, config.lr_max))
          if config.lr_decay else config.lr)
    alpha = (config.alpha0 if config.alpha_schedule == 'constant' else
             config.alpha0 / iteration**0.3) if config.magnet else 0.0
    return lr, alpha


def decision_terms(new_lp, old_lp, mask, choices, owners, count, reg_lp=None):
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
    magnet_kl = (uniform_kl if reg_lp is None else
                 (p * (new - reg_lp.detach().masked_fill(~mask, 0))).sum(-1))
    def reduce(values):
        return values.new_zeros(count).index_add(0, owners, values)
    log_new, log_old = reduce(selected_new), reduce(selected_old)
    return dict(log_prob=log_new, old_log_prob=log_old,
                ratio=(log_new-log_old).exp(), kl=reduce(kl),
                magnet=reduce(magnet_kl), entropy=reduce(entropy))


class Learner:
    def __init__(self, model, config=None, device='cpu', dtype='fp32'):
        self.config = config or LearnerConfig()
        if self.config.minibatch_size <= 0 or self.config.microbatch_size < 0:
            raise ValueError('batch sizes must be positive')
        if not 0 < self.config.gamma <= 1:
            raise ValueError('gamma must be in (0,1]')
        for name, choices in (('magnet_source', ('uniform', 'reg')),
                              ('alpha_schedule', ('decay', 'constant')),
                              ('reg_source', ('online', 'ema'))):
            if getattr(self.config, name) not in choices:
                raise ValueError(f'invalid {name}')
        if self.config.reg_update_every <= 0:
            raise ValueError('reg_update_every must be positive')
        if self.config.critic not in ('shared', 'privileged'):
            raise ValueError('critic must be shared or privileged')
        if (not np.isfinite(self.config.critic_lr) or self.config.critic_lr <= 0
                or self.config.critic_d < 0 or self.config.critic_depth < 0):
            raise ValueError('invalid critic learning rate or dimensions')
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.to(self.device)
        self.ema = deepcopy(model).eval().requires_grad_(False)
        self.reg = (deepcopy(model).eval().requires_grad_(False)
                    if self.config.magnet_source == 'reg' else None)
        self.reg_last_update = 0
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        self.optimizer_steps = 0
        self.critic = None
        if self.config.critic == 'privileged':
            from .critic import create_critic
            self.critic = create_critic(self.model, self.config)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
            self.critic_optimizer_steps = 0
            self._critic_forward = self.critic
        if self.config.learner_backend != 'eager':
            from .fast_learner import configure_fast_learner
            configure_fast_learner(self, self.config.learner_backend)

    def restore_critic(self, state, *, resume):
        if self.critic is None:
            return
        if state.get('critic') is not None:
            from dataclasses import asdict
            if state['critic_model_config'] != asdict(self.critic.config):
                if resume:
                    raise ValueError('critic model configuration differs from checkpoint')
                self.critic.initialize_from_policy(self.model)
                return
            self.critic.load_state_dict(state['critic'])
            self.critic_optimizer.load_state_dict(state['critic_optimizer'])
            self.critic_optimizer_steps = state['critic_optimizer_steps']
        elif resume:
            raise ValueError('privileged resume requires a saved critic; use --init-from')
        else:
            self.critic.initialize_from_policy(self.model)

    def _critic_inputs(self, batch, rows):
        from .critic import PRIVILEGED_FIELDS
        arrays = {k: batch.columns[k][rows].to(self.device) for k in (*OBS_FIELDS, *PRIVILEGED_FIELDS)}
        return ({k: arrays[k] for k in OBS_FIELDS}, *(arrays[k] for k in PRIVILEGED_FIELDS))

    @torch.no_grad()
    def _critic_baseline(self, batch):
        from .rollout import RolloutBatch
        from .runtime import encoder_fastpath_disabled
        from .critic import PRIVILEGED_FIELDS
        if not hasattr(batch, 'columns') or any(k not in batch.columns for k in PRIVILEGED_FIELDS):
            raise ValueError('privileged critic requires training rollout with unused hand labels')
        self.critic.eval()
        micro = self.config.microbatch_size or (self.config.minibatch_size if self.device.type == 'cuda' else 32)
        values = []
        with self.autocast(), encoder_fastpath_disabled():
            for start in range(0, len(batch), micro):
                rows = torch.arange(start, min(start+micro, len(batch)), device=batch.env.device)
                values.append(self._critic_forward(*self._critic_inputs(batch, rows)).float().softmax(-1))
        probs = torch.cat(values).to(batch.env.device)
        return RolloutBatch(dict(batch.columns, old_value=probs), batch.stages,
                            cpu_storage=batch.cpu_storage)

    def _update_critic(self, batch, targets, order):
        cfg = self.config
        for group in self.critic_optimizer.param_groups:
            group['lr'] = cfg.critic_lr
            if hasattr(self, '_packed_forward'):
                group.update(fused=True, foreach=False)
        if hasattr(self, '_packed_forward'):
            for state in self.critic_optimizer.state.values():
                if 'step' in state:
                    state['step'] = state['step'].to(self.device)
        loss_sum = torch.zeros((), device=self.device)
        for start in range(0, len(order), cfg.minibatch_size):
            rows = order[start:start+cfg.minibatch_size]
            self.critic_optimizer.zero_grad(set_to_none=True)
            micro = cfg.microbatch_size or (len(rows) if self.device.type == 'cuda' else 32)
            for offset in range(0, len(rows), micro):
                selected = rows[offset:offset+micro]
                with self.autocast():
                    logits = self._critic_forward(*self._critic_inputs(batch, selected))
                    target = targets[selected].to(self.device)
                    ce = -(target*F.log_softmax(logits.float(), -1)).sum(-1)
                    loss = ce.sum()/len(rows)
                loss.backward()
                loss_sum += ce.detach().sum()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
            self.critic_optimizer.step()
            self.critic_optimizer_steps += 1
        return float(loss_sum.cpu())/len(batch)

    def reset_reg(self, iteration=0):
        """Initialize from the starting online weights, including on init-from."""
        if self.reg is not None:
            self.reg.load_state_dict(self.model.state_dict())
        self.reg_last_update = iteration

    def restore_reg(self, state, *, resume):
        if resume and self.reg is not None and state.get('reg') is not None:
            self.reg.load_state_dict(state['reg'])
            self.reg_last_update = state['reg_last_update']
        else:
            self.reset_reg(state['iteration'])

    def update_reg(self, iteration):
        """Refresh after the complete PPO/EMA update; cadence is relative to start."""
        if self.reg is None or iteration-self.reg_last_update < self.config.reg_update_every:
            return False
        source = self.model if self.config.reg_source == 'online' else self.ema
        self.reg.load_state_dict(source.state_dict())
        self.reg_last_update = iteration
        return True

    def autocast(self):
        return (nullcontext() if self.dtype == 'fp32' else torch.autocast(
            self.device.type, dtype={'fp16': torch.float16, 'bf16': torch.bfloat16}[self.dtype]))

    def _tensor(self, array):
        return torch.as_tensor(array, device=self.device)

    def _prepare(self, batch, rows, xi=None, advantages=None, keep=None):
        rows = torch.as_tensor(rows, device=batch.env.device, dtype=torch.long)
        stages, owners = batch.stage_rows(rows)
        arrays = {k: batch.columns[k][rows] for k in OBS_FIELDS}
        arrays.update({k: batch.stages[k][stages] for k in PREFIX_FIELDS})
        arrays.update(owners=owners, choice=batch.stages['choice'][stages].long(),
                      old_lp=batch.stages['old_log_probs'][stages].float())
        if xi is not None:
            arrays.update(xi=xi[rows], advantage=advantages[rows], selected=keep[rows].float())
        if batch.env.device != self.device:
            if self.device.type == 'cuda' and batch.env.device.type == 'cpu':
                # Advanced indexing returns pageable tensors. Pin only this
                # microbatch, and enqueue all copies on the current CUDA stream.
                arrays = {k: v.pin_memory() for k, v in arrays.items()}
                arrays = {k: v.to(self.device, non_blocking=True) for k, v in arrays.items()}
            else:
                arrays = {k: v.to(self.device) for k, v in arrays.items()}
        return arrays

    def forward(self, decisions):
        if hasattr(decisions, 'columns'):
            decisions = self._prepare(decisions, torch.arange(len(decisions), device=self.device))
        if isinstance(decisions, dict) and hasattr(self, '_packed_forward'):
            return self._packed_forward(decisions)
        if isinstance(decisions, dict):
            obs = {k: decisions[k] for k in OBS_FIELDS}
            prefix = {k: decisions[k] for k in PREFIX_FIELDS}
            owners, choices, old_lp = decisions['owners'], decisions['choice'], decisions['old_lp']
        else:
            # Compatibility for external callers constructing individual records.
            obs = {k: self._tensor(np.stack([d.obs[k] for d in decisions])) for k in OBS_FIELDS}
            stages = [s for d in decisions for s in d.stages]
            owners = self._tensor([i for i,d in enumerate(decisions) for _ in d.stages]).long()
            prefix = {k: self._tensor(np.stack([s.prefix[k] for s in stages])) for k in PREFIX_FIELDS}
            choices = self._tensor([s.choice for s in stages]).long()
            old_lp = self._tensor(np.stack([s.old_log_probs for s in stages]).astype(np.float32))
        with self.autocast():
            encoded = self.model.encode(obs)
            lp, _ = self.model.heads(encoded.index_select(0, owners), **prefix)
            cls = encoded[:, self.model.positions['cls']]
            value = self.model.value_head(cls.detach() if self.critic is not None else cls).float()
        reg_lp = None
        if self.reg is not None:
            # Avoid the eval-only fused Transformer path, so equal weights use
            # the same arithmetic as the differentiable online forward.
            from .runtime import encoder_fastpath_disabled
            with torch.no_grad(), self.autocast(), encoder_fastpath_disabled():
                reg_encoded = self.reg.encode(obs)
                reg_lp, _ = self.reg.heads(reg_encoded.index_select(0, owners), **prefix)
        terms = decision_terms(lp, old_lp, prefix['mask'].bool(), choices, owners, len(value), reg_lp)
        return terms, value

    def _tick(self):
        if hasattr(self, '_packed_forward'):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        synchronize(self.device)
        return perf_counter()

    def _elapsed(self, since, name, timings):
        now = self._tick()
        if hasattr(self, '_packed_forward'):
            self._timing_events.append((name, since, now))
        else:
            timings[name] += now-since

    def update(self, decisions, iteration):
        if not decisions:
            raise ValueError('no completed training decisions')
        cfg = self.config
        fast = hasattr(self, '_packed_forward')
        if fast:
            # Checkpoint loading restores the saved execution flags as well.
            for group in self.optimizer.param_groups:
                group.update(fused=True, foreach=False)
            for state in self.optimizer.state.values():
                if 'step' in state and state['step'].device != self.device:
                    state['step'] = state['step'].to(self.device)
        lr, alpha = schedules(cfg, iteration)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        timings = dict(transfer=0., forward=0., backward=0., targets=0., optimizer=0.)
        synchronize(self.device)
        update_begin = perf_counter()
        self._timing_events = []
        since = self._tick()
        target_batch = self._critic_baseline(decisions) if self.critic is not None else decisions
        if fast and hasattr(decisions, 'columns') and decisions.env.is_cuda:
            from .gpu.targets import compute_targets as gpu_targets
            xi, advantages = gpu_targets(target_batch, cfg.lambda_v, cfg.lambda_a, cfg.gamma)
        else:
            xi, advantages = compute_targets(target_batch, cfg.lambda_v, cfg.lambda_a, cfg.gamma)
        critic_metrics = {}
        if self.critic is not None:
            from .critic import prediction_metrics
            critic_metrics = prediction_metrics(target_batch.old_value, decisions.old_value, decisions.outcome,
                torch.as_tensor(advantages, device=decisions.env.device))
            critic_metrics = dict(zip(critic_metrics, torch.stack(list(critic_metrics.values())).cpu().tolist()))
            del target_batch
        self._elapsed(since, 'targets', timings)
        if torch.is_tensor(advantages):
            keep = ((advantages.abs() >= advantages.abs().quantile(.75).clamp_min(.01))
                    if cfg.filtering else torch.ones_like(advantages, dtype=torch.bool))
        else:
            keep = (np.abs(advantages) >= max(float(np.quantile(np.abs(advantages), .75)), .01)
                    if cfg.filtering else np.ones(len(decisions), dtype=bool))
        metrics = dict(policy_loss=0., kl=0., kl_loss=0., magnet_loss=0., value_loss=0.,
                       entropy=0., ratio_mean=0., clip_fraction=0., value_accuracy=0.)
        if self.reg is not None:
            metrics['reg_kl'] = 0.
        metric_sums = []
        # Dropout remains off to match the behavior policies (gradients stay enabled).
        self.model.eval()
        order = np.random.permutation(len(decisions))
        columnar = hasattr(decisions, 'columns')
        cpu_storage = columnar and getattr(decisions, 'cpu_storage', False)
        if cpu_storage:
            order = torch.as_tensor(order, device=decisions.env.device)
            xi, advantages, keep = (torch.as_tensor(v, device=decisions.env.device)
                                    for v in (xi, advantages, keep))
        elif columnar:
            # Preserve the numpy permutation/RNG while transferring only 1-D indices.
            order = self._tensor(order)
            xi, advantages, keep = self._tensor(xi), self._tensor(advantages), self._tensor(keep)
        for start in range(0, len(order), cfg.minibatch_size):
            batch = order[start:start+cfg.minibatch_size]
            policy_count = (len(batch) if fast and not cfg.filtering else
                            max(int(keep[batch].sum()), 1))
            self.optimizer.zero_grad(set_to_none=True)
            packed = None
            if columnar and not cpu_storage:
                since = self._tick()
                packed = self._prepare(decisions, batch, xi, advantages, keep)
                self._elapsed(since, 'transfer', timings)
                stage_offsets = torch.cat((batch.new_zeros(1), decisions.stage_count[batch].cumsum(0)))
            microbatch = cfg.microbatch_size or (len(batch) if self.device.type == 'cuda' else 32)
            # Accumulate microbatches for bounded activation memory; one Adam step per minibatch.
            for offset in range(0, len(batch), microbatch):
                rows = batch[offset:offset+microbatch]
                if cpu_storage:
                    since = self._tick()
                    inputs = self._prepare(decisions, rows, xi, advantages, keep)
                    self._elapsed(since, 'transfer', timings)
                since = self._tick()
                if cpu_storage:
                    terms, value = self.forward(inputs)
                    advantage, selected, target = inputs['advantage'], inputs['selected'], inputs['xi']
                elif packed is not None:
                    end = offset+len(rows)
                    if offset == 0 and end == len(batch):
                        inputs = packed
                    else:
                        stage_slice = slice(int(stage_offsets[offset]), int(stage_offsets[end]))
                        inputs = {k: v[offset:end] if k in (*OBS_FIELDS, 'xi', 'advantage', 'selected')
                                  else v[stage_slice] for k,v in packed.items()}
                        inputs['owners'] = inputs['owners']-offset
                    terms, value = self.forward(inputs)
                    advantage, selected, target = inputs['advantage'], inputs['selected'], inputs['xi']
                else:
                    terms, value = self.forward([decisions[i] for i in rows])
                    advantage, selected = self._tensor(advantages[rows]), self._tensor(keep[rows]).float()
                    target = self._tensor(xi[rows])
                ratio = terms['ratio']
                policy = -torch.minimum(ratio*advantage, ratio.clamp(1-cfg.clip, 1+cfg.clip)*advantage)
                ce = -(target*F.log_softmax(value, -1)).sum(-1)
                total_policy = policy + cfg.kl_coef*terms['kl'] + alpha*terms['magnet']
                loss = (total_policy*selected).sum()/policy_count + ce.sum()/len(batch)
                self._elapsed(since, 'forward', timings)
                since = self._tick()
                loss.backward()
                self._elapsed(since, 'backward', timings)
                values = dict(policy_loss=policy*selected, kl=terms['kl'], kl_loss=cfg.kl_coef*terms['kl']*selected,
                              magnet_loss=alpha*terms['magnet']*selected, value_loss=ce,
                              entropy=terms['entropy'], ratio_mean=ratio,
                              clip_fraction=((ratio-1).abs() > cfg.clip).float(),
                              value_accuracy=(value.argmax(-1) == target.argmax(-1)).float())
                if self.reg is not None:
                    values['reg_kl'] = terms['magnet']
                if hasattr(self, '_packed_forward'):
                    metric_sums.append(torch.stack([v.detach().sum() for v in values.values()]))
                else:
                    for key, val in values.items():
                        metrics[key] += float(val.detach().sum().cpu())
            since = self._tick()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            decay = min(cfg.ema, (1 + self.optimizer_steps) / (10 + self.optimizer_steps))
            self.optimizer_steps += 1
            with torch.no_grad():
                if hasattr(self, '_packed_forward'):
                    torch._foreach_lerp_(list(self.ema.parameters()), list(self.model.parameters()), 1-decay)
                else:
                    for ema, current in zip(self.ema.parameters(), self.model.parameters()):
                        ema.lerp_(current, 1-decay)
                for ema, current in zip(self.ema.buffers(), self.model.buffers()):
                    ema.copy_(current)
            self._elapsed(since, 'optimizer', timings)
        if metric_sums:
            # Preserve the original microbatch sum and Python double accumulation.
            for sums in torch.stack(metric_sums).cpu().tolist():
                for key, value in zip(metrics, sums):
                    metrics[key] += value
        if self.critic is not None:
            since = self._tick()
            timings['critic'] = 0.
            metrics['critic_loss'] = self._update_critic(decisions, xi, order)*len(decisions)
            self._elapsed(since, 'critic', timings)
        for key in metrics:
            denom = max(int(keep.sum()), 1) if key in ('policy_loss', 'kl_loss', 'magnet_loss') else len(decisions)
            metrics[key] /= denom
        metrics.update(lr=lr, alpha=alpha, filtering_fraction=float(keep.float().mean())
                       if columnar else float(keep.mean()))
        if self.critic is not None:
            metrics.update(critic_metrics, critic_lr=cfg.critic_lr)
        else:
            metrics['advantage_std'] = float(torch.as_tensor(advantages).float().std(correction=0))
        if self.reg is not None:
            metrics.update(reg_updated=int(self.update_reg(iteration)),
                           reg_last_update=self.reg_last_update)
        synchronize(self.device)
        for name, begin, end in self._timing_events:
            timings[name] += begin.elapsed_time(end)/1000
        self._timing_events = []
        timings['other'] = perf_counter()-update_begin-sum(timings.values())
        metrics.update({'learn_'+k+'_s': v for k,v in timings.items()})
        return metrics

    def pool_entry(self, iteration):
        return dict(iteration=iteration, weights={k: v.detach().cpu().clone()
                                                for k, v in self.ema.state_dict().items()})
