"""Online fixed-policy collection, device replay, and validation for belief."""
import numpy as np
import torch
from . import _core
from .actor import Actor
from .evaluation import list_pool_iterations, load_policy_model
from .model import OBS_FIELDS
from .belief_metrics import likelihood_metrics, calibration_metrics, _calibration_summary

FIELDS = (*OBS_FIELDS, 'label_opp_hand', 'label_opp_hand_count')


class PositionStream:
    """Same episode routing as belief_data: latest seat 0, 80:20 seat 1."""
    def __init__(self, checkpoint, device='cpu', num_envs=64, threads=1, seed=0,
                 sample_prob=.25, pool_fraction=.2):
        self.env = _core.VecEnv(num_envs, threads, seed, controllers=('external', 'external'))
        specs = [str(checkpoint)] + [f'{checkpoint}@pool:{n}' for n in list_pool_iterations(checkpoint)]
        if pool_fraction and len(specs) < 2:
            raise ValueError('self with pool_fraction > 0 requires a saved pool')
        self.policies = [Actor(load_policy_model(s, device), self.env, device) for s in specs]
        self.sampling, self.routing = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        self.previous = np.full(num_envs, -1, dtype=np.int64)
        self.route = np.zeros(num_envs, dtype=np.int32)
        self.sample_prob, self.pool_fraction = sample_prob, pool_fraction

    def next(self):
        env = self.env
        games = env.episode_id.astype(np.int64)*env.num_envs + np.arange(env.num_envs)
        for r in np.flatnonzero(games != self.previous):
            self.route[r] = (self.routing.integers(1, len(self.policies))
                             if self.routing.random() < self.pool_fraction else 0)
        self.previous[:] = games
        active = env.stage >= 0
        fresh = np.flatnonzero(active & (env.substep == 0))
        rows = fresh[self.sampling.random(len(fresh)) < self.sample_prob]
        # Snapshot before stepping, always using the public observation allowlist.
        data = {k: torch.from_numpy(getattr(env, k)[rows].copy()) for k in FIELDS}
        data['obs_num'] = data['obs_num'].half()
        choices = np.zeros(env.num_envs, dtype=np.int32)
        routes = np.where(env.actor_seat == 0, 0, self.route)
        for i in np.unique(routes[active]):
            selected = np.flatnonzero(active & (routes == i))
            choices[selected] = self.policies[i](env, selected)[selected]
        env.step(choices)
        return data


class ReplayBuffer:
    def __init__(self, capacity, example, device):
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.data = {k: torch.empty((capacity, *example[k].shape[1:]),
                                   dtype=example[k].dtype, device=device) for k in FIELDS}

    def add(self, batch):
        n = len(batch['label_opp_hand'])
        if n > self.capacity:
            batch = {k: v[-self.capacity:] for k, v in batch.items()}
            n = self.capacity
        # Slice copies support the uint16 observation dtype on CUDA as well.
        first = min(n, self.capacity-self.cursor)
        for k, dest in self.data.items():
            dest[self.cursor:self.cursor+first].copy_(batch[k][:first])
            if n > first:
                dest[:n-first].copy_(batch[k][first:])
        self.cursor = (self.cursor+n) % self.capacity
        self.size = min(self.capacity, self.size+n)

    def sample(self, n):
        rows = torch.randint(self.size, (n,), device=next(iter(self.data.values())).device)
        # index_select preserves the compact storage dtypes.
        return {k: torch.index_select(v, 0, rows) for k, v in self.data.items()}


@torch.no_grad()
def evaluate_belief(model, data, device, batch_size=32, num_samples=64, presence_sample_batch_size=None):
    if batch_size <= 0 or num_samples <= 0 or (presence_sample_batch_size is not None and presence_sample_batch_size <= 0):
        raise ValueError('evaluation batch and sample sizes must be positive')
    from .belief import IndependentBelief
    previous = model.training
    model.eval()
    lp, cp, accuracy = [], [], []
    bc = torch.zeros(15, dtype=torch.int64)
    ps, ys = torch.zeros(15, dtype=torch.float64), torch.zeros(15, dtype=torch.float64)
    try:
        for start in range(0, len(data['decision']), batch_size):
            obs = {k: data[k][start:start+batch_size].to(device) for k in OBS_FIELDS}
            labels = data['label_opp_hand'][start:start+batch_size].to(device)
            total, count, _ = model.log_prob(obs, labels)
            lp.append(total.cpu()); cp.append(count.cpu())
            accuracy.append((model.count_probs(obs).argmax(1) == labels.long().sum(1)).cpu())
            if isinstance(model, IndependentBelief) or presence_sample_batch_size is None:
                presence = model.presence_probs(obs, num_samples)
            else:
                presence = torch.zeros_like(labels, dtype=torch.float32)
                for offset in range(0, num_samples, presence_sample_batch_size):
                    size = min(presence_sample_batch_size, num_samples-offset)
                    presence.add_((model.sample(obs, size) > 0).float().sum(1))
                presence.div_(num_samples)
            metrics = calibration_metrics(presence.cpu(), labels.cpu())
            for i, row in enumerate(metrics['reliability']):
                bc[i] += row['count']
                ps[i] += (row['mean_prediction'] or 0)*row['count']
                ys[i] += (row['actual_fraction'] or 0)*row['count']
        return dict(**likelihood_metrics(torch.cat(lp).numpy(), data['decision']),
                    count_nll=float(-torch.cat(cp).mean()),
                    count_accuracy=float(torch.cat(accuracy).float().mean()),
                    **_calibration_summary(bc, ps, ys))
    finally:
        model.train(previous)
