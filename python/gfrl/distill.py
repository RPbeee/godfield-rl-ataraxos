"""Public-only search distillation format and teacher-forced objectives.

v1: root -> candidate_offsets -> stage_offsets. gzip compresses padding and
repeated prefixes; integer columns use their smallest exact signed/byte dtype.
No env objects or privileged fields are serialized.
"""
import gzip
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .actor import PREFIX_FIELDS
from .model import OBS_FIELDS

FORMAT_VERSION = 1


def outcome_target(result, seat, agent_seat=0, truncated=False):
    value = 0 if truncated else result * (1 if seat == agent_seat else -1)
    return np.eye(3, dtype=np.uint8)[{1: 0, -1: 1, 0: 2}[value]]


def compact(values):
    a = np.asarray(values)
    if a.dtype.kind in 'iu':
        lo, hi = a.min(initial=0), a.max(initial=0)
        for dtype in (np.uint8, np.int8, np.int16, np.int32, np.int64):
            lim = np.iinfo(dtype)
            if lo >= lim.min and hi <= lim.max:
                a = a.astype(dtype)
                break
    return torch.from_numpy(a.copy())


def pack_records(records, metadata):
    if not records:
        raise ValueError('no recorded roots')
    candidates = [c for r in records for c in r['candidates']]
    stages = [p for c in candidates for p in c['prefixes']]
    return dict(format_version=FORMAT_VERSION, metadata=metadata,
                obs={k: compact(np.stack([r['obs'][k] for r in records])) for k in OBS_FIELDS},
                prefix={k: compact(np.stack([s[k] for s in stages])) for k in PREFIX_FIELDS},
                candidate_offsets=torch.tensor([0]+list(np.cumsum([len(r['candidates']) for r in records]))),
                stage_offsets=torch.tensor([0]+list(np.cumsum([len(c['path']) for c in candidates]))),
                choice=compact([a for c in candidates for a in c['path']]),
                probabilities=torch.tensor([p for r in records for p in r['probabilities']], dtype=torch.float64),
                q=torch.tensor([float('nan') if c['q'] is None else c['q'] for c in candidates], dtype=torch.float64),
                log_pi=torch.tensor([c['log_pi'] for c in candidates], dtype=torch.float64),
                log_rho=torch.tensor([c['log_rho'] for c in candidates], dtype=torch.float64),
                valid_worlds=compact([c['valid_worlds'] for c in candidates]),
                game=compact([r['game'] for r in records]), seat=compact([r['seat'] for r in records]),
                decision=compact([r['decision'] for r in records]),
                phase=compact([int(r['phase'] == 'response') for r in records]),
                outcome=compact(np.stack([r['outcome'] for r in records])))


def save_shard(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with gzip.open(tmp, 'wb', compresslevel=1) as f:
        torch.save(data, f)
    tmp.replace(path)


def load_shard(path):
    with gzip.open(path, 'rb') as f:
        data = torch.load(f, map_location='cpu', weights_only=False)
    if data['format_version'] != FORMAT_VERSION:
        raise ValueError('unsupported distillation format')
    if set(data['obs']) != set(OBS_FIELDS) or set(data['prefix']) != set(PREFIX_FIELDS):
        raise ValueError('unexpected observation/prefix fields')
    return data


def select_rows(data, rows):
    """Copy a compact CPU shard subset, rebasing both ragged offset columns."""
    rows = torch.as_tensor(rows, dtype=torch.long)
    co, so = data['candidate_offsets'], data['stage_offsets']
    counts = co[rows+1]-co[rows]
    candidates = torch.cat([torch.arange(int(co[r]), int(co[r+1])) for r in rows])
    stage_counts = so[candidates+1]-so[candidates]
    stages = torch.cat([torch.arange(int(so[c]), int(so[c+1])) for c in candidates])
    subset = dict(format_version=data['format_version'], metadata=data['metadata'],
                  obs={k: v[rows].clone() for k, v in data['obs'].items()},
                  prefix={k: v[stages].clone() for k, v in data['prefix'].items()},
                  candidate_offsets=torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0))),
                  stage_offsets=torch.cat((torch.zeros(1, dtype=torch.long), stage_counts.cumsum(0))))
    for k in ('game', 'seat', 'decision', 'phase', 'outcome'):
        subset[k] = data[k][rows].clone()
    for k in ('probabilities', 'q', 'log_pi', 'log_rho', 'valid_worlds'):
        subset[k] = data[k][candidates].clone()
    subset['choice'] = data['choice'][stages].clone()
    return subset


def search_values(batch):
    """Root-actor search expectation; every candidate must have finite q."""
    owners, q = batch['candidate_owners'], batch['q']
    n = len(batch['outcome'])
    finite = torch.isfinite(q)
    invalid = q.new_zeros(n).index_add(0, owners, (~finite).float())
    expected = q.new_zeros(n).index_add(0, owners, batch['probabilities']*q.nan_to_num())
    return expected, invalid == 0


def value_targets(outcome, search_value, valid, mix=0.):
    """Mix outcome with T(v) in win/loss/draw order; missing search falls back."""
    if mix == 0:
        return outcome
    v = search_value.clamp(-1, 1)
    target = torch.stack(((1+v)/2, (1-v)/2, torch.zeros_like(v)), -1)
    return torch.where(valid[:, None], (1-mix)*outcome+mix*target, outcome)


def prepare_batch(data, rows, device='cpu'):
    rows = torch.as_tensor(rows, dtype=torch.long)
    co, so = data['candidate_offsets'], data['stage_offsets']
    cs = [torch.arange(int(co[r]), int(co[r+1])) for r in rows]
    candidates = torch.cat(cs)
    root_of_c = torch.repeat_interleave(torch.arange(len(rows)), torch.tensor([len(c) for c in cs]))
    ss = [torch.arange(int(so[c]), int(so[c+1])) for c in candidates]
    stages = torch.cat(ss)
    candidate_of_s = torch.repeat_interleave(torch.arange(len(candidates)), torch.tensor([len(s) for s in ss]))
    batch = dict(obs={k: v[rows].to(device) for k, v in data['obs'].items()},
                 prefix={k: v[stages].to(device) for k, v in data['prefix'].items()},
                 owners=root_of_c[candidate_of_s].to(device),
                 candidate_owners=root_of_c.to(device), stage_owners=candidate_of_s.to(device),
                 choice=data['choice'][stages].long().to(device),
                 outcome=data['outcome'][rows].float().to(device))
    batch.update({k: data[k][candidates].float().to(device) for k in ('probabilities', 'q', 'log_pi')})
    return batch


def forward_paths(model, batch):
    encoded = model.encode(batch['obs'])
    lp, _ = model.heads(encoded.index_select(0, batch['owners']), **batch['prefix'])
    value = model.value_head(encoded[:, model.positions['cls']]).float()
    selected = lp.gather(1, batch['choice'][:, None]).squeeze(1)
    paths = selected.new_zeros(len(batch['probabilities'])).index_add(0, batch['stage_owners'], selected)
    return lp, paths, value


def objective(model, reference, batch, kl_coef=.1, value_coef=.5,
              value_target_mix=0., policy_coef=1.):
    lp, paths, value = forward_paths(model, batch)
    with torch.no_grad():
        ref_lp, ref_paths, _ = forward_paths(reference, batch)
    mask = batch['prefix']['mask'].bool()
    # Same mask-before-arithmetic convention as learner.decision_terms.
    kl_stage = (lp.exp().masked_fill(~mask, 0) *
                (lp.masked_fill(~mask, 0)-ref_lp.masked_fill(~mask, 0))).sum(-1)
    kl_c = paths.new_zeros(len(paths)).index_add(0, batch['stage_owners'], kl_stage)
    p, n = batch['probabilities'], len(value)
    policy = -(p*paths).sum()/n
    anchor = (p*kl_c).sum()/n
    search_value, valid = search_values(batch)
    target = value_targets(batch['outcome'], search_value, valid, value_target_mix)
    log_value = F.log_softmax(value, -1)
    ce = -(target*log_value).sum(-1).mean()
    loss = policy_coef*policy + kl_coef*anchor + value_coef*ce
    metrics = dict(policy=float(policy.detach()), anchor_kl=float(anchor.detach()),
                   value_ce=float(ce.detach()), value_accuracy=float((value.argmax(-1) == batch['outcome'].argmax(-1)).float().mean()))
    with torch.no_grad():
        owners = batch['candidate_owners']
        def normalized(scores):
            maximum = scores.new_full((n,), -torch.inf).scatter_reduce_(0, owners, scores, reduce='amax')
            shifted = scores-maximum[owners]
            normalizer = scores.new_zeros(n).index_add(0, owners, shifted.exp())
            return shifted-normalizer[owners].log()
        theta, ref = normalized(paths), normalized(ref_paths)
        search_to_policy_kl = (p*(p.clamp_min(1e-30).log()-theta)).sum()/n
        candidate_kl = (theta.exp()*(theta-p.clamp_min(1e-30).log())).sum()/n
        def argmax_by_root(scores):
            maximum = scores.new_full((n,), -torch.inf).scatter_reduce_(0, owners, scores, reduce='amax')
            indices = torch.arange(len(scores), device=scores.device)
            indices = indices.masked_fill(scores != maximum[owners], len(scores))
            return indices.new_full((n,), len(scores)).scatter_reduce_(0, owners, indices, reduce='amin')
        agreement = (argmax_by_root(paths) == argmax_by_root(p)).float().mean()
        q = batch['q'].nan_to_num()
        q_count = int(valid.sum())
        q_metrics = {}
        for name, probabilities in (('q_theta', theta.exp()), ('q_ref', ref.exp()), ('q_search', p)):
            expected = paths.new_zeros(n).index_add(0, owners, probabilities*q)
            q_metrics[name] = float(expected[valid].sum())/max(q_count, 1)
        prediction = value.softmax(-1)[:, 0]-value.softmax(-1)[:, 1]
        outcome_scalar = batch['outcome'][:, 0]-batch['outcome'][:, 1]
        metrics.update(outcome_ce=float(-(batch['outcome']*log_value).sum(-1).mean()),
                       outcome_brier=float((prediction-outcome_scalar).square().mean()),
                       search_value_mse=float((prediction[valid]-search_value[valid]).square().sum())/max(q_count, 1),
                       candidate_kl=float(candidate_kl), search_to_policy_kl=float(search_to_policy_kl),
                       top1=float(agreement),
                       q_roots=q_count, **q_metrics)
    return loss, metrics
