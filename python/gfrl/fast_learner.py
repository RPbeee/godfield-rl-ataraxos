"""Packed autograd execution of the existing model, without changing parameters.

Only the final layer's queries consumed by the heads are evaluated. Every key
and value is retained: this is ordinary full, bidirectional attention.
"""
import torch
from torch.nn import functional as F
from .learner import decision_terms


def embed(model, obs, positions, reduced=False):
    t = obs['obs_type'].shape[1]
    flat = {k: v.flatten(0, 1).index_select(0, positions) for k, v in obs.items()
            if k != 'obs_valid'}
    p = model.positions
    slots = positions % t
    role = torch.full_like(slots, 8)
    role = torch.where(slots == p['cls'], 0, role)
    role = torch.where(slots == p['self'], 1, role)
    role = torch.where(slots == p['opponent'], 2, role)
    for lo, hi, r in ((p['hand_start'], p['pending'], 3),
                      (p['self_used_start'], p['opponent_used_start'], 6),
                      (p['opponent_used_start'], p['history_start'], 7)):
        role = torch.where((slots >= lo) & (slots < hi), r, role)
    role = torch.where(slots == p['pending'], 4, role)
    role = torch.where(slots == p['offer'], 5, role)
    table = model.card_embedding.weight + model.card_projection(model.card_features)
    if reduced:
        from .gpu.embedding import EmbeddingSum

        def frozen_padding(weight):
            return torch.cat((weight[:1].detach(), weight[1:]))

        tables = [frozen_padding(model.type_embedding.weight), model.role_embedding.weight,
                  table, model.aux_projection(table).float()]
        indices = [flat['obs_type'].long(), role+len(tables[0])]
        offset = len(tables[0])+len(tables[1])
        indices += [flat['obs_cards'][:, j].long()+offset for j in range(8)]
        offset += len(table)
        indices.append(flat['obs_aux_card'].long()+offset)
        offset += len(table)
        for column, embedding in enumerate(model.category_embeddings):
            tables.append(frozen_padding(embedding.weight))
            indices.append(flat['obs_cat'][:, column].long()+offset)
            offset += embedding.num_embeddings
        x = EmbeddingSum.apply(torch.cat(tables), torch.stack(indices, 1))
        return x + model.num_projection(flat['obs_num'].float())
    x = model.type_embedding(flat['obs_type'].long()) + model.role_embedding(role)
    for slot in range(8):
        x = x + F.embedding(flat['obs_cards'][:, slot].long(), table)
    x = x + model.aux_projection(F.embedding(flat['obs_aux_card'].long(), table))
    for column, embedding in enumerate(model.category_embeddings):
        x = x + embedding(flat['obs_cat'][:, column].long())
    return x + model.num_projection(flat['obs_num'].float())


def encode(model, x, cu, head_indices, head_cu, max_length, short=False, fp8=False, low_precision=False, grouped=False):
    from torch.nn.attention.varlen import varlen_attn
    if short:
        from .gpu.attention_training import short_attention
        varlen_attn = short_attention
    linear = F.linear
    if fp8:
        from .gpu.fp8_training import linear
    d, h = model.config.d, model.config.heads
    if low_precision:
        x = x.to(torch.bfloat16)
    for i, layer in enumerate(model.encoder.layers):
        norm = layer.norm1(x)
        qkv = linear(norm, layer.self_attn.in_proj_weight,
                       layer.self_attn.in_proj_bias).view(-1, 3, h, d//h)
        q, k, v = qkv.unbind(1)
        final = i == len(model.encoder.layers)-1
        if grouped:
            from .gpu.attention_packed_training import packed_attention
            attended = packed_attention(qkv, head_cu if final else cu.clone(), cu,
                len(head_indices) if final else len(x), 21 if final else max_length)
            if final:
                x = x.index_select(0, head_indices)
        else:
            if final:
                q = q.index_select(0, head_indices)
                x = x.index_select(0, head_indices)
            attended = varlen_attn(q, k, v, head_cu if final else cu.clone() if short else cu, cu,
                                  21 if final else max_length, max_length)
        proj = layer.self_attn.out_proj
        x = x + linear(attended.reshape(-1, d), proj.weight, proj.bias)
        norm = layer.norm2(x)
        ff = layer.activation(linear(norm, layer.linear1.weight, layer.linear1.bias))
        x = x + linear(ff, layer.linear2.weight, layer.linear2.bias)
    return model.encoder.norm(x)


def heads(model, encoded, inputs):
    owners = inputs['owners']
    stage = inputs['stage'].long()
    b, _, d = encoded.shape
    slots = inputs['chosen_cards'].long().clamp(min=0)
    # Gather just the selected hand tokens, not [stages,all_tokens,d].
    hand = encoded[:, 3:21]
    selected = hand[owners[:, None], slots]
    valid = torch.arange(18, device=slots.device)[None] < inputs['chosen_count'][:, None]
    selected = (selected * valid.unsqueeze(-1)).sum(1)
    cls = encoded[:, 0]
    q = model.query(torch.cat((cls[owners],
        model.kind_embedding(inputs['chosen_kind'].long()+1), selected,
        model.stage_embedding(stage), model.target_embedding(inputs['chosen_target'].long()+1),
        model.hp_embedding(inputs['chosen_alloc_hp'].long()+1)), -1))
    kind = F.pad(model.kind_head(q), (0, 11))
    card = F.pad(torch.cat(((hand[owners]*model.card_query(q)[:, None]).sum(-1),
                            model.stop_head(q)), -1), (0, 2))
    target = F.pad(model.target_head(q), (0, 18))
    alternatives = torch.stack((kind, card, target, model.hp_head(q), model.mp_head(q)), 1)
    logits = alternatives.gather(1, stage[:, None, None].expand(-1, 1, 21)).squeeze(1)
    lp = F.log_softmax(logits.float().masked_fill(~inputs['mask'].bool(), -1e9), -1)
    value = model.value_head(cls).float()
    return decision_terms(lp, inputs['old_lp'], inputs['mask'].bool(),
                           inputs['choice'], owners, b), value


class PackedForward:
    def __init__(self, learner, compile=True, reduced=False, short=False, fp8=False, low_precision=False, grouped=False):
        self.learner = learner
        self.embed, self.encode, self.heads = embed, encode, heads
        if reduced:
            self.embed = lambda model, obs, positions: embed(model, obs, positions, reduced=True)
        if short or fp8 or low_precision or grouped:
            self.encode = lambda model, x, cu, indices, hcu, length: encode(model, x, cu, indices, hcu, length, short=short, fp8=fp8, low_precision=low_precision, grouped=grouped)
        if compile:
            self.embed = torch.compile(self.embed, dynamic=True, fullgraph=True)
            self.encode = torch.compile(self.encode, dynamic=True, fullgraph=True)
            self.heads = torch.compile(heads, dynamic=True)

    def __call__(self, inputs):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        learner, model = self.learner, self.learner.model
        valid = inputs['obs_valid'].bool()
        b, t = valid.shape
        positions = torch.where(valid.flatten())[0]
        cu = F.pad(valid.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
        head_valid = valid[:, :21]
        head_cu = F.pad(head_valid.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
        head_indices = torch.where((positions % t) < 21)[0]
        with learner.autocast(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            obs = {k: inputs[k] for k in ('obs_type', 'obs_cards', 'obs_aux_card', 'obs_cat', 'obs_num', 'obs_valid')}
            x = self.embed(model, obs, positions)
            x = self.encode(model, x, cu, head_indices, head_cu, t)
            encoded = x.new_zeros((b*21, model.config.d)).index_copy(0, torch.where(head_valid.flatten())[0], x).view(b, 21, -1)
            return self.heads(model, encoded, inputs)


BACKENDS = ('eager', 'packed', 'packed-eager', 'packed-reduce', 'packed-short',
            'packed-reduce-short', 'packed-bf16', 'packed-grouped', 'packed-fp8')


def configure_fast_learner(learner, backend='packed'):
    if backend not in BACKENDS[1:]:
        raise ValueError('unknown packed learner backend: '+backend)
    if learner.device.type != 'cuda' or learner.dtype not in ('bf16', 'fp16'):
        raise ValueError('packed learner requires CUDA and bf16 or fp16')
    low_precision = backend in ('packed-bf16', 'packed-grouped', 'packed-fp8')
    grouped = backend in ('packed-grouped', 'packed-fp8')
    if low_precision and learner.dtype != 'bf16':
        raise ValueError(backend+' requires bf16')
    if low_precision and (learner.model.config.heads % 2 or
                          learner.model.config.d // learner.model.config.heads != 32):
        raise ValueError(backend+' requires an even head count and head dimension 32')
    learner._packed_forward = PackedForward(
        learner, compile=backend != 'packed-eager',
        reduced='reduce' in backend or low_precision,
        short='short' in backend or low_precision,
        fp8=backend == 'packed-fp8', low_precision=low_precision, grouped=grouped)
    # Recreate with the same state; fused Adam preserves Adam's update rule.
    state = learner.optimizer.state_dict()
    learner.optimizer = torch.optim.Adam(learner.model.parameters(), lr=learner.config.lr, fused=True)
    learner.optimizer.load_state_dict(state)
    for group in learner.optimizer.param_groups:
        group.update(fused=True, foreach=False)
    learner.config.learner_backend = backend
