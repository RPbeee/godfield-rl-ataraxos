"""Normalized autoregressive distribution over sorted opponent-hand multisets."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import Model, ModelConfig, OBS_FIELDS
from .belief_metrics import validate_labels


@dataclass(frozen=True)
class BeliefConfig:
    depth: int = 2
    heads: int = 8
    ff: int = 0  # zero means 4 * encoder width
    dropout: float = 0.0

    def __post_init__(self):
        if self.depth <= 0 or self.heads <= 0 or self.ff < 0 or not 0 <= self.dropout < 1:
            raise ValueError('invalid decoder depth, heads, ff, or dropout')


class Belief(Model):
    def __init__(self, encoder_config=ModelConfig(), config=BeliefConfig()):
        super().__init__(encoder_config)
        # Reuse Model.encode, including packing, without retaining policy heads.
        for name in ('kind_embedding', 'stage_embedding', 'target_embedding', 'hp_embedding',
                     'query', 'kind_head', 'card_query', 'stop_head', 'target_head',
                     'hp_head', 'mp_head', 'value_head'):
            delattr(self, name)
        self.belief_config = config
        d = encoder_config.d
        self.count_head = nn.Linear(d, 19)
        self.decoder_cards = nn.Embedding(297, d)  # 0 is BOS, never an output
        self.decoder_positions = nn.Embedding(18, d)
        self.decoder_counts = nn.Embedding(19, d)
        layer = nn.TransformerDecoderLayer(d, config.heads, config.ff or 4*d,
                                           config.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, config.depth, norm=nn.LayerNorm(d))
        for block in self.decoder.layers:
            for p in block.parameters():
                if p.ndim > 1:
                    nn.init.xavier_uniform_(p)
        self.card_head = nn.Linear(d, 296)

    @classmethod
    def from_policy(cls, checkpoint, config=BeliefConfig()):
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = cls(ModelConfig(**state['model_config']), config)
        encoder_keys = set(model.state_dict()) - {
            k for k in model.state_dict() if k.startswith(('count_head.', 'decoder', 'card_head.'))}
        weights = {k: state['ema'][k] for k in encoder_keys}
        model.load_state_dict(weights, strict=False)
        return model

    def memory(self, obs):
        # Strict input allowlist: even callers supplying a full record cannot leak labels.
        public = {k: obs[k] for k in OBS_FIELDS}
        encoded = self.encode(public)
        valid = public['obs_valid'].bool()
        order = torch.argsort((~valid).int(), dim=1, stable=True)
        width = int(valid.sum(1).max())
        slots = order[:, :width]
        return encoded[:, 0], encoded.gather(1, slots[..., None].expand(-1, -1, encoded.shape[-1])), ~valid.gather(1, slots)

    def decode(self, memory, padding, inputs, counts):
        """inputs=[BOS,c1,...]; returns normalized log P(next | prefix,n)."""
        length = inputs.shape[1]
        x = (self.decoder_cards(inputs.long()) + self.decoder_positions(
            torch.arange(length, device=inputs.device))[None] + self.decoder_counts(counts.long())[:, None])
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        out = self.decoder(x, memory, tgt_mask=causal, memory_key_padding_mask=padding)
        logits = self.card_head(out).float()
        codes = torch.arange(1, 297, device=x.device)
        return F.log_softmax(logits.masked_fill(codes < inputs[..., None], -torch.inf), -1)

    def log_prob(self, obs, labels):
        counts = labels.long().sum(1)
        validate_labels(labels, counts)
        cls, memory, padding = self.memory(obs)
        count_part = self.count_head(cls).float().log_softmax(-1).gather(1, counts[:, None]).squeeze(1)
        length = int(counts.max())
        if length == 0:
            return count_part, count_part, count_part * 0
        # searchsorted of cumulative multiplicities yields ascending codes, duplicates included.
        ranks = torch.arange(length, device=labels.device)[None].expand(len(labels), -1).contiguous()
        sequence = torch.searchsorted(labels.long().cumsum(1).contiguous(), ranks, right=True).clamp_max(296)
        inputs = F.pad(sequence[:, :-1], (1, 0))
        lp = self.decode(memory, padding, inputs, counts).gather(2, (sequence-1).clamp_min(0)[..., None]).squeeze(-1)
        card_part = lp.masked_fill(ranks >= counts[:, None], 0).sum(1)
        return count_part + card_part, count_part, card_part

    @torch.no_grad()
    def count_probs(self, obs):
        cls, _, _ = self.memory(obs)
        return self.count_head(cls).float().softmax(-1)

    @torch.no_grad()
    def sample(self, obs, num_samples, generator=None):
        if num_samples <= 0:
            raise ValueError('num_samples must be positive')
        cls, memory, padding = self.memory(obs)
        b = len(cls)
        counts = torch.multinomial(self.count_head(cls).float().softmax(-1), num_samples,
                                   replacement=True, generator=generator).flatten()
        memory = memory.repeat_interleave(num_samples, 0)
        padding = padding.repeat_interleave(num_samples, 0)
        result = torch.zeros(b*num_samples, 297, dtype=torch.long, device=cls.device)
        inputs = torch.zeros(b*num_samples, 1, dtype=torch.long, device=cls.device)
        for t in range(int(counts.max())):
            lp = self.decode(memory, padding, inputs, counts)[:, -1]
            card = torch.multinomial(lp.exp(), 1, generator=generator) + 1
            result.scatter_add_(1, card, (counts > t).long()[:, None])
            inputs = torch.cat((inputs, card), 1)
        return result.reshape(b, num_samples, 297)

    @torch.no_grad()
    def presence_probs(self, obs, num_samples=64):
        return (self.sample(obs, num_samples) > 0).float().mean(1)
