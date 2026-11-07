"""Public-observation Transformer and conditional, sequential action heads."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from . import _core

OBS_FIELDS = ("obs_type", "obs_cards", "obs_aux_card", "obs_cat", "obs_num", "obs_valid")


@dataclass(frozen=True)
class ModelConfig:
    d: int = 128
    depth: int = 4
    heads: int = 4
    ff: int = 512
    dropout: float = 0.0
    pack: bool = True
    chunk: int = 256

    def __post_init__(self):
        if self.chunk <= 0:
            raise ValueError("chunk must be positive")

    @classmethod
    def preset(cls, size="S", **kwargs):
        return cls(**dict(zip(("d", "depth", "heads", "ff"),
                             {"S": (128, 4, 4, 512), "M": (256, 6, 8, 1024),
                              "L": (384, 8, 8, 1536)}[size])), **kwargs)


PRESETS = {size: ModelConfig.preset(size) for size in ("S", "M", "L")}


class Model(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self._compiled_encoder = None
        schema = _core.obs_schema()
        self.positions = schema["positions"]
        d = config.d
        self.register_buffer("card_features", torch.tensor(_core.card_features().copy()))
        self.type_embedding = nn.Embedding(len(schema["types"]), d, padding_idx=0)
        self.card_embedding = nn.Embedding(297, d, padding_idx=0)
        # No bias: the absent card remains exactly zero in both paths.
        self.card_projection = nn.Linear(schema["Fc"], d, bias=False)
        self.aux_projection = nn.Linear(d, d, bias=False)
        self.category_embeddings = nn.ModuleList([
            nn.Embedding(c["vocab_size"], d, padding_idx=0)
            for c in sorted(schema["cat"].values(), key=lambda c: c["index"])])
        self.num_projection = nn.Linear(schema["F"], d)
        self.role_embedding = nn.Embedding(9, d)
        layer = nn.TransformerEncoderLayer(d, config.heads, config.ff, config.dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.depth, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        # TransformerEncoder clones its input layer; initialize each independently.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.kind_embedding = nn.Embedding(11, d, padding_idx=0)
        self.stage_embedding = nn.Embedding(5, d)
        self.target_embedding = nn.Embedding(4, d, padding_idx=0)
        self.hp_embedding = nn.Embedding(22, d, padding_idx=0)
        self.query = nn.Sequential(nn.Linear(6*d, d), nn.GELU(), nn.Linear(d, d))
        self.kind_head = nn.Linear(d, 10)
        self.card_query = nn.Linear(d, d)
        self.stop_head = nn.Linear(d, 1)
        self.target_head = nn.Linear(d, 3)
        self.hp_head = nn.Linear(d, 21)
        self.mp_head = nn.Linear(d, 21)
        self.value_head = nn.Linear(d, 3)

    def _run_encoder(self, packed, padding):
        if self._compiled_encoder is None:
            return self.encoder(packed, src_key_padding_mask=padding)
        from .runtime import encoder_bucket_shape
        rows, width, _ = packed.shape
        bucket_rows, bucket_width = encoder_bucket_shape(rows, width)
        padded = F.pad(packed, (0, 0, 0, bucket_width-width, 0, bucket_rows-rows))
        mask = F.pad(padding, (0, bucket_width-width, 0, bucket_rows-rows), value=True)
        # Also protect any real zero-length row in a nonempty chunk. Its output
        # is masked to zero by encode; dummy outputs are discarded here.
        mask = mask.clone()
        mask[:, 0] = mask[:, 0] & ~mask.all(1)
        encoded = self._compiled_encoder(padded.contiguous(), mask.contiguous())
        return encoded[:rows, :width]

    def encode(self, obs, *, pack=None, chunk=None):
        """Return [B,T,d] in original positions, with invalid positions zeroed.

        Packing preserves valid-token order and groups rows by length. Dropout
        must be disabled for numerical equivalence with the unpacked path.
        last_encode_tokens counts padded tokens actually sent to the encoder.
        """
        pack = self.config.pack if pack is None else pack
        chunk = self.config.chunk if chunk is None else chunk
        if chunk <= 0:
            raise ValueError("chunk must be positive")
        types = obs["obs_type"].long()
        t = types.shape[1]
        p = self.positions
        # Every hand slot has role 3; history time comes only from obs_num.age.
        role = torch.full((t,), 8, dtype=torch.long, device=types.device)
        role[p["cls"]] = 0
        role[p["self"]] = 1
        role[p["opponent"]] = 2
        role[p["hand_start"]:p["pending"]] = 3
        role[p["pending"]] = 4
        role[p["offer"]] = 5
        role[p["self_used_start"]:p["opponent_used_start"]] = 6
        role[p["opponent_used_start"]:p["history_start"]] = 7
        table = self.card_embedding.weight + self.card_projection(self.card_features)
        x = self.type_embedding(types) + self.role_embedding(role)
        # Sum one slot at a time to bound peak memory for large VecEnv batches.
        for slot in range(8):
            x = x + F.embedding(obs["obs_cards"][..., slot].long(), table)
        x = x + self.aux_projection(F.embedding(obs["obs_aux_card"].long(), table))
        for column, embedding in enumerate(self.category_embeddings):
            x = x + embedding(obs["obs_cat"][..., column].long())
        x = x + self.num_projection(obs["obs_num"].float())
        valid = obs["obs_valid"].bool()
        b, t, _ = x.shape
        self.last_encode_tokens = 0
        if b == 0:
            return x
        if self._compiled_encoder is not None:
            from .runtime import encoder_bucket_shape
            chunk = min(chunk, 256)
        if not pack and self._compiled_encoder is not None:
            self.last_encode_tokens = sum(
                r * w for r, w in (encoder_bucket_shape(min(chunk, b-i), t)
                                    for i in range(0, b, chunk)))
            return torch.cat([self._run_encoder(x[i:i+chunk], ~valid[i:i+chunk])
                              for i in range(0, b, chunk)]).masked_fill(~valid.unsqueeze(-1), 0)
        if not pack:
            self.last_encode_tokens = b * t
            return self.encoder(x, src_key_padding_mask=~valid).masked_fill(
                ~valid.unsqueeze(-1), 0)

        # Stable sort retains original order within valid (and invalid) tokens.
        positions = torch.argsort((~valid).to(torch.int32), dim=1, stable=True)
        lengths = valid.sum(1)
        rows = torch.argsort(lengths, stable=True)
        sorted_lengths = lengths.index_select(0, rows)
        ends = (torch.arange((b + chunk - 1) // chunk, device=x.device) * chunk
                + chunk - 1).clamp(max=b - 1)
        # The only device-to-host synchronization in packing: one length/chunk.
        widths = sorted_lengths.index_select(0, ends).tolist()
        output = None
        for start, width in zip(range(0, b, chunk), widths):
            group = rows[start:start + chunk]
            if width == 0:
                continue
            if self._compiled_encoder is None:
                self.last_encode_tokens += len(group) * width
            else:
                padded_rows, padded_width = encoder_bucket_shape(len(group), width)
                self.last_encode_tokens += padded_rows * padded_width
            slots = positions.index_select(0, group)[:, :width]
            packed = x[group[:, None], slots]
            padding = torch.arange(width, device=x.device)[None] >= lengths[group, None]
            encoded = self._run_encoder(packed, padding)
            encoded = encoded.masked_fill(padding.unsqueeze(-1), 0)
            if output is None:
                output = x.new_zeros(x.shape, dtype=encoded.dtype)
            output[group[:, None], slots] = encoded
        return output if output is not None else torch.zeros_like(x)

    def heads(self, token_outputs, stage, mask, chosen_kind, chosen_cards,
              chosen_count, chosen_target, chosen_alloc_hp):
        """Return masked log probabilities and viewer-relative win/loss/draw logits."""
        b, _, d = token_outputs.shape
        hand = token_outputs[:, self.positions["hand_start"]:self.positions["pending"]]
        slots = chosen_cards.long()
        selected = hand.gather(1, slots.clamp(min=0).unsqueeze(-1).expand(-1, -1, d))
        valid = torch.arange(18, device=slots.device)[None] < chosen_count[:, None]
        selected = (selected * valid.unsqueeze(-1)).sum(1)
        cls = token_outputs[:, self.positions["cls"]]
        q = self.query(torch.cat((cls, self.kind_embedding(chosen_kind.long()+1), selected,
                                 self.stage_embedding(stage.long()),
                                 self.target_embedding(chosen_target.long()+1),
                                 self.hp_embedding(chosen_alloc_hp.long()+1)), dim=-1))
        # All rows run the small heads, avoiding device/host synchronization per stage.
        kind = F.pad(self.kind_head(q), (0, 11))
        card = F.pad(torch.cat(((hand * self.card_query(q)[:, None]).sum(-1),
                               self.stop_head(q)), -1), (0, 2))
        target = F.pad(self.target_head(q), (0, 18))
        alternatives = torch.stack((kind, card, target, self.hp_head(q), self.mp_head(q)), 1)
        logits = alternatives.gather(1, stage.long()[:, None, None].expand(b, 1, 21)).squeeze(1)
        # Finite fp32 sentinel works on MPS and under CUDA bf16 autocast.
        log_probs = F.log_softmax(logits.float().masked_fill(~mask.bool(), -1e9), dim=-1)
        return log_probs, self.value_head(cls).float()
