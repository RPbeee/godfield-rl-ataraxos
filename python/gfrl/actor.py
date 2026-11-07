"""Inference-only Actor. Privileged label_* buffers are never policy inputs."""
from contextlib import nullcontext
from time import perf_counter
from typing import NamedTuple

import numpy as np
import torch

from .model import OBS_FIELDS
from .transfer import tensor_batch

PREFIX_FIELDS = ("stage", "mask", "chosen_kind", "chosen_cards", "chosen_count",
                 "chosen_target", "chosen_alloc_hp")


class ActResult(NamedTuple):
    choices: np.ndarray       # int32 [N], zero for idle rows
    log_prob: np.ndarray | torch.Tensor      # float32 [N], selected stage log probability
    value_logits: np.ndarray | torch.Tensor  # float32 [N,3], viewer-relative win/loss/draw
    log_probs: np.ndarray | torch.Tensor     # float32 [N,21], complete stage distribution


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class Actor:
    def __init__(self, model, env, device, sample=True, *, dtype="fp32",
                 cuda_bf16=False, profile=False):
        if dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError("dtype must be fp32, fp16, or bf16")
        self.device = torch.device(device)
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.sample = sample
        self.dtype = "bf16" if cuda_bf16 else dtype
        self.profile = profile
        self.env = None
        self.token_cache = None
        self.last_profile = {}
        self._bind(env)

    def _bind(self, env):
        if env is self.env:
            return
        self.env = env
        self.token_cache = None
        self._cached = np.zeros(env.num_envs, dtype=bool)

    @torch.inference_mode()
    def act(self, rows=None, *, buffers=None, tensors=None, device_result=False):
        env = self.env
        buffers = buffers if buffers is not None else {
            key: getattr(env, key) for key in (*OBS_FIELDS, *PREFIX_FIELDS, 'substep')}
        # Explicit public allowlist. Never inspect __dict__, dir(), or label_*.
        active = np.flatnonzero(buffers["stage"] >= 0) if rows is None else np.asarray(rows)
        fresh = active[buffers["substep"][active] == 0]
        if np.any(~self._cached[active] & (buffers["substep"][active] != 0)):
            raise RuntimeError("attach Actor at the start of a decision (substep == 0)")
        timings = dict(transfer=0.0, encoder=0.0, heads=0.0, encoded_envs=len(fresh),
                       encoder_tokens=0, unpacked_tokens=0)

        def start():
            if self.profile:
                synchronize(self.device)
            return perf_counter()

        def finish(name, since):
            if self.profile:
                synchronize(self.device)
            timings[name] += perf_counter() - since

        choices = np.zeros(env.num_envs, dtype=np.int32)
        log_prob = torch.zeros(env.num_envs, device=self.device)
        values = torch.zeros((env.num_envs, 3), device=self.device)
        all_lp = torch.zeros((env.num_envs, 21), device=self.device)
        def result():
            outputs = (log_prob, values, all_lp)
            if not device_result:
                outputs = tuple(v.cpu().numpy() for v in outputs)
            return ActResult(choices, *outputs)
        if not len(active):
            self.last_profile = timings
            return result()
        since = start()
        if tensors is None:
            tensors = tensor_batch({k: buffers[k] for k in (*OBS_FIELDS, *PREFIX_FIELDS)}, self.device)
        active_t = torch.as_tensor(active, device=self.device, dtype=torch.long)
        fresh_t = torch.as_tensor(fresh, device=self.device, dtype=torch.long)
        obs = {key: tensors[key].index_select(0, fresh_t) for key in OBS_FIELDS} if len(fresh) else None
        prefix = {key: tensors[key].index_select(0, active_t) for key in PREFIX_FIELDS}
        finish("transfer", since)
        autocast = (torch.autocast(self.device.type,
                                  dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[self.dtype])
                    if self.dtype != "fp32" else nullcontext())
        with autocast:
            since = start()
            if obs is not None:
                encoded = self.model.encode(obs)
                timings["encoder_tokens"] = self.model.last_encode_tokens
                timings["unpacked_tokens"] = encoded.shape[0] * encoded.shape[1]
                if self.token_cache is None:
                    self.token_cache = torch.empty((env.num_envs, *encoded.shape[1:]),
                                                   device=self.device, dtype=encoded.dtype)
                self.token_cache.index_copy_(0, fresh_t, encoded)
                self._cached[fresh] = True
            finish("encoder", since)
            since = start()
            lp, vl = self.model.heads(self.token_cache.index_select(0, active_t), **prefix)
            action = (torch.multinomial(lp.exp(), 1).squeeze(1) if self.sample else lp.argmax(-1))
            picked = lp.gather(1, action[:, None]).squeeze(1)
            finish("heads", since)
        since = start()
        choices[active] = action.cpu().numpy().astype(np.int32)
        log_prob.index_copy_(0, active_t, picked.float())
        values.index_copy_(0, active_t, vl.float())
        all_lp.index_copy_(0, active_t, lp.float())
        finish("transfer", since)
        self.last_profile = timings
        return result()

    def __call__(self, env):
        """evaluate_policy callback; a new VecEnv starts a new cache."""
        self._bind(env)
        return self.act().choices
