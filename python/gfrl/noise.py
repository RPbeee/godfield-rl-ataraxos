"""Stage-wise epsilon-uniform legal action distribution (no extra RNG at zero)."""
import torch


def epsilon_log_probs(log_probs, mask, epsilon):
    if epsilon == 0:
        return log_probs
    legal = mask.bool()
    uniform = legal.float()/legal.sum(-1, keepdim=True).clamp_min(1)
    probs = (1-epsilon)*log_probs.exp()+epsilon*uniform
    return probs.clamp_min(torch.finfo(probs.dtype).tiny).log().masked_fill(~legal, -1e9)
