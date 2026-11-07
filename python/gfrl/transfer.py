"""Host-to-device transfers without per-batch pinned-memory allocation."""
import torch


def tensor_batch(arrays, device):
    device = torch.device(device)
    # Native buffers are persistent views. Blocking transfers finish before the
    # next env.step; CPU consumers must select/copy rows before that mutation.
    return {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
