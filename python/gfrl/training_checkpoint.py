"""Training-only checkpoint reader; inference never deserializes the critic.

The main .pt retains the existing policy/EMA/pool schema and a relative critic
companion reference. Keep both files when moving a training checkpoint. The
companion is published before the main file is atomically replaced.
"""
from pathlib import Path

import torch


def load_training_checkpoint(path, *, load_critic=True):
    path = Path(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    reference = state.get('critic_checkpoint')
    if load_critic and reference is not None:
        name = reference['file']
        if Path(name).name != name:
            raise ValueError('critic companion must be a sibling checkpoint file')
        companion = torch.load(path.parent/name, map_location='cpu', weights_only=False)
        if companion['iteration'] != state['iteration']:
            raise ValueError('critic companion iteration differs from policy checkpoint')
        state.update({k: v for k, v in companion.items() if k != 'iteration'})
    return state
