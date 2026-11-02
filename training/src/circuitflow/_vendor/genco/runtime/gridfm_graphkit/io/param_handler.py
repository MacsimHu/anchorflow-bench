"""Adapted: only upstream param_handler.py lines 155-164, plus required imports.

Source: gridfm/gridfm-graphkit, commit 0912c734b0912202e22aad79dde996ae0f1190a0.
Apache-2.0; full original retained under pristine/.
"""
import torch
from gridfm_graphkit.io.registries import PHYSICS_DECODER_REGISTRY


def get_physics_decoder(args) -> torch.nn.Module:
    """
    Load the task module
    """
    task = args.task.task_name

    try:
        return PHYSICS_DECODER_REGISTRY.create(task)
    except KeyError:
        raise ValueError(f"No physics decoder associate to {task} task")
