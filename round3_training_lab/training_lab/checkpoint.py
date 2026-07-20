"""Round 3 checkpointing scaffold.

Goal:
    Save and restore training state so runs can resume.

Suggested test:
    python3 -m pytest round3_training_lab/tests/test_checkpoint.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    extra: dict | None = None,
) -> None:
    """Save model + optimizer + step to a single file.

    Implementation checkpoints:
        1. Build a dict with:
               "model": model.state_dict()
               "optimizer": optimizer.state_dict()
               "step": step
               "extra": extra or {}
        2. Persist it with torch.save.
    """
    raise NotImplementedError


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict:
    """Load a checkpoint into a model (and optionally an optimizer).

    Implementation checkpoints:
        1. torch.load the file (pass map_location).
        2. Load "model" into the provided model via load_state_dict.
        3. If an optimizer is provided, load "optimizer" into it.
        4. Return a small metadata dict with keys "step" and "extra".

    Invariant:
        After loading, the model's parameters must match the saved parameters.
    """
    raise NotImplementedError
