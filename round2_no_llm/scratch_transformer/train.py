"""Round 2 training-loop scaffold.

Goal:
    Rebuild next-token training helpers.

Suggested test:
    python3 -m pytest tests/test_train.py
"""

from collections.abc import Iterable

import torch

from scratch_transformer.model import TransformerLM


def train_step(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    grad_clip: float | None = None,
) -> torch.Tensor:
    """Run one optimizer step.

    Implementation checkpoints:
        1. Put model in train mode.
        2. Clear stale gradients before backward.
        3. Call model(input_ids) and unpack logits.
        4. Compute next-token cross entropy.
        5. Backpropagate.
        6. Optionally clip gradients.
        7. Step optimizer.
        8. Clear gradients again so helper leaves a clean state.
        9. Return detached scalar loss.
    """
    raise NotImplementedError


def train_epoch(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    grad_clip: float | None = None,
) -> float:
    """Train over batches and return mean loss."""
    raise NotImplementedError


def next_token_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Compute cross entropy over flattened batch and sequence dimensions."""
    raise NotImplementedError

