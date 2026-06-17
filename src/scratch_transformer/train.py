"""Training-loop scaffold for next-token language modeling."""

from collections.abc import Iterable

import torch
from torch import nn

from scratch_transformer.model import TransformerLM


def train_step(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    grad_clip: float | None = None,
) -> torch.Tensor:
    """Run one optimization step and return the detached loss.

    TODO:
    - Set the model to train mode.
    - Call model(input_ids) and get logits.
    - Flatten logits/targets and compute cross-entropy loss.
    - Backpropagate, optionally clip gradients, step the optimizer, and zero grads.
    - Return a detached scalar loss tensor for logging.
    """
    raise NotImplementedError("TODO: implement one training step")


def train_epoch(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    grad_clip: float | None = None,
) -> float:
    """Train over an iterable of ``(input_ids, target_ids)`` batches.

    TODO:
    - Iterate over batches and call train_step.
    - Accumulate scalar losses.
    - Return the mean loss for the epoch.
    """
    raise NotImplementedError("TODO: implement the epoch training loop")


def next_token_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy for next-token prediction.

    This helper is intentionally small and safe to inspect when implementing
    train_step. It is not the training loop itself.
    """
    vocab_size = logits.shape[-1]
    return nn.functional.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
