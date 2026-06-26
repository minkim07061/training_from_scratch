"""Training-loop scaffold for next-token language modeling."""

from collections.abc import Iterable

import torch
from torch import nn

from scratch_transformer.model import TransformerLM
import numpy as np


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
    model.train()
    batch, seq_len = input_ids.shape
    logits, _ = model(input_ids)
    logits = logits.reshape(batch*seq_len, -1)
    target_ids = target_ids.reshape(-1)
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(logits, target_ids)
    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )
    optimizer.step()
    optimizer.zero_grad()
    return loss.detach()


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
    losses = []
    for batch in batches:
        loss = train_step(model, optimizer, batch[0], batch[1], grad_clip=grad_clip)
        losses.append(loss)
    return np.mean(losses)


def next_token_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy for next-token prediction.

    This helper is intentionally small and safe to inspect when implementing
    train_step. It is not the training loop itself.
    """
    vocab_size = logits.shape[-1]
    return nn.functional.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
