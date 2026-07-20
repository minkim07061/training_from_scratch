"""Round 3 data pipeline scaffold.

Goal:
    Turn a flat stream of token ids into next-token training batches.

Suggested test:
    python3 -m pytest round3_training_lab/tests/test_data.py
"""

from __future__ import annotations

import torch


def train_val_split(
    token_ids: list[int] | torch.Tensor,
    val_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1D token stream into train and validation tensors.

    Implementation checkpoints:
        1. Convert token_ids into a 1D LongTensor.
        2. Compute the split index so the validation set is the LAST
           val_fraction of the stream (chronological hold-out).
        3. Return (train_tokens, val_tokens) as LongTensors.

    Invariants:
        - Ordering is preserved (do not shuffle before splitting).
        - train + val together cover the whole stream with no overlap.
    """
    raise NotImplementedError


class TokenDataset:
    """Fixed-context next-token dataset over a 1D token stream."""

    def __init__(self, token_ids: list[int] | torch.Tensor, context_length: int) -> None:
        self.tokens = torch.as_tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        """Return the number of valid (input, target) windows.

        Implementation checkpoint:
            - A window needs context_length inputs plus one shifted target,
              so the number of valid starting positions is
              len(tokens) - context_length.
        """
        raise NotImplementedError

    def get_batch(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of contiguous windows.

        Returns:
            inputs:  LongTensor of shape (batch_size, context_length)
            targets: LongTensor of shape (batch_size, context_length)

        Implementation checkpoints:
            1. Sample batch_size random start positions in [0, len(self)).
            2. For each start s, inputs = tokens[s : s + context_length].
            3. targets are inputs shifted by one: tokens[s + 1 : s + 1 + context_length].
            4. Stack rows into (batch_size, context_length) tensors.

        Invariants:
            - targets[:, :-1] must equal inputs[:, 1:].
            - Use the provided generator (if any) for reproducible sampling.
        """
        raise NotImplementedError
