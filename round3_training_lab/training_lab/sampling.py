"""Round 3 sampling scaffold.

Goal:
    Implement the logit-shaping tools used during text generation:
    temperature scaling, top-k filtering, and top-p (nucleus) filtering.

Suggested test:
    python3 -m pytest round3_training_lab/tests/test_sampling.py

Convention:
    All logits tensors have shape (batch, vocab_size). Filtering functions
    return a tensor of the same shape where removed positions are set to
    -inf so a following softmax gives them zero probability.
"""

from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by 1 / temperature.

    Implementation checkpoints:
        - Return logits / temperature.
        - temperature == 1.0 leaves logits unchanged.
        - Assume temperature > 0 (caller handles the greedy case).
    """
    raise NotImplementedError


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k largest logits per row; set the rest to -inf.

    Implementation checkpoints:
        1. If k <= 0 or k >= vocab_size, return logits unchanged.
        2. Find the k-th largest value in each row (the threshold).
        3. Set every position strictly below that threshold to -inf.

    Invariant:
        Each row keeps exactly k finite entries (ties aside).
    """
    raise NotImplementedError


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of top tokens whose cumulative
    probability is at least p; set the rest to -inf.

    Implementation checkpoints:
        1. If p >= 1.0, return logits unchanged.
        2. Convert logits to probabilities (softmax over the last dim).
        3. Sort probabilities descending and take the cumulative sum.
        4. Keep tokens up to and including the first one where the cumulative
           sum reaches p; drop the rest.
        5. Always keep at least the single most probable token.
        6. Map the kept/dropped decision back to the original token order and
           set dropped positions to -inf.
    """
    raise NotImplementedError


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one next token id per row.

    Args:
        logits: shape (batch, vocab_size).

    Returns:
        LongTensor of shape (batch,) with sampled token ids.

    Implementation checkpoints:
        1. Apply temperature.
        2. Apply top_k filtering when top_k is provided.
        3. Apply top_p filtering when top_p is provided.
        4. Softmax to probabilities.
        5. Sample with torch.multinomial (pass the generator for reproducibility).

    Note:
        With top_k == 1 this reduces to greedy argmax; the test relies on that.
    """
    raise NotImplementedError
