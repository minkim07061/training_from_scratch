"""Round 3 metrics scaffold.

Goal:
    Implement the small logging/metric helpers used during training.

Suggested test:
    python3 -m pytest round3_training_lab/tests/test_metrics.py
"""

from __future__ import annotations

import math  # noqa: F401  (perplexity needs math.exp)


def perplexity(loss: float) -> float:
    """Convert a mean cross-entropy loss (in nats) to perplexity.

    Implementation checkpoint:
        - perplexity = exp(loss).
        - loss 0.0 -> perplexity 1.0.
    """
    raise NotImplementedError


class AverageMeter:
    """Track a running average of a scalar value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the accumulated total and count to zero."""
        raise NotImplementedError

    def update(self, value: float, n: int = 1) -> None:
        """Add ``value`` observed ``n`` times to the running average.

        Implementation checkpoints:
            - Accumulate a weighted sum (value * n) and a total count (n).
            - Support n > 1 for batch-weighted averaging.
        """
        raise NotImplementedError

    @property
    def avg(self) -> float:
        """Return the current mean, or 0.0 when nothing has been recorded."""
        raise NotImplementedError


def tokens_per_second(num_tokens: int, elapsed_seconds: float) -> float:
    """Return training throughput in tokens per second.

    Implementation checkpoint:
        - Return num_tokens / elapsed_seconds.
        - Decide how to handle elapsed_seconds == 0 (e.g. return 0.0).
    """
    raise NotImplementedError
