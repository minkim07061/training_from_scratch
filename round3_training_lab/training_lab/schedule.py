"""Round 3 learning-rate schedule scaffold.

Goal:
    Implement a cosine learning-rate schedule with linear warmup.

Suggested test:
    python3 -m pytest round3_training_lab/tests/test_schedule.py
"""

from __future__ import annotations

import math  # noqa: F401  (you will likely need math.cos and math.pi)


def cosine_lr(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """Return the learning rate for a given step.

    Phases:
        1. Warmup (step < warmup_steps):
               lr grows linearly from 0 up to base_lr.
               Use lr = base_lr * step / warmup_steps.
               So step 0 -> 0.0 and step warmup_steps -> base_lr.
        2. Cosine decay (warmup_steps <= step <= total_steps):
               progress = (step - warmup_steps) / (total_steps - warmup_steps)
               lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
               So progress 0 -> base_lr and progress 1 -> min_lr.
        3. After training (step > total_steps):
               lr stays at min_lr (clamp progress to 1).

    Edge cases to consider:
        - warmup_steps == 0 should skip the warmup phase entirely.
        - Guard against division by zero.
    """
    raise NotImplementedError
