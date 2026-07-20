"""Round 3 AdamW optimizer scaffold.

Goal:
    Implement AdamW from scratch as a torch.optim.Optimizer subclass.

Suggested test:
    python3 -m pytest round3_training_lab/tests/test_optim.py

The test compares your optimizer against torch.optim.AdamW with identical
hyperparameters and gradients, so your update rule must match the standard
"decoupled weight decay" formulation.
"""

from __future__ import annotations

import torch


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay.

    The constructor boilerplate is provided. You implement ``step``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Implementation checkpoints (per parameter with a gradient):
            1. Decoupled weight decay: p <- p - lr * weight_decay * p
               (applied directly to the parameter, not the gradient).
            2. Maintain per-parameter state:
                   step count t
                   exp_avg      (first moment m)
                   exp_avg_sq   (second moment v)
               Initialize m and v to zeros on the first step.
            3. Update moments with the current gradient g:
                   m <- beta1 * m + (1 - beta1) * g
                   v <- beta2 * v + (1 - beta2) * g * g
            4. Bias correction:
                   m_hat = m / (1 - beta1 ** t)
                   v_hat = v / (1 - beta2 ** t)
            5. Parameter update:
                   p <- p - lr * m_hat / (sqrt(v_hat) + eps)

        Notes:
            - Access hyperparameters via each param_group.
            - Store per-parameter state in self.state[p].
            - If closure is provided, call it to recompute the loss and return it.
        """
        raise NotImplementedError
