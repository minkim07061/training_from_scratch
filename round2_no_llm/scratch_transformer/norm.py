"""Round 2 RMSNorm scaffold.

Goal:
    Rebuild root-mean-square normalization.

Suggested test:
    python3 -m pytest tests/test_norm.py
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize by RMS over the last dimension and apply learned scale."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Implementation checkpoints:
            1. Square x.
            2. Mean over last dimension with keepdim=True.
            3. Add eps.
            4. Square root.
            5. Divide x by RMS.
            6. Multiply by learned weight.

        Important:
            - RMSNorm does not subtract the mean.
            - Output shape and dtype should match input.
        """
        raise NotImplementedError

