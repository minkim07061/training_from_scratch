"""RMSNorm scaffold."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square layer normalization.

    RMSNorm normalizes by the root mean square over the last dimension and then
    applies a learned scale. It does not subtract the mean.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over the final dimension.

        TODO:
        - Compute rms = sqrt(mean(x ** 2, dim=-1, keepdim=True) + eps).
        - Divide x by rms.
        - Multiply by self.weight with broadcasting.
        """
        raise NotImplementedError("TODO: implement RMSNorm.forward")
