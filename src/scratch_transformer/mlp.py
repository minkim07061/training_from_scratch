"""SwiGLU feed-forward network scaffold."""

import torch
from torch import nn

from scratch_transformer.config import TransformerConfig


class SwiGLU(nn.Module):
    """SwiGLU MLP used in many modern decoder-only transformers."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU feed-forward transformation.

        TODO:
        - Compute gate = silu(gate_proj(x)).
        - Compute candidate = up_proj(x).
        - Multiply gate * candidate elementwise.
        - Project back to d_model with down_proj.
        """
        silu_layer = nn.SiLU()
        gate = silu_layer(self.gate_proj(x))
        candidate = self.up_proj(x)
        mult = gate * candidate
        return self.dropout(self.down_proj(mult))
