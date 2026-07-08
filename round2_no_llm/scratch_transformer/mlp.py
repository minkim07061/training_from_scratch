"""Round 2 SwiGLU scaffold.

Goal:
    Rebuild the feed-forward sublayer used inside a transformer block.

Suggested test:
    python3 -m pytest tests/test_mlp.py
"""

import torch
from torch import nn

from scratch_transformer.config import TransformerConfig


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU.

        Implementation checkpoints:
            1. Project x through gate_proj.
            2. Apply SiLU to the gate.
            3. Project x through up_proj.
            4. Multiply gate and candidate elementwise.
            5. Project back to d_model with down_proj.
            6. Apply dropout to the output.

        Shape invariant:
            Input and output should both be (batch, seq_len, d_model).
        """
        gate_x = self.gate_proj(x) # (batch, seq_len, d_ff)
        gate_silu = gate_x / (1 + torch.exp(-1.0 * gate_x)) # (batch, seq_len, d_ff)
        up_proj = self.up_proj(x) # (batch, seq_len, d_ff)
        mult = gate_silu * up_proj # (batch, seq_len, d_ff)
        down_proj = self.down_proj(mult) # (batch, seq_len, d_model)
        return self.dropout(down_proj) # (batch, seq_len, d_model)

