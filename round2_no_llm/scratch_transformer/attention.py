"""Round 2 causal attention scaffold.

Goal:
    Rebuild causal masking, KV cache append, and multi-head causal self-attention.

Suggested tests:
    python3 -m pytest tests/test_kv_cache.py
    python3 -m pytest tests/test_attention.py
"""

from dataclasses import dataclass

import torch
from torch import nn

from scratch_transformer.config import TransformerConfig


def build_causal_mask(
    seq_len: int,
    *,
    past_len: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return boolean causal visibility mask.

    Shape:
        (seq_len, past_len + seq_len)

    Semantics:
        True means a query position may attend to a key position.

    Implementation checkpoints:
        1. Cached past positions are visible to every current query.
        2. Current positions use a lower-triangular mask.
        3. Concatenate past and current masks along key dimension.
    """
    raise NotImplementedError


@dataclass
class KVCache:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """Return cached sequence length."""
        raise NotImplementedError

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values along sequence dimension.

        Expected shape:
            (batch, n_heads, seq_len, head_dim)

        Implementation checkpoints:
            1. If empty, store incoming tensors.
            2. Otherwise concatenate along dim=-2.
            3. Return the full cached keys and values.
        """
        raise NotImplementedError


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Apply masked multi-head self-attention.

        Input shape:
            x: (batch, seq_len, d_model)

        Implementation checkpoints:
            1. Project x to combined qkv.
            2. Split q, k, v.
            3. Reshape each to (batch, n_heads, seq_len, head_dim).
            4. Compute past_len before appending to cache.
            5. Apply RoPE to q and k when cos/sin are provided, using offset=past_len.
            6. Append k/v to cache if cache is provided.
            7. Compute scaled dot-product scores.
            8. Apply causal mask.
            9. Softmax over key dimension.
            10. Multiply attention weights by values.
            11. Merge heads back to (batch, seq_len, d_model).
            12. Apply output projection and dropout.
            13. Return output and cache.

        Common bugs:
            - Using / instead of // for head_dim.
            - Forgetting to return cache.
            - Dropping the batch dimension when merging heads.
            - Applying RoPE with offset 0 during cached generation.
        """
        raise NotImplementedError

