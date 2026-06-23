"""Causal self-attention and KV-cache scaffolds."""

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
    """Build a boolean mask for causal attention.

    Expected shape: ``(seq_len, past_len + seq_len)`` where True means the key
    position is visible to the query position.

    TODO:
    - Allow every query to see all cached positions ``[:past_len]``.
    - In the current block, allow each query position to see itself and earlier
      positions only.
    - Return a boolean tensor on the requested device.
    """
    cache_mask_past = torch.ones((seq_len, past_len), dtype=torch.bool, device=device)
    cache_mask_curr = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    cache_mask_curr = torch.tril(cache_mask_curr)
    final_cache_mask = torch.cat([cache_mask_past, cache_mask_curr], dim=1)
    return final_cache_mask

@dataclass
class KVCache:
    """Per-layer key/value cache used during autoregressive generation."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    @property
    def length(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.shape[-2]

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value states and return the full cached tensors.

        Expected key/value shape: ``(batch, n_heads, seq_len, head_dim)``.

        TODO:
        - If the cache is empty, store the incoming keys and values.
        - Otherwise concatenate along the sequence dimension.
        - Keep keys and values on the same device/dtype as the incoming tensors.
        """
        raise NotImplementedError("TODO: implement KVCache.append")


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head masked self-attention for a decoder-only transformer."""

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
        """Apply causal self-attention.

        Args:
            x: Input tensor with shape ``(batch, seq_len, d_model)``.
            cos/sin: Optional RoPE caches. Apply them to q and k when provided.
            cache: Optional KV cache. Use it during autoregressive generation.

        TODO:
        - Project x to q, k, v and split into heads.
        - Apply RoPE to q and k when cos/sin are supplied.
        - Append k/v to cache when cache is supplied.
        - Build and apply a causal mask before softmax.
        - Compute attention weights, weighted values, and output projection.
        """
        raise NotImplementedError("TODO: implement MultiHeadCausalSelfAttention.forward")
