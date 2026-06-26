"""Causal self-attention and KV-cache scaffolds."""

from dataclasses import dataclass

import torch
from torch import nn

from scratch_transformer.config import TransformerConfig
from scratch_transformer.rope import apply_rope
import math


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
        if self.keys is None or self.values is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat([self.keys, keys], dim=-2)
            self.values = torch.cat([self.values, values], dim=-2)
        return (self.keys, self.values)


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
        batch, seq_len, d_model = x.shape
        num_heads = self.config.n_heads
        head_dim = d_model // num_heads
        qkv = self.qkv(x)
        q = qkv[:,:,:d_model].reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2) # batch, num_heads, seq_len, head_dim
        k = qkv[:,:,d_model:2*d_model].reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2) # batch, num_heads, seq_len, head_dim
        v = qkv[:,:,d_model*2:d_model*3].reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2) # batch, num_heads, seq_len, head_dim
        
        past_len = cache.length if cache is not None else 0
        if cache:
            offset = cache.length
        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin, offset=past_len)
            k = apply_rope(k, cos, sin, offset=past_len)
    
        
        if cache is not None:
            k, v = cache.append(k, v) # (batch, num_heads, seq_len+past_len, head_dim)
    
        mask = build_causal_mask(seq_len, past_len=past_len)

        scores = q @ k.transpose(2, 3) # shape (batch, num_heads, seq_len, seq_len+past_len)
        scores = scores / math.sqrt(head_dim)
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1) # (batch, num_heads, seq_len, seq_len+past_len)
        weighted_values = attention @ v # (batch, num_heads, seq_len, head_dim)
        weighted_values = weighted_values.transpose(1, 2).reshape(batch, seq_len, d_model)
        return (self.dropout(self.out_proj(weighted_values)), cache)
