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
from scratch_transformer.rope import apply_rope

import math
import torch.nn.functional as F



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
    past = torch.ones(seq_len, past_len)
    current = torch.ones(seq_len, seq_len)
    current = torch.tril(current, diagonal=0)
    mask = torch.cat([past, current], dim=1).bool()
    return mask


@dataclass
class KVCache:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """Return cached sequence length."""
        if self.keys is not None:
            return self.keys.shape[-2]
        return 0

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values along sequence dimension.

        Expected shape:
            (batch, n_heads, seq_len, head_dim)

        Implementation checkpoints:
            1. If empty, store incoming tensors.
            2. Otherwise concatenate along dim=-2.
            3. Return the full cached keys and values.
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat([self.keys, keys], dim=-2)
            self.values = torch.cat([self.values, values], dim=-2)
        return self.keys, self.values


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
        batch, seq_len, d_model = x.shape
        n_heads = self.config.n_heads
        head_dim = d_model // n_heads
        # apply wq, wk, wv all at the same time.
        qkv = self.qkv(x)
        q = qkv[:, :, :d_model] # shape (batch, seq_len, d_model)
        k = qkv[:, :, d_model:2*d_model]
        v = qkv[:, :, 2*d_model:]

        q = q.reshape(batch, seq_len, n_heads, head_dim).transpose(1, 2) # shape (batch, n_heads, seq_len, head_dim)
        k = k.reshape(batch, seq_len, n_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, n_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        past_len = 0
        if cache:
            past_len = cache.length
        if cos:
            q = apply_rope(q, cos, sin, offset=past_len)
            k = apply_rope(k, cos, sin, offset=past_len)

        # append k and v to cache.
        if cache:
            k, v = cache.append(k, v) # (batch, n_heads, seq_len+past_len, head_dim)
        
        # compute raw attention scores
        scores = q @ torch.transpose(k, -2, -1) / math.sqrt(head_dim) # shape: (batch, n_heads, seq_len, seq_len+past_len)

        # apply mask
        mask = build_causal_mask(seq_len, past_len=past_len)
        scores = scores.masked_fill(~mask, float('-inf'))
        softmax = F.softmax(scores, -1) # (batch, n_heads, seq_len, seq_len+past_len)

        # multiply attention weights by values
        attention_values = softmax @ v # shape (batch, n_heads, seq_len, head_dim)

        # Merge heads back to (batch, seq_len, d_model).
        attention_values = torch.transpose(attention_values, 1, 2).reshape(batch, seq_len, d_model)
        
        # Apply output projection and dropout.
        out = self.dropout(self.out_proj(attention_values))
        return out, cache






