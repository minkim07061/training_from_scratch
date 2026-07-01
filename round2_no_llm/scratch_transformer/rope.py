"""Round 2 RoPE scaffold.

Goal:
    Rebuild rotary positional embeddings for query/key tensors.

Shapes:
    x:   (batch, n_heads, seq_len, head_dim)
    cos: (cache_seq_len, head_dim // 2)
    sin: (cache_seq_len, head_dim // 2)

Suggested tests:
    python3 -m pytest tests/test_rope.py
"""

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cosine and sine tables.

    Implementation checkpoints:
        1. Validate head_dim is even.
        2. Create even dimension indices: 0, 2, 4, ...
        3. Compute inverse frequencies using base and head_dim.
        4. Create position indices 0..seq_len-1.
        5. Build angle matrix with shape (seq_len, head_dim // 2).
        6. Return cos(angle), sin(angle) on requested device/dtype.

    Invariant:
        Position 0 should have cos=1 and sin=0 for every pair.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim is not even")
    even_dim = torch.arange(0, head_dim, 2, dtype=dtype, device=device)
    inv_frequencies = 1.0 / (base ** (even_dim / head_dim))
    position_idx = torch.arange(0, seq_len, dtype=dtype, device=device)
    angle_matrix = torch.outer(position_idx, inv_frequencies)
    return torch.cos(angle_matrix), torch.sin(angle_matrix)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    offset: int = 0,
) -> torch.Tensor:
    """Rotate query/key pairs using cached cos/sin tables.

    Implementation checkpoints:
        1. Infer seq_len from x.shape[-2].
        2. Slice cos/sin from offset to offset + seq_len.
        3. Reshape cos/sin for broadcasting over batch and heads.
        4. Split x into even and odd coordinates.
        5. Apply pair rotation.
        6. Interleave rotated even/odd coordinates back into original shape.

    Rotation rule for each pair (a, b):
        rotated_even = a * cos - b * sin
        rotated_odd  = a * sin + b * cos

    Invariant:
        Pairwise norm should be preserved.
    """
    batch, n_heads, seq_len, head_dim = x.shape
    cos = cos[offset:offset + seq_len]
    sin = sin[offset:offset + seq_len]
    cos = cos.reshape(1, 1, seq_len, -1)
    sin = sin.reshape(1, 1, seq_len, -1)
    x_even = x[:,:,:,0::2]
    x_odd = x[:,:,:,1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even*sin + x_odd * cos
    x[:,:,:,0::2] = rotated_even
    x[:,:,:,1::2] = rotated_odd
    return x

