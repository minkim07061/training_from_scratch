"""Rotary positional embedding (RoPE) scaffold."""

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine and sine tables used to rotate query/key tensors.

    Expected output shape for each tensor: ``(seq_len, head_dim // 2)``.

    TODO:
    - Compute inverse frequencies for even dimensions.
    - Build a positions x frequencies table.
    - Return cos(table) and sin(table), cast to the requested dtype/device.
    """
    if head_dim % 2 != 0:
        raise ValueError("Head dimension needs to be even.")    
    # Create inverse frequency vector with the following eq: inv_freq = 1 / base^(dimension_index / D)
    dim_indices = torch.arange(0, head_dim, 2, device=device, dtype=dtype) # shape (head_dim/2, )
    inv_freq = 1 / base ** (dim_indices / head_dim) # shape (head_dim/2, )
    positions = torch.arange(0, seq_len, device=device, dtype=torch.int32)

    # angle[position, pair] = position * inv_freq[pair]
    angles = positions[:, None] * inv_freq[None, :]

    # Apply sin and cosin to create two vectors.
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    
    return (cos, sin)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    offset: int = 0,
) -> torch.Tensor:
    """Apply RoPE to queries or keys.

    Args:
        x: Tensor with shape ``(batch, n_heads, seq_len, head_dim)``.
        cos: Cosine cache from :func:`build_rope_cache`.
        sin: Sine cache from :func:`build_rope_cache`.
        offset: Position offset used during cached autoregressive generation.

    TODO:
    - Split the final dimension into even and odd coordinates.
    - Rotate each pair using the cos/sin values for the current positions.
    - Interleave the rotated coordinates back into the original shape.
    """

    # rotation: [a cos θ - b sin θ, a sin θ + b cos θ]

    # Create cos and sin vertor with offset:offset+T
    # reshape cos and sin vector to be (1, 1, T, D/2)
    seq_len = x.shape[2]
    head_dim = x.shape[3]
    cos = cos[offset:offset+seq_len][None,None,:,:] # shape(1, 1, seq_len, head_dim / 2)
    sin = sin[offset:offset+seq_len][None,None,:,:] # shape(1, 1, seq_len, head_dim / 2)

    # split the x vector with even and odd. apply the rotation and merge.
    x_even = x[:,:,:,::2]
    x_odd = x[:,:,:,1::2]

    x_even_rotate = x_even * cos - x_odd * sin # shape (batch , n_heads, seq_len, head_dim/2)
    x_odd_rotate = x_even * sin + x_odd * cos # shape (batch, n_heasd, seq_len, head_dim/2)

    x_rotated = torch.empty_like(x)
    x_rotated[:,:,:,::2] = x_even_rotate
    x_rotated[:,:,:,1::2] = x_odd_rotate

    # return the final output.
    return x_rotated
