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
    raise NotImplementedError("TODO: implement RoPE cache construction")


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
    raise NotImplementedError("TODO: implement RoPE application")
