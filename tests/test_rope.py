import torch

from scratch_transformer.rope import apply_rope, build_rope_cache


def test_build_rope_cache_returns_expected_shape_dtype_and_values() -> None:
    seq_len = 4
    head_dim = 6
    dtype = torch.float64

    cos, sin = build_rope_cache(seq_len, head_dim, base=100.0, dtype=dtype)

    inv_freq = 1.0 / (100.0 ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, dtype=dtype)
    expected_angles = torch.outer(positions, inv_freq)

    assert cos.shape == (seq_len, head_dim // 2)
    assert sin.shape == (seq_len, head_dim // 2)
    assert cos.dtype == dtype
    assert sin.dtype == dtype
    torch.testing.assert_close(cos, torch.cos(expected_angles))
    torch.testing.assert_close(sin, torch.sin(expected_angles))


def test_build_rope_cache_position_zero_is_identity_rotation() -> None:
    cos, sin = build_rope_cache(seq_len=3, head_dim=4)

    torch.testing.assert_close(cos[0], torch.ones(2))
    torch.testing.assert_close(sin[0], torch.zeros(2))


def test_apply_rope_position_zero_leaves_values_unchanged() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos = torch.ones(1, 2)
    sin = torch.zeros(1, 2)

    y = apply_rope(x, cos, sin)

    torch.testing.assert_close(y, x)


def test_apply_rope_matches_manual_pair_rotation() -> None:
    x = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            ]
        ]
    )
    cos = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    sin = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    y = apply_rope(x, cos, sin)

    expected = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [-6.0, 5.0, -8.0, 7.0],
                ]
            ]
        ]
    )
    torch.testing.assert_close(y, expected)


def test_apply_rope_offset_uses_later_cache_positions() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    sin = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    y = apply_rope(x, cos, sin, offset=1)

    torch.testing.assert_close(y, torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]]))


def test_apply_rope_preserves_pairwise_norms() -> None:
    x = torch.randn(2, 3, 5, 8)
    cos, sin = build_rope_cache(seq_len=5, head_dim=8, dtype=x.dtype)

    y = apply_rope(x, cos, sin)

    assert y.shape == x.shape
    torch.testing.assert_close(
        y[..., ::2].square() + y[..., 1::2].square(),
        x[..., ::2].square() + x[..., 1::2].square(),
    )
