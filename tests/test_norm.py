import torch

from scratch_transformer.norm import RMSNorm


def test_rms_norm_matches_reference_formula() -> None:
    layer = RMSNorm(d_model=4, eps=1e-5)
    layer.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]],
            [[0.5, -0.5, 1.5, -1.5], [2.0, 2.0, 2.0, 2.0]],
        ]
    )

    actual = layer(x)

    expected = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + layer.eps)
    expected = expected * layer.weight
    torch.testing.assert_close(actual, expected)


def test_rms_norm_preserves_shape_and_dtype() -> None:
    layer = RMSNorm(d_model=6).double()
    x = torch.randn(2, 3, 6, dtype=torch.float64)

    y = layer(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_rms_norm_zero_input_is_finite_zero() -> None:
    layer = RMSNorm(d_model=4)
    x = torch.zeros(2, 3, 4)

    y = layer(x)

    torch.testing.assert_close(y, torch.zeros_like(x))
    assert torch.isfinite(y).all()


def test_rms_norm_weight_is_learnable_and_receives_gradients() -> None:
    layer = RMSNorm(d_model=4)
    x = torch.randn(2, 3, 4, requires_grad=True)

    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert layer.weight.requires_grad
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    assert x.grad is not None
    assert x.grad.shape == x.shape
