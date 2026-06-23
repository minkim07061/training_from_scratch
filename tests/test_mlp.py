import torch
from torch import nn

from scratch_transformer.config import TransformerConfig
from scratch_transformer.mlp import SwiGLU


def tiny_config(dropout: float = 0.0) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=32,
        context_length=8,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=dropout,
    )


def test_swiglu_constructs_expected_projection_shapes() -> None:
    config = tiny_config()
    layer = SwiGLU(config)

    assert layer.gate_proj.weight.shape == (config.d_ff, config.d_model)
    assert layer.up_proj.weight.shape == (config.d_ff, config.d_model)
    assert layer.down_proj.weight.shape == (config.d_model, config.d_ff)


def test_swiglu_matches_reference_formula() -> None:
    config = tiny_config()
    layer = SwiGLU(config)
    x = torch.randn(2, 3, config.d_model)

    actual = layer(x)

    hidden = nn.functional.silu(layer.gate_proj(x)) * layer.up_proj(x)
    expected = layer.down_proj(hidden)
    torch.testing.assert_close(actual, expected)


def test_swiglu_preserves_batch_sequence_and_model_dimensions() -> None:
    config = tiny_config()
    layer = SwiGLU(config)
    x = torch.randn(4, 5, config.d_model)

    y = layer(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_swiglu_zero_input_without_bias_returns_zero() -> None:
    config = tiny_config()
    layer = SwiGLU(config)
    x = torch.zeros(2, 3, config.d_model)

    y = layer(x)

    torch.testing.assert_close(y, torch.zeros_like(x))


def test_swiglu_parameters_and_input_receive_gradients() -> None:
    config = tiny_config()
    layer = SwiGLU(config)
    x = torch.randn(2, 3, config.d_model, requires_grad=True)

    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert parameter.grad.shape == parameter.shape


