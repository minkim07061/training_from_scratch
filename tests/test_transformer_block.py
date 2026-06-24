import torch

from scratch_transformer.attention import KVCache
from scratch_transformer.config import TransformerConfig
from scratch_transformer.model import TransformerBlock
from scratch_transformer.rope import build_rope_cache


def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=32,
        context_length=8,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )


def test_transformer_block_returns_hidden_states_and_no_cache_when_cache_not_supplied() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    block = TransformerBlock(config)
    x = torch.randn(2, 3, config.d_model)

    y, cache = block(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert cache is None


def test_transformer_block_matches_manual_pre_norm_residual_formula() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    block = TransformerBlock(config)
    x = torch.randn(2, 3, config.d_model)

    actual, cache = block(x)

    attn_out, manual_cache = block.attn(block.attn_norm(x))
    expected = x + attn_out
    expected = expected + block.mlp(block.mlp_norm(expected))

    assert cache is None
    assert manual_cache is None
    torch.testing.assert_close(actual, expected)


def test_transformer_block_updates_supplied_cache() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    block = TransformerBlock(config)
    cache = KVCache()
    x = torch.randn(2, 3, config.d_model)

    y, new_cache = block(x, cache=cache)

    assert y.shape == x.shape
    assert new_cache is cache
    assert cache.length == 3
    assert cache.keys is not None
    assert cache.values is not None
    assert cache.keys.shape == (2, config.n_heads, 3, config.head_dim)
    assert cache.values.shape == (2, config.n_heads, 3, config.head_dim)


def test_transformer_block_accepts_rope_cache() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    block = TransformerBlock(config)
    x = torch.randn(2, 3, config.d_model)
    cos, sin = build_rope_cache(seq_len=3, head_dim=config.head_dim, dtype=x.dtype)

    y, cache = block(x, cos=cos, sin=sin)

    assert y.shape == x.shape
    assert cache is None


def test_transformer_block_zeroed_sublayers_preserve_residual() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    block = TransformerBlock(config)
    for parameter in block.attn.parameters():
        parameter.data.zero_()
    for parameter in block.mlp.parameters():
        parameter.data.zero_()
    x = torch.randn(2, 3, config.d_model)

    y, _ = block(x)

    torch.testing.assert_close(y, x)


def test_transformer_block_backward_computes_gradients() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    block = TransformerBlock(config)
    x = torch.randn(2, 3, config.d_model, requires_grad=True)

    y, _ = block(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert parameter.grad.shape == parameter.shape
