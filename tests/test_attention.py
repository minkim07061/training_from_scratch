import torch

from scratch_transformer.attention import KVCache, MultiHeadCausalSelfAttention
from scratch_transformer.config import TransformerConfig


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


def test_attention_returns_hidden_states_and_no_cache_when_cache_not_supplied() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    attention = MultiHeadCausalSelfAttention(config)
    x = torch.randn(2, 3, config.d_model)

    y, cache = attention(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert cache is None


def test_attention_updates_supplied_kv_cache() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    attention = MultiHeadCausalSelfAttention(config)
    cache = KVCache()
    x = torch.randn(2, 3, config.d_model)

    y, new_cache = attention(x, cache=cache)

    assert y.shape == x.shape
    assert new_cache is cache
    assert cache.length == 3
    assert cache.keys is not None
    assert cache.values is not None
    assert cache.keys.shape == (2, config.n_heads, 3, config.head_dim)
    assert cache.values.shape == (2, config.n_heads, 3, config.head_dim)


def test_attention_is_causal_so_future_tokens_do_not_change_prefix_outputs() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    attention = MultiHeadCausalSelfAttention(config)
    attention.eval()
    x = torch.randn(1, 4, config.d_model)
    changed_future = x.clone()
    changed_future[:, 2:] = torch.randn_like(changed_future[:, 2:]) * 10.0

    y, _ = attention(x)
    y_changed, _ = attention(changed_future)

    torch.testing.assert_close(y[:, :2], y_changed[:, :2])


def test_attention_cached_single_token_steps_match_full_sequence() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    attention = MultiHeadCausalSelfAttention(config)
    attention.eval()
    x = torch.randn(1, 4, config.d_model)

    full_output, _ = attention(x)

    cache = KVCache()
    step_outputs = []
    for position in range(x.shape[1]):
        step_output, returned_cache = attention(x[:, position : position + 1], cache=cache)
        assert returned_cache is cache
        step_outputs.append(step_output)

    cached_output = torch.cat(step_outputs, dim=1)
    torch.testing.assert_close(cached_output, full_output)
    assert cache.length == x.shape[1]


def test_attention_backward_computes_input_and_parameter_gradients() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    attention = MultiHeadCausalSelfAttention(config)
    x = torch.randn(2, 3, config.d_model, requires_grad=True)

    y, _ = attention(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    for parameter in attention.parameters():
        assert parameter.grad is not None
        assert parameter.grad.shape == parameter.shape
