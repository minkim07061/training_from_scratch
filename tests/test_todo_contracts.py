import pytest
import torch
from torch import nn

from scratch_transformer import (
    BPETokenizer,
    KVCache,
    MultiHeadCausalSelfAttention,
    RMSNorm,
    SwiGLU,
    TransformerConfig,
    TransformerLM,
    apply_rope,
    build_causal_mask,
    build_rope_cache,
    generate,
    train_step,
)


@pytest.mark.xfail(raises=NotImplementedError, reason="BPE tokenizer is a learning TODO")
def test_bpe_tokenizer_round_trips_text() -> None:
    tokenizer = BPETokenizer.train(["hello hello", "hello world"], vocab_size=300)

    token_ids = tokenizer.encode("hello world")

    assert tokenizer.decode(token_ids) == "hello world"


@pytest.mark.xfail(raises=NotImplementedError, reason="RoPE is a learning TODO")
def test_rope_preserves_shape_and_pairwise_norms() -> None:
    x = torch.randn(2, 4, 3, 8)
    cos, sin = build_rope_cache(seq_len=3, head_dim=8, dtype=x.dtype)

    y = apply_rope(x, cos, sin)

    assert y.shape == x.shape
    torch.testing.assert_close(
        y[..., ::2].square() + y[..., 1::2].square(),
        x[..., ::2].square() + x[..., 1::2].square(),
    )


@pytest.mark.xfail(raises=NotImplementedError, reason="RMSNorm is a learning TODO")
def test_rms_norm_matches_reference_formula() -> None:
    layer = RMSNorm(d_model=4, eps=1e-5)
    layer.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = torch.randn(2, 3, 4)

    actual = layer(x)

    expected = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + layer.eps)
    expected = expected * layer.weight
    torch.testing.assert_close(actual, expected)


@pytest.mark.xfail(raises=NotImplementedError, reason="causal masking is a learning TODO")
def test_causal_mask_includes_cache_and_blocks_future_tokens() -> None:
    mask = build_causal_mask(seq_len=3, past_len=2)

    expected = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    torch.testing.assert_close(mask, expected)


@pytest.mark.xfail(raises=NotImplementedError, reason="KV cache append is a learning TODO")
def test_kv_cache_appends_along_sequence_axis() -> None:
    cache = KVCache()
    first_keys = torch.randn(2, 4, 1, 8)
    first_values = torch.randn(2, 4, 1, 8)
    second_keys = torch.randn(2, 4, 2, 8)
    second_values = torch.randn(2, 4, 2, 8)

    keys, values = cache.append(first_keys, first_values)
    keys, values = cache.append(second_keys, second_values)

    assert cache.length == 3
    torch.testing.assert_close(keys, torch.cat([first_keys, second_keys], dim=-2))
    torch.testing.assert_close(values, torch.cat([first_values, second_values], dim=-2))


@pytest.mark.xfail(raises=NotImplementedError, reason="attention is a learning TODO")
def test_attention_returns_hidden_states_and_updates_cache() -> None:
    config = TransformerConfig(vocab_size=32, context_length=8, d_model=16, n_heads=4, d_ff=32)
    attn = MultiHeadCausalSelfAttention(config)
    cache = KVCache()
    x = torch.randn(2, 3, config.d_model)

    y, new_cache = attn(x, cache=cache)

    assert y.shape == x.shape
    assert new_cache is cache
    assert new_cache.length == 3


@pytest.mark.xfail(raises=NotImplementedError, reason="SwiGLU is a learning TODO")
def test_swiglu_matches_reference_formula() -> None:
    config = TransformerConfig(vocab_size=32, d_model=8, n_heads=2, d_ff=16)
    layer = SwiGLU(config)
    x = torch.randn(2, 3, config.d_model)

    actual = layer(x)

    hidden = nn.functional.silu(layer.gate_proj(x)) * layer.up_proj(x)
    expected = layer.down_proj(hidden)
    torch.testing.assert_close(actual, expected)


@pytest.mark.xfail(raises=NotImplementedError, reason="TransformerLM forward is a learning TODO")
def test_transformer_lm_forward_returns_logits() -> None:
    config = TransformerConfig(vocab_size=32, context_length=8, d_model=16, n_heads=4, d_ff=32)
    model = TransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits, caches = model(input_ids)

    assert logits.shape == (2, 5, config.vocab_size)
    assert caches is None


class ToyNextTokenModel(nn.Module):
    """Tiny duck-typed model for generation and training-loop tests."""

    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.config = TransformerConfig(vocab_size=vocab_size, d_model=8, n_heads=2, d_ff=16)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def init_caches(self, batch_size: int, device: torch.device | str | None = None) -> list[KVCache]:
        return [KVCache()]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.config.vocab_size, device=input_ids.device)
        next_ids = (input_ids + 1) % self.config.vocab_size
        index = next_ids.unsqueeze(-1)
        source = torch.ones_like(index, dtype=logits.dtype) * (self.scale * 5.0)
        logits.scatter_(-1, index, source)
        return logits, caches


@pytest.mark.xfail(raises=NotImplementedError, reason="generation is a learning TODO")
def test_generate_appends_requested_number_of_tokens() -> None:
    model = ToyNextTokenModel(vocab_size=5)
    prompt = torch.tensor([[0, 1]])

    output = generate(model, prompt, max_new_tokens=3, temperature=1e-6, top_k=1)

    assert output.shape == (1, 5)
    torch.testing.assert_close(output, torch.tensor([[0, 1, 2, 3, 4]]))


@pytest.mark.xfail(raises=NotImplementedError, reason="training loop is a learning TODO")
def test_train_step_updates_parameters_and_returns_detached_loss() -> None:
    model = ToyNextTokenModel(vocab_size=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])

    before = model.scale.detach().clone()
    loss = train_step(model, optimizer, inputs, targets)

    assert loss.ndim == 0
    assert not loss.requires_grad
    assert not torch.equal(model.scale.detach(), before)
