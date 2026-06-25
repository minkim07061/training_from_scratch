import torch

from scratch_transformer.attention import KVCache
from scratch_transformer.config import TransformerConfig
from scratch_transformer.model import TransformerLM


def tiny_config(n_layers: int = 2) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=32,
        context_length=8,
        d_model=8,
        n_heads=2,
        n_layers=n_layers,
        d_ff=16,
        dropout=0.0,
    )


def test_transformer_lm_returns_logits_and_no_cache_without_caches() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    model = TransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits, caches = model(input_ids)

    assert logits.shape == (2, 5, config.vocab_size)
    assert logits.dtype == model.token_embedding.weight.dtype
    assert caches is None


def test_transformer_lm_updates_one_cache_per_layer() -> None:
    torch.manual_seed(0)
    config = tiny_config(n_layers=3)
    model = TransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))
    caches = model.init_caches(batch_size=2, device=input_ids.device)

    logits, returned_caches = model(input_ids, caches=caches)

    assert logits.shape == (2, 4, config.vocab_size)
    assert returned_caches is caches
    assert len(returned_caches) == config.n_layers
    for cache in returned_caches:
        assert cache.length == 4
        assert cache.keys is not None
        assert cache.values is not None
        assert cache.keys.shape == (2, config.n_heads, 4, config.head_dim)
        assert cache.values.shape == (2, config.n_heads, 4, config.head_dim)


def test_transformer_lm_cached_single_token_steps_match_full_sequence() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    model = TransformerLM(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 5))

    full_logits, _ = model(input_ids)

    caches = model.init_caches(batch_size=1, device=input_ids.device)
    step_logits = []
    for position in range(input_ids.shape[1]):
        logits, returned_caches = model(input_ids[:, position : position + 1], caches=caches)
        assert returned_caches is caches
        step_logits.append(logits)

    cached_logits = torch.cat(step_logits, dim=1)
    torch.testing.assert_close(cached_logits, full_logits)


def test_transformer_lm_zeroed_blocks_reduce_to_embedding_norm_and_head() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    model = TransformerLM(config)
    for block in model.blocks:
        for parameter in block.attn.parameters():
            parameter.data.zero_()
        for parameter in block.mlp.parameters():
            parameter.data.zero_()
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits, caches = model(input_ids)

    hidden = model.token_embedding(input_ids)
    expected = model.lm_head(model.final_norm(hidden))
    assert caches is None
    torch.testing.assert_close(logits, expected)


def test_transformer_lm_backward_computes_gradients() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    model = TransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits, _ = model(input_ids)
    loss = logits.square().mean()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.shape == parameter.shape
