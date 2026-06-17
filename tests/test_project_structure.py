import pytest
import torch

from scratch_transformer import (
    BPETokenizer,
    KVCache,
    MultiHeadCausalSelfAttention,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    TransformerConfig,
    TransformerLM,
    build_causal_mask,
    generate,
    train_epoch,
    train_step,
)
from scratch_transformer.train import next_token_loss


def test_public_imports_are_available() -> None:
    assert BPETokenizer is not None
    assert KVCache is not None
    assert MultiHeadCausalSelfAttention is not None
    assert RMSNorm is not None
    assert SwiGLU is not None
    assert TransformerBlock is not None
    assert TransformerLM is not None
    assert build_causal_mask is not None
    assert generate is not None
    assert train_epoch is not None
    assert train_step is not None


def test_config_validates_attention_dimensions() -> None:
    config = TransformerConfig(vocab_size=32, d_model=16, n_heads=4, d_ff=32)
    assert config.head_dim == 4

    with pytest.raises(ValueError, match="d_model must be divisible"):
        TransformerConfig(vocab_size=32, d_model=10, n_heads=4, d_ff=32)

    with pytest.raises(ValueError, match="head_dim must be even"):
        TransformerConfig(vocab_size=32, d_model=12, n_heads=4, d_ff=32)


def test_modules_can_be_constructed_from_config() -> None:
    config = TransformerConfig(
        vocab_size=32,
        context_length=8,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
    )

    assert isinstance(RMSNorm(config.d_model), RMSNorm)
    assert isinstance(SwiGLU(config), SwiGLU)
    assert isinstance(MultiHeadCausalSelfAttention(config), MultiHeadCausalSelfAttention)
    assert isinstance(TransformerBlock(config), TransformerBlock)
    assert isinstance(TransformerLM(config), TransformerLM)


def test_next_token_loss_returns_scalar() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    targets = torch.tensor([[0, 1]])

    loss = next_token_loss(logits, targets)

    assert loss.ndim == 0
    assert loss.item() < 0.2
