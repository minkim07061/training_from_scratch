"""Decoder-only transformer model scaffold."""

import torch
from torch import nn

from scratch_transformer.attention import KVCache, MultiHeadCausalSelfAttention
from scratch_transformer.config import TransformerConfig
from scratch_transformer.mlp import SwiGLU
from scratch_transformer.norm import RMSNorm


class TransformerBlock(nn.Module):
    """One pre-norm decoder block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = MultiHeadCausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Run attention and MLP residual paths.

        TODO:
        - Apply attn_norm, attention, and residual addition.
        - Apply mlp_norm, SwiGLU, and residual addition.
        - Return the updated hidden states and cache.
        """
        raise NotImplementedError("TODO: implement TransformerBlock.forward")


class TransformerLM(nn.Module):
    """Tiny decoder-only language model scaffold."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def init_caches(self, batch_size: int, device: torch.device | str | None = None) -> list[KVCache]:
        """Create empty caches, one per transformer block.

        The batch_size/device arguments are included so your implementation can
        later preallocate cache tensors if you choose that route.
        """
        return [KVCache() for _ in range(self.config.n_layers)]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        """Return logits for next-token prediction.

        Args:
            input_ids: Token ids with shape ``(batch, seq_len)``.
            caches: Optional per-layer KV caches for generation.

        TODO:
        - Embed tokens.
        - Build RoPE cache for the sequence length plus any cached positions.
        - Run each TransformerBlock, passing the matching layer cache.
        - Normalize final hidden states and project to vocab logits.
        - Return logits with shape ``(batch, seq_len, vocab_size)``.
        """
        raise NotImplementedError("TODO: implement TransformerLM.forward")
