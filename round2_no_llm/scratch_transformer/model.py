"""Round 2 transformer model scaffold.

Goal:
    Rebuild TransformerBlock and TransformerLM wiring after low-level modules
    are working.

Suggested tests:
    python3 -m pytest tests/test_transformer_block.py
    python3 -m pytest tests/test_transformer_lm.py
"""

import torch
from torch import nn

from scratch_transformer.attention import KVCache, MultiHeadCausalSelfAttention
from scratch_transformer.config import TransformerConfig
from scratch_transformer.mlp import SwiGLU
from scratch_transformer.norm import RMSNorm


class TransformerBlock(nn.Module):
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
        """Run one pre-norm decoder block.

        Implementation checkpoints:
            1. Normalize x before attention.
            2. Run attention with optional RoPE/cache.
            3. Add attention residual.
            4. Normalize updated x before MLP.
            5. Run SwiGLU.
            6. Add MLP residual.
            7. Return x and cache.
        """
        raise NotImplementedError


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def init_caches(self, batch_size: int, device: torch.device | str | None = None) -> list[KVCache]:
        """Create one cache object per layer."""
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        """Return next-token logits.

        Implementation checkpoints:
            1. Embed input_ids.
            2. Determine past_len from caches when provided.
            3. Build RoPE cache with length past_len + seq_len.
            4. Iterate over self.blocks manually; ModuleList is not callable.
            5. Pass each layer its own cache.
            6. Store updated cache back into caches when cache mode is active.
            7. Apply final RMSNorm.
            8. Project to vocabulary logits.
            9. Return (logits, caches).

        Output shape:
            (batch, seq_len, vocab_size)
        """
        raise NotImplementedError

