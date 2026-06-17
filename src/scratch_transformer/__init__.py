"""Learning scaffold for implementing a decoder-only transformer from scratch."""

from scratch_transformer.attention import KVCache, MultiHeadCausalSelfAttention, build_causal_mask
from scratch_transformer.config import TransformerConfig
from scratch_transformer.generate import generate
from scratch_transformer.mlp import SwiGLU
from scratch_transformer.model import TransformerBlock, TransformerLM
from scratch_transformer.norm import RMSNorm
from scratch_transformer.rope import apply_rope, build_rope_cache
from scratch_transformer.tokenizer import BPETokenizer
from scratch_transformer.train import train_epoch, train_step

__all__ = [
    "BPETokenizer",
    "KVCache",
    "MultiHeadCausalSelfAttention",
    "RMSNorm",
    "SwiGLU",
    "TransformerBlock",
    "TransformerConfig",
    "TransformerLM",
    "apply_rope",
    "build_causal_mask",
    "build_rope_cache",
    "generate",
    "train_epoch",
    "train_step",
]
