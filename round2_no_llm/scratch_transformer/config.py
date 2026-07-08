"""Round 2 config scaffold.

This file intentionally includes only simple configuration structure and
validation hints. Rebuild or simplify as you prefer during the exercise.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 256
    context_length: int = 128
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 4 * 128
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate config invariants.

        Suggested checks:
            - vocab_size > 0
            - context_length > 0
            - d_model > 0
            - n_layers > 0
            - n_heads > 0
            - d_model divisible by n_heads
            - head_dim even for RoPE
            - d_ff > 0
            - dropout in [0, 1)
        """
        assert(self.vocab_size > 0)
        assert(self.context_length > 0)
        assert(self.d_model > 0)
        assert(self.n_layers > 0)
        assert(self.n_heads > 0)
        assert(self.d_model % self.n_heads == 0)
        assert(self.head_dim % 2 == 0)
        assert(self.d_ff > 0)
        assert(self.dropout >=0 and self.dropout < 1)


    @property
    def head_dim(self) -> int:
        """Return per-head hidden dimension."""
        return self.d_model // self.n_heads

