"""Configuration objects shared by the transformer components."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters for a small decoder-only transformer.

    These defaults are intentionally tiny so unit tests and first experiments can
    run on a laptop CPU. Increase them only after the implementation is correct.
    """

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
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
