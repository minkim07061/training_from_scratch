import torch
from torch import nn

from scratch_transformer.attention import KVCache
from scratch_transformer.config import TransformerConfig
from scratch_transformer.generate import generate


class ToyNextTokenModel(nn.Module):
    """Small deterministic model that predicts token_id + 1."""

    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.config = TransformerConfig(vocab_size=vocab_size, d_model=8, n_heads=2, d_ff=16)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.forward_input_shapes: list[tuple[int, int]] = []
        self.forward_training_flags: list[bool] = []
        self.forward_caches: list[list[KVCache] | None] = []

    def init_caches(self, batch_size: int, device: torch.device | str | None = None) -> list[KVCache]:
        return [KVCache()]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        self.forward_input_shapes.append(tuple(input_ids.shape))
        self.forward_training_flags.append(self.training)
        self.forward_caches.append(caches)

        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.config.vocab_size, device=input_ids.device)
        next_ids = (input_ids + 1) % self.config.vocab_size
        logits.scatter_(-1, next_ids.unsqueeze(-1), self.scale * 5.0)
        return logits, caches


def test_generate_appends_requested_number_of_tokens() -> None:
    model = ToyNextTokenModel(vocab_size=5)
    prompt = torch.tensor([[0, 1]])

    output = generate(model, prompt, max_new_tokens=3, temperature=1e-6, top_k=1)

    assert output.shape == (1, 5)
    torch.testing.assert_close(output, torch.tensor([[0, 1, 2, 3, 4]]))


def test_generate_zero_new_tokens_returns_prompt() -> None:
    model = ToyNextTokenModel(vocab_size=5)
    prompt = torch.tensor([[0, 1, 2]])

    output = generate(model, prompt, max_new_tokens=0)

    torch.testing.assert_close(output, prompt)
    assert output is not prompt


def test_generate_restores_original_training_mode() -> None:
    model = ToyNextTokenModel(vocab_size=5)
    model.train()
    prompt = torch.tensor([[0, 1]])

    generate(model, prompt, max_new_tokens=2, temperature=1e-6, top_k=1)

    assert model.training is True
    assert model.forward_training_flags
    assert all(flag is False for flag in model.forward_training_flags)


def test_generate_uses_cache_and_feeds_single_tokens_after_prompt() -> None:
    model = ToyNextTokenModel(vocab_size=5)
    prompt = torch.tensor([[0, 1, 2]])

    generate(model, prompt, max_new_tokens=3, temperature=1e-6, top_k=1)

    assert model.forward_input_shapes == [(1, 3), (1, 1), (1, 1)]
    assert all(caches is not None for caches in model.forward_caches)
    assert model.forward_caches[0] is model.forward_caches[1] is model.forward_caches[2]


def test_generate_top_k_one_selects_highest_allowed_token() -> None:
    model = ToyNextTokenModel(vocab_size=5)
    prompt = torch.tensor([[4]])

    output = generate(model, prompt, max_new_tokens=2, temperature=1.0, top_k=1)

    torch.testing.assert_close(output, torch.tensor([[4, 0, 1]]))
