import torch

from training_lab.sampling import (
    apply_temperature,
    sample_next_token,
    top_k_filter,
    top_p_filter,
)


def test_apply_temperature_scales_logits() -> None:
    logits = torch.tensor([[2.0, 4.0, 6.0]])

    scaled = apply_temperature(logits, temperature=2.0)

    torch.testing.assert_close(scaled, logits / 2.0)


def test_top_k_filter_keeps_only_k_largest() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])

    filtered = top_k_filter(logits, k=2)

    finite = torch.isfinite(filtered)
    assert finite.tolist() == [[False, True, True, False]]


def test_top_k_filter_noop_when_k_covers_vocab() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])

    filtered = top_k_filter(logits, k=4)

    torch.testing.assert_close(filtered, logits)


def test_top_p_filter_keeps_smallest_nucleus() -> None:
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = torch.log(probs).unsqueeze(0)

    filtered = top_p_filter(logits, p=0.75)

    finite = torch.isfinite(filtered)
    # cumulative 0.5 (<0.75) then 0.8 (>=0.75): keep the top two tokens
    assert finite.tolist() == [[True, True, False, False]]


def test_top_p_filter_always_keeps_top_token() -> None:
    probs = torch.tensor([0.9, 0.06, 0.04])
    logits = torch.log(probs).unsqueeze(0)

    filtered = top_p_filter(logits, p=0.5)

    finite = torch.isfinite(filtered)
    assert finite.tolist() == [[True, False, False]]


def test_sample_next_token_top_k_one_is_greedy() -> None:
    logits = torch.tensor([[0.1, 5.0, 0.2, -1.0], [3.0, 0.0, 9.0, 1.0]])

    tokens = sample_next_token(logits, temperature=1.0, top_k=1)

    assert tokens.shape == (2,)
    assert tokens.dtype == torch.long
    assert tokens.tolist() == [1, 2]
