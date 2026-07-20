import math

import pytest

from training_lab.schedule import cosine_lr


def test_warmup_is_linear_from_zero_to_base() -> None:
    kwargs = dict(base_lr=1.0, warmup_steps=10, total_steps=110, min_lr=0.1)

    assert cosine_lr(0, **kwargs) == pytest.approx(0.0)
    assert cosine_lr(5, **kwargs) == pytest.approx(0.5)


def test_cosine_decay_endpoints_and_midpoint() -> None:
    kwargs = dict(base_lr=1.0, warmup_steps=10, total_steps=110, min_lr=0.1)

    # end of warmup / start of cosine
    assert cosine_lr(10, **kwargs) == pytest.approx(1.0)
    # halfway through the cosine phase
    assert cosine_lr(60, **kwargs) == pytest.approx(0.55)
    # end of schedule
    assert cosine_lr(110, **kwargs) == pytest.approx(0.1)


def test_lr_clamps_to_min_after_total_steps() -> None:
    kwargs = dict(base_lr=1.0, warmup_steps=10, total_steps=110, min_lr=0.1)

    assert cosine_lr(500, **kwargs) == pytest.approx(0.1)


def test_no_warmup_starts_at_base_lr() -> None:
    lr = cosine_lr(0, base_lr=2.0, warmup_steps=0, total_steps=100, min_lr=0.0)

    assert lr == pytest.approx(2.0)


def test_cosine_formula_matches_reference_value() -> None:
    lr = cosine_lr(30, base_lr=1.0, warmup_steps=10, total_steps=110, min_lr=0.0)

    progress = (30 - 10) / (110 - 10)
    expected = 0.5 * (1 + math.cos(math.pi * progress))
    assert lr == pytest.approx(expected)
