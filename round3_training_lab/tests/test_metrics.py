import math

import pytest

from training_lab.metrics import AverageMeter, perplexity, tokens_per_second


def test_perplexity_of_zero_loss_is_one() -> None:
    assert perplexity(0.0) == pytest.approx(1.0)


def test_perplexity_matches_exp_of_loss() -> None:
    assert perplexity(math.log(2.0)) == pytest.approx(2.0)


def test_average_meter_tracks_weighted_mean() -> None:
    meter = AverageMeter()

    assert meter.avg == pytest.approx(0.0)

    meter.update(1.0)
    meter.update(3.0)
    assert meter.avg == pytest.approx(2.0)

    meter.update(6.0, n=2)  # totals: (1 + 3 + 12) / (1 + 1 + 2) = 4.0
    assert meter.avg == pytest.approx(4.0)


def test_average_meter_reset_clears_state() -> None:
    meter = AverageMeter()
    meter.update(5.0)
    meter.reset()

    assert meter.avg == pytest.approx(0.0)


def test_tokens_per_second_is_rate() -> None:
    assert tokens_per_second(1000, 2.0) == pytest.approx(500.0)
