"""Statistics tests: the interquartile mean by hand, bootstrap intervals by hand on tiny cases,
their coverage on simulated eval sets, one draw of problems shared by every seed, and the
interval's reproducibility, monotonicity in the confidence and refusal of bad input."""

import math

import numpy as np
import pytest

from scalelab.analysis.stats import Interval, bootstrap_interval, iqm


def test_iqm_cuts_a_quarter_off_each_end() -> None:
    assert iqm([4.0, 1.0, 3.0, 2.0]) == 2.5  # 2 and 3 remain
    assert iqm([1.0, 2.0, 3.0, 4.0, 100.0]) == 3.0  # 2, 3, 4 remain: the outlier is cut
    assert iqm([float(value) for value in range(8)]) == 3.5  # 2, 3, 4, 5 remain


def test_iqm_of_fewer_than_four_values_is_the_mean() -> None:
    assert iqm([0.5]) == 0.5
    assert iqm([1.0, 2.0, 6.0]) == 3.0


@pytest.mark.parametrize("values", [[], [[1.0, 2.0]]])
def test_iqm_refuses_anything_but_a_non_empty_1d_array(values: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="non-empty 1-D"):
        iqm(values)


def test_identical_values_give_a_zero_width_interval() -> None:
    assert bootstrap_interval(np.full((3, 50), 0.25)) == Interval(0.25, 0.25, 0.25)


def test_three_seeds_by_hand() -> None:
    # A resample is all three copies of the lowest seed with probability 1/27 = 3.7%, more
    # than the 2.5% tail, so the interval runs from the lowest seed to the highest.
    assert bootstrap_interval([0.0, 1.0, 2.0]) == Interval(1.0, 0.0, 2.0)


def test_the_estimate_is_the_statistic_of_the_values() -> None:
    # Seed scores are means over problems: 0.5 and 1.0.
    assert bootstrap_interval([[0.0, 1.0], [1.0, 1.0]]).estimate == 0.75
    scores = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert bootstrap_interval(scores, statistic="iqm").estimate == 3.0
    assert bootstrap_interval(scores).estimate == 22.0


def test_the_iqm_of_three_seeds_is_their_mean() -> None:
    values = np.random.default_rng(0).random((3, 40))
    by_iqm = bootstrap_interval(values, statistic="iqm")
    by_mean = bootstrap_interval(values)
    assert by_iqm.low == pytest.approx(by_mean.low, rel=1e-12)
    assert by_iqm.high == pytest.approx(by_mean.high, rel=1e-12)


def test_intervals_cover_the_true_pass_rate_about_95_percent_of_the_time() -> None:
    # 200 simulated eval sets of 300 problems, each passed with probability 1/pi: no pass
    # rate k/300 equals it, so no interval ends exactly on it.
    rate = 1 / math.pi
    rng = np.random.default_rng(0)
    covered = 0
    for trial in range(200):
        passed = (rng.random(300) < rate).astype(np.float64)
        interval = bootstrap_interval(passed[np.newaxis], resamples=1_000, seed=trial)
        covered += interval.low <= rate <= interval.high
    assert 0.9 <= covered / 200 <= 0.99


def test_every_seed_is_scored_on_the_same_drawn_problems() -> None:
    # Three identical seeds hold no more information than one: the interval stays as wide.
    # Problems drawn separately for each seed would average out and shrink it by sqrt(3).
    row = (np.random.default_rng(0).random(400) < 0.5).astype(np.float64)
    one = bootstrap_interval(row[np.newaxis])
    three = bootstrap_interval(np.stack([row, row, row]))
    assert (three.high - three.low) / (one.high - one.low) == pytest.approx(1.0, abs=0.1)


def test_the_same_seed_gives_the_same_interval() -> None:
    values = np.random.default_rng(0).random((3, 100))
    first = bootstrap_interval(values)
    assert bootstrap_interval(values) == first
    other = bootstrap_interval(values, seed=1)
    assert other != first
    assert other.low == pytest.approx(first.low, abs=0.01)
    assert other.high == pytest.approx(first.high, abs=0.01)


def test_more_confidence_gives_a_wider_interval() -> None:
    values = np.random.default_rng(0).random((3, 100))
    narrow = bootstrap_interval(values, confidence=0.5)
    wide = bootstrap_interval(values, confidence=0.99)
    assert wide.low < narrow.low < narrow.estimate < narrow.high < wide.high
    one = bootstrap_interval(values, resamples=1)
    assert one.low == one.high  # one resample: one value


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([], "seeds or seeds x problems"),
        ([[[1.0]]], "seeds or seeds x problems"),
        ([math.nan], "finite"),
    ],
)
def test_bad_values_are_refused(values: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        bootstrap_interval(values)


def test_bad_options_are_refused() -> None:
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_interval([1.0, 2.0], resamples=0)
    for confidence in (0.0, 1.0):
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_interval([1.0, 2.0], confidence=confidence)
