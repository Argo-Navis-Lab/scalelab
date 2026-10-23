"""Spike detector tests: the robust z-score by hand, the MAD as a standard deviation, spikes
found on synthetic series and none on clean ones, and what each window keeps."""

import math
import random
import statistics

import pytest

from scalelab.telemetry.detectors import (
    MAD_TO_STD,
    SPIKE_MIN_HISTORY,
    SPIKE_THRESHOLD,
    SPIKE_WINDOW,
    Spike,
    SpikeDetector,
    median_and_mad,
    robust_z,
)

LOSS, GRAD_NORM = "optim/loss", "optim/grad_norm"
STEADY = [10.0 + step % 5 for step in range(SPIKE_MIN_HISTORY)]  # 10..14 ten times each


def feed(detector: SpikeDetector, values: list[float], metric: str = LOSS) -> list[Spike]:
    """Each value as one step's `metric`, steps from 1; every spike found."""
    spikes: list[Spike] = []
    for step, value in enumerate(values, 1):
        spikes += detector.check(step, {metric: value})
    return spikes


def noisy(steps: int, seed: int) -> list[float]:
    """A clean loss curve: falling slowly from 1 toward 0.5, with normal noise (std 0.02)."""
    rng = random.Random(seed)
    return [0.5 + 0.5 * math.exp(-step / 300) + rng.gauss(0, 0.02) for step in range(steps)]


def test_the_median_the_mad_and_the_robust_z_by_hand():
    assert median_and_mad(STEADY) == (12.0, 1.0)  # distances 0, 1, 1, 2, 2: their median is 1
    assert robust_z(20.0, 12.0, 1.0) == pytest.approx(8 / 1.4826)
    assert robust_z(4.0, 12.0, 1.0) == pytest.approx(-8 / 1.4826)
    quiet = SpikeDetector()
    assert feed(quiet, [*STEADY, 20.0]) == []  # z = 5.40: inside the threshold
    loud = SpikeDetector()
    [spike] = feed(loud, [*STEADY, 21.0])  # z = 6.07: beyond it
    assert spike == Spike(SPIKE_MIN_HISTORY + 1, LOSS, 21.0, 12.0, 1.0, 9 / MAD_TO_STD)


def test_the_mad_times_its_constant_estimates_a_normal_standard_deviation():
    assert MAD_TO_STD == pytest.approx(1 / statistics.NormalDist().inv_cdf(0.75), rel=1e-5)
    rng = random.Random(0)
    _, mad = median_and_mad([rng.gauss(5.0, 2.0) for _ in range(100_000)])
    assert MAD_TO_STD * mad == pytest.approx(2.0, rel=0.02)


@pytest.mark.parametrize("seed", range(10))
def test_a_clean_series_raises_no_spike(seed: int):
    assert feed(SpikeDetector(), noisy(1000, seed)) == []


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("jump", [0.5, -0.5])  # 25 standard deviations of the noise
def test_a_synthetic_spike_is_found_and_hardly_moves_the_window(seed: int, jump: float):
    values = noisy(1000, seed)
    values[499] += jump  # step 500
    values[699] += jump  # step 700 and 701: a spike right after a spike is still one
    values[700] += jump
    spikes = feed(SpikeDetector(), values)
    assert [spike.step for spike in spikes] == [500, 700, 701]
    assert all(abs(spike.z) > SPIKE_THRESHOLD and spike.z * jump > 0 for spike in spikes)


def test_a_lasting_jump_is_a_spike_for_half_a_window_then_the_new_normal():
    values = noisy(SPIKE_WINDOW, 0) + [value + 10.0 for value in noisy(2 * SPIKE_WINDOW, 1)]
    spikes = feed(SpikeDetector(), values)
    assert [spike.step for spike in spikes] == list(range(201, 201 + SPIKE_WINDOW // 2))


def test_no_verdict_before_enough_history_nor_without_spread():
    assert feed(SpikeDetector(), [*STEADY[:-1], 1000.0]) == []  # one value short
    assert feed(SpikeDetector(), [1.0] * 100 + [2.0]) == []  # MAD 0: no yardstick
    assert len(feed(SpikeDetector(), [1.0, 3.0] * 25 + [12.0])) == 1  # MAD 1: z = 6.75


def test_each_metric_has_its_own_window_and_unusable_values_are_skipped():
    detector = SpikeDetector()
    for step, value in enumerate(STEADY, 1):
        assert detector.check(step, {LOSS: value, GRAD_NORM: value, "rollout/length_max": 1}) == []
    assert detector.check(51, {LOSS: math.nan, GRAD_NORM: math.inf}) == []  # judged by no one
    assert detector.check(52, {}) == []  # e.g. policy/kl_ref without a reference
    assert [len(window) for window in detector.windows.values()] == [SPIKE_MIN_HISTORY] * 2 + [0]
    [spike] = detector.check(53, {LOSS: 12.0, GRAD_NORM: 50.0})
    assert (spike.metric, spike.value) == (GRAD_NORM, 50.0)
    for step in range(54, 1000):
        detector.check(step, {LOSS: 12.0})
    assert len(detector.windows[LOSS]) == SPIKE_WINDOW  # its last 200 values only
