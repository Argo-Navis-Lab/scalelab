"""Detector tests. Spikes: the robust z-score by hand, the MAD as a standard deviation, spikes
found on synthetic series and none on clean ones, what each window keeps. Divergence: the log
slope by hand, NaN counts, blowups found and clean series quiet. Collapse: window means by
hand, a collapse found, noisy single steps ignored."""

import math
import random
import statistics

import pytest

from scalelab.telemetry.detectors import (
    COLLAPSE_WINDOW,
    DIVERGENCE_WINDOW,
    ENTROPY,
    ENTROPY_FLOOR,
    FLAT_GROUPS,
    FLAT_GROUPS_LIMIT,
    MAD_TO_STD,
    REGRESSION_STEPS,
    SPIKE_MIN_HISTORY,
    SPIKE_THRESHOLD,
    SPIKE_WINDOW,
    STEP_TIME,
    Collapse,
    CollapseDetector,
    Divergence,
    DivergenceDetector,
    Spike,
    SpikeDetector,
    ThroughputDetector,
    median_and_mad,
    regression_limit,
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


FAST = [math.exp(0.03 * step) for step in range(1, DIVERGENCE_WINDOW + 1)]  # e-fold in 33


def diverge(detector: DivergenceDetector, values: list[float]) -> list[Divergence]:
    """Each value as one step's gradient norm, steps from 1; every divergence found."""
    return [
        found
        for step, value in enumerate(values, 1)
        for found in detector.check(step, {GRAD_NORM: value})
    ]


def grad_norms(steps: int, seed: int, growth: float = 0.0, start: int = 0) -> list[float]:
    """A healthy gradient norm: around 0.5, each step off by a factor e^N(0, 0.3), about 35%;
    after step `start`, growing e^growth times a step."""
    rng = random.Random(seed)
    return [
        0.5 * math.exp(rng.gauss(0, 0.3) + growth * max(0, step - start))
        for step in range(1, steps + 1)
    ]


def test_a_steady_growth_is_found_as_its_log_slope_by_hand():
    [found] = diverge(DivergenceDetector(), FAST)  # at step 100: the first full window
    assert (found.step, found.metric) == (DIVERGENCE_WINDOW, GRAD_NORM)
    assert found.value == pytest.approx(0.03)
    slow = [math.exp(0.005 * step) for step in range(1, 1000)]  # e-fold in 200 steps
    assert diverge(DivergenceDetector(), slow) == []


@pytest.mark.parametrize("seed", range(10))
def test_a_clean_gradient_norm_shows_no_divergence(seed: int):
    assert diverge(DivergenceDetector(), grad_norms(1000, seed)) == []


@pytest.mark.parametrize("seed", range(5))
def test_a_blowup_is_found_within_steps_and_stays_found(seed: int):
    values = grad_norms(1000, seed, growth=0.05, start=500)  # 12 times larger every 50 steps
    steps = [found.step for found in diverge(DivergenceDetector(), values)]
    assert 515 <= steps[0] <= 540  # some 25 steps in: the window still holds the calm before
    assert steps == list(range(steps[0], 1001))


def test_any_nan_or_infinity_counted_is_a_divergence_at_once():
    counts = {"numerics/logp_nan": 0.0, "numerics/grads_inf": 2.0, "numerics/weights_nan": 7.0}
    assert DivergenceDetector().check(1, counts) == [
        Divergence(1, "numerics/grads_inf", 2.0),
        Divergence(1, "numerics/weights_nan", 7.0),
    ]
    assert DivergenceDetector().check(1, dict.fromkeys(counts, 0.0)) == []


def test_the_slope_needs_a_full_window_of_values_with_a_log_and_is_per_step():
    junk = [0.0, -1.0, math.nan, math.inf]  # no log, or none worth keeping
    assert diverge(DivergenceDetector(), [*junk, *FAST[:-1]]) == []  # 99 kept: one short
    assert len(diverge(DivergenceDetector(), [*junk, *FAST])) == 1
    detector = DivergenceDetector()  # a value every other step: a step with no update has none
    found = [
        divergence
        for step, value in enumerate(FAST, 1)
        for divergence in detector.check(2 * step, {GRAD_NORM: value})
    ]
    assert [(divergence.step, divergence.value) for divergence in found] == [
        (2 * DIVERGENCE_WINDOW, pytest.approx(0.015))
    ]


def collapse(detector: CollapseDetector, rows: list[dict[str, float]]) -> list[Collapse]:
    """Each row as one step's metrics, steps from 1; every sign of collapse found."""
    return [found for step, row in enumerate(rows, 1) for found in detector.check(step, row)]


def learning(
    steps: int, seed: int, flat: float = 0.85, entropy: float = 0.12
) -> list[dict[str, float]]:
    """A policy's steps: each step's flat share counted over 64 groups, each one flat with
    probability `flat`, and its entropy off by up to 10% either way."""
    rng = random.Random(seed)
    return [
        {
            ENTROPY: entropy * rng.uniform(0.9, 1.1),
            FLAT_GROUPS: sum(rng.random() < flat for _ in range(64)) / 64,
        }
        for _ in range(steps)
    ]


def test_a_window_mean_beyond_its_limit_is_a_collapse_by_hand():
    stuck = [{ENTROPY: 0.005, FLAT_GROUPS: 1.0}] * COLLAPSE_WINDOW
    assert collapse(CollapseDetector(), stuck[:-1]) == []  # one value short
    assert collapse(CollapseDetector(), stuck) == [
        Collapse(COLLAPSE_WINDOW, ENTROPY, 0.005, ENTROPY_FLOOR),
        Collapse(COLLAPSE_WINDOW, FLAT_GROUPS, 1.0, FLAT_GROUPS_LIMIT),
    ]
    edge = [{ENTROPY: ENTROPY_FLOOR, FLAT_GROUPS: FLAT_GROUPS_LIMIT}] * 100
    assert collapse(CollapseDetector(), edge) == []  # at a limit is not beyond it


@pytest.mark.parametrize("seed", range(10))
def test_a_learning_policy_shows_no_collapse_though_single_steps_cross_the_limit(seed: int):
    rows = learning(1000, seed)
    assert any(row[FLAT_GROUPS] > FLAT_GROUPS_LIMIT for row in rows)  # one step is noisy
    assert collapse(CollapseDetector(), rows) == []


@pytest.mark.parametrize("seed", range(5))
def test_a_policy_that_stops_exploring_is_found(seed: int):
    rows = learning(200, seed) + learning(200, seed, flat=0.97, entropy=0.004)
    found = collapse(CollapseDetector(), rows)
    entropy = [collapse.step for collapse in found if collapse.metric == ENTROPY]
    flat = [collapse.step for collapse in found if collapse.metric == FLAT_GROUPS]
    assert entropy == list(range(219, 401))  # 19 values of 0.004 and one of 0.12: below 0.01
    assert 205 <= flat[0] <= 215  # some 9 steps in: (9 * 0.97 + 11 * 0.85) / 20 > 0.9
    assert flat == list(range(flat[0], 401))


def test_each_metric_has_its_own_window_and_unusable_values_are_not_kept():
    detector = CollapseDetector()
    for step in range(1, COLLAPSE_WINDOW):
        assert detector.check(step, {FLAT_GROUPS: 1.0}) == []
    assert detector.check(COLLAPSE_WINDOW, {ENTROPY: math.nan, FLAT_GROUPS: math.inf}) == []
    [found] = detector.check(COLLAPSE_WINDOW + 1, {FLAT_GROUPS: 1.0, "policy/kl_ref": 0.0})
    assert (found.step, found.metric) == (COLLAPSE_WINDOW + 1, FLAT_GROUPS)
    assert [len(window) for window in detector.windows.values()] == [0, COLLAPSE_WINDOW]


# m3s25: the throughput detector.


def test_the_regression_limit_by_hand():
    assert regression_limit(2.0, 0.0, 0.05) == pytest.approx(2.0 / 0.85)
    assert regression_limit(2.0, -0.3, 0.05) == pytest.approx(2.0 / 0.55)  # a model that runs fast
    assert regression_limit(2.0, 0.006, 0.05, sigmas=1.0) == pytest.approx(2.0 / 0.956)
    assert regression_limit(2.0, -0.3, 0.25) == math.inf  # 1 - 0.3 - 0.75: past -100%


def test_slow_steps_in_a_row_are_a_regression_until_a_step_is_not():
    detector = ThroughputDetector(predicted=1.0, limit=1.2)
    for step in range(1, REGRESSION_STEPS):
        assert detector.check(step, {STEP_TIME: 1.3}) == []
    [found] = detector.check(REGRESSION_STEPS, {STEP_TIME: 1.5})
    assert (found.step, found.metric, found.limit, found.predicted) == (20, STEP_TIME, 1.2, 1.0)
    assert (found.steps, found.mean) == (20, pytest.approx((19 * 1.3 + 1.5) / 20))
    [still] = detector.check(21, {STEP_TIME: 1.3})
    assert still.steps == 21
    assert detector.check(22, {STEP_TIME: 1.2}) == []  # at the limit is not slow
    assert (detector.slow, detector.slow_seconds) == (0, 0.0)
    for step in range(23, 23 + REGRESSION_STEPS - 1):
        assert detector.check(step, {STEP_TIME: 1.3}) == []  # counting from 0 again


def test_a_missing_or_non_finite_step_time_is_skipped():
    detector = ThroughputDetector(predicted=1.0, limit=1.2, steps=3)
    assert detector.check(1, {STEP_TIME: 2.0}) == []
    assert detector.check(2, {"optim/loss": 0.1}) == []
    assert detector.check(3, {STEP_TIME: math.nan}) == []
    assert detector.check(4, {STEP_TIME: math.inf}) == []
    assert detector.check(5, {STEP_TIME: 2.0}) == []
    [found] = detector.check(6, {STEP_TIME: 2.0})
    assert (found.steps, found.mean) == (3, 2.0)


@pytest.mark.parametrize("seed", range(10))
def test_steps_scattered_like_the_models_errors_are_no_regression(seed: int):
    rng = random.Random(seed)
    median, spread = 0.006, 0.05  # about the T4's whole steps (m3s12)
    detector = ThroughputDetector(2.0, regression_limit(2.0, median, spread))
    for step in range(1, 1001):
        assert detector.check(step, {STEP_TIME: 2.0 / (1 + rng.gauss(median, spread))}) == []


@pytest.mark.parametrize("seed", range(10))
def test_a_lasting_slowdown_is_found_once_it_has_lasted_long_enough(seed: int):
    rng = random.Random(seed)
    median, spread = 0.006, 0.05
    detector = ThroughputDetector(2.0, regression_limit(2.0, median, spread))
    found: list[int] = []
    for step in range(1, 201):
        slower = 1.5 if step > 100 else 1.0  # from step 101, each step takes 1.5 times as long
        seconds = slower * 2.0 / (1 + rng.gauss(median, spread))
        found += [regression.step for regression in detector.check(step, {STEP_TIME: seconds})]
    assert found == list(range(100 + REGRESSION_STEPS, 201))
