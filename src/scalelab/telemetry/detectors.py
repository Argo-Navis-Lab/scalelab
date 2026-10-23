"""Online detectors: they watch a run's metrics step by step and say when something looks
wrong. The spike detector judges each value against the metric's recent values, measured with
the median and the median absolute deviation, which a few spikes barely move. The divergence
detector looks for NaNs and for a metric growing steadily; the collapse detector, for a policy
that has stopped exploring or learning."""

import math
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# m2s33: what the spike detector watches, and how.
SPIKE_METRICS = ("optim/loss", "optim/grad_norm", "policy/kl_ref")
SPIKE_WINDOW = 200  # each value is judged against its metric's last 200 values
SPIKE_THRESHOLD = 6.0  # a spike lies more than 6 standard deviations from their median
SPIKE_MIN_HISTORY = 50  # fewer values than this say too little to judge by
MAD_TO_STD = 1.4826  # a normal distribution's standard deviation is 1.4826 times its MAD


def median_and_mad(values: Sequence[float]) -> tuple[float, float]:
    """The median of `values`, and the median of their distances from it (the MAD)."""
    median = statistics.median(values)
    return median, statistics.median(abs(value - median) for value in values)


def robust_z(value: float, median: float, mad: float) -> float:
    """How many standard deviations `value` lies above the median (below, if negative), the
    standard deviation estimated from the MAD. Unlike the mean and the standard deviation, the
    median and the MAD hardly move when a few values are wild."""
    return (value - median) / (MAD_TO_STD * mad)


@dataclass(frozen=True)
class Spike:
    """A value far from its metric's recent values."""

    step: int
    metric: str
    value: float
    median: float  # of the window the value was judged against
    mad: float  # of that window
    z: float  # robust_z: how far, in standard deviations


class SpikeDetector:
    """Judges each step's value of each watched metric against that metric's last `window`
    values: a spike when its robust_z is beyond `threshold` either way. Every value judged
    joins its window afterwards, spikes too: a few hardly move the median or the MAD, and a
    lasting jump is a spike for half a window, then the new normal. No verdict until a
    window holds SPIKE_MIN_HISTORY values, nor while its MAD is 0 (no spread to measure
    against). Non-finite values are neither judged nor kept. A metric missing from a step,
    such as policy/kl_ref without a reference, is skipped."""

    def __init__(
        self,
        metrics: Sequence[str] = SPIKE_METRICS,
        window: int = SPIKE_WINDOW,
        threshold: float = SPIKE_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.windows = {metric: deque[float](maxlen=window) for metric in metrics}

    def check(self, step: int, values: Mapping[str, float]) -> list[Spike]:
        """The spikes among one step's metric values; steps come in order."""
        spikes: list[Spike] = []
        for metric, window in self.windows.items():
            value = values.get(metric)
            if value is None or not math.isfinite(value):
                continue
            if len(window) >= SPIKE_MIN_HISTORY:
                median, mad = median_and_mad(window)
                if mad > 0:
                    z = robust_z(value, median, mad)
                    if abs(z) > self.threshold:
                        spikes.append(Spike(step, metric, value, median, mad, z))
            window.append(value)
        return spikes


# m2s34: what the divergence detector watches, and how. Not the loss (Section 7.3 fits the slope
# of the log-loss): RL's averages advantages that sum to 0 in every group, so it stays near 0,
# often below, and has no log.
DIVERGENCE_METRIC = "optim/grad_norm"
DIVERGENCE_WINDOW = 100  # the trend is fitted over the metric's last 100 values
DIVERGENCE_SLOPE = 0.01  # in log units a step: growing e-fold (2.7 times) every 100 steps
NONFINITE_PREFIX = "numerics/"  # NumericsCollector's counts of NaNs and infinities


@dataclass(frozen=True)
class Divergence:
    """A run blowing up: NaNs or infinities counted (`value` is the count), or the watched
    metric rising steadily (`value` is the slope of its log, per step)."""

    step: int
    metric: str
    value: float


class DivergenceDetector:
    """Watches a run for blowing up, two ways. Any NaN or infinity NumericsCollector counted
    (a numerics/ metric above 0) is a divergence at once: in fp32 no loss scaler skips a bad
    step, and clipping turns an infinite gradient into NaN weights. And `metric` rising
    steadily is one: the least-squares slope of its log against the step, over its last
    `window` values, above `threshold`. The log makes the slope a growth rate, the same at any
    scale. No slope verdict until the window is full. Values with no log (0, negative or not
    finite) are not kept: a NaN shows in the counts. The learning rate is not watched: the
    blueprint's "while it is not increasing" always holds in RL, whose rate is constant."""

    def __init__(
        self,
        metric: str = DIVERGENCE_METRIC,
        window: int = DIVERGENCE_WINDOW,
        threshold: float = DIVERGENCE_SLOPE,
    ) -> None:
        self.metric = metric
        self.threshold = threshold
        self.window = deque[tuple[int, float]](maxlen=window)  # (step, log of the value)

    def check(self, step: int, values: Mapping[str, float]) -> list[Divergence]:
        """The signs of divergence in one step's metric values; steps come in order."""
        divergences = [
            Divergence(step, metric, count)
            for metric, count in values.items()
            if metric.startswith(NONFINITE_PREFIX) and count > 0
        ]
        value = values.get(self.metric)
        if value is not None and math.isfinite(value) and value > 0:
            self.window.append((step, math.log(value)))
        if len(self.window) == self.window.maxlen:
            steps, logs = zip(*self.window, strict=True)
            slope = statistics.linear_regression(steps, logs).slope
            if slope > self.threshold:
                divergences.append(Divergence(step, self.metric, slope))
        return divergences


# m2s34: what the collapse detector watches, and how.
ENTROPY, FLAT_GROUPS = "policy/entropy", "policy/flat_groups"
COLLAPSE_WINDOW = 20  # each metric's mean over its last 20 values is judged: one step is noisy
ENTROPY_FLOOR = 0.01  # nats per answer token: a group's samples are then mostly copies
FLAT_GROUPS_LIMIT = 0.9  # above it, next to no group's samples got different rewards


@dataclass(frozen=True)
class Collapse:
    """A policy that has stopped learning: a metric's recent mean beyond its limit."""

    step: int
    metric: str
    mean: float  # of the metric's window
    limit: float  # the floor or the ceiling it crossed


class CollapseDetector:
    """Watches for a policy that has stopped learning, judging each metric's mean over its
    last `window` values. Entropy below `entropy_floor` means it samples the same answers
    again and again: two samples of an answer whose entropy is H nats (per token times the
    tokens) are the same with probability at least e^-H, 64% for 45 tokens at 0.01. Flat
    groups above `flat_groups_limit` mean nearly every group's rewards are all equal, so their
    advantages are 0 and a step learns next to nothing. Means, not single steps: 64 groups
    measure a step's flat share with a standard deviation of about 0.045 (at 0.85), 20
    steps' mean with about 0.01. No verdict until a window is full; values that are not
    finite are not kept; a missing metric is skipped."""

    def __init__(
        self,
        window: int = COLLAPSE_WINDOW,
        entropy_floor: float = ENTROPY_FLOOR,
        flat_groups_limit: float = FLAT_GROUPS_LIMIT,
    ) -> None:
        self.window = window
        self.entropy_floor = entropy_floor
        self.flat_groups_limit = flat_groups_limit
        self.windows = {metric: deque[float](maxlen=window) for metric in (ENTROPY, FLAT_GROUPS)}

    def check(self, step: int, values: Mapping[str, float]) -> list[Collapse]:
        """The signs of collapse in one step's metric values; steps come in order."""
        collapses: list[Collapse] = []
        for metric, window in self.windows.items():
            value = values.get(metric)
            if value is not None and math.isfinite(value):
                window.append(value)
            if len(window) < self.window:
                continue
            mean = statistics.fmean(window)
            if metric == ENTROPY and mean < self.entropy_floor:
                collapses.append(Collapse(step, metric, mean, self.entropy_floor))
            if metric == FLAT_GROUPS and mean > self.flat_groups_limit:
                collapses.append(Collapse(step, metric, mean, self.flat_groups_limit))
        return collapses
