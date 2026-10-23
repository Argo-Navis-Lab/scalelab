"""Online detectors: they watch a run's metrics step by step and say when something looks
wrong. The spike detector judges each value against the metric's recent values, measured with
the median and the median absolute deviation, which a few spikes barely move."""

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
