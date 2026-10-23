"""The statistics every report uses: interquartile means and bootstrap confidence intervals
over seeds and eval problems."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

RESAMPLES = 10_000
CONFIDENCE = 0.95
BOOTSTRAP_SEED = 0  # fixed: the same values always give the same interval
CHUNK = 1_000  # resamples drawn at once: memory ~ CHUNK x seeds x problems x 8 bytes

Statistic = Literal["mean", "iqm"]


@dataclass(frozen=True)
class Interval:
    """A statistic of the values and the bootstrap confidence interval around it."""

    estimate: float  # the statistic of the values themselves, not of the resamples
    low: float
    high: float


def _iqm(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The interquartile mean along the last axis: sort, cut n // 4 values off each end, take
    the mean of the rest (scipy's trim_mean(values, 0.25), as rliable uses it)."""
    n = values.shape[-1]
    cut = n // 4
    return np.sort(values, axis=-1)[..., cut : n - cut].mean(axis=-1)


def _statistic(values: npt.NDArray[np.float64], statistic: Statistic) -> npt.NDArray[np.float64]:
    return values.mean(axis=-1) if statistic == "mean" else _iqm(values)


def iqm(values: npt.ArrayLike) -> float:
    """The interquartile mean of one value per run: one bad run can't drag it far. Below 4
    values nothing is cut, so it is the plain mean."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"iqm needs a non-empty 1-D array, got shape {array.shape}")
    return float(_iqm(array))


def bootstrap_interval(
    values: npt.ArrayLike,
    *,
    statistic: Statistic = "mean",
    resamples: int = RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """The percentile bootstrap interval of `statistic` across seeds.

    values[s] is seed s's score, or values[s, i] its score on eval problem i (seeds x
    problems), a seed's score then being the mean over its problems. Each resample draws
    problems with replacement, the same problems for every seed (they were all graded on one
    eval set), then seeds with replacement, and takes `statistic` of the drawn seeds' scores.
    The interval holds the middle `confidence` of the resampled statistics.

    To compare two things graded on the same problems, such as a run's last grading against
    its first, pass the per-problem differences (last - first): resampling the differences
    keeps the pairing.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, np.newaxis]  # one score per seed: a single "problem" each
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"values must be seeds or seeds x problems, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    seeds, problems = array.shape
    rng = np.random.default_rng(seed)
    results: list[npt.NDArray[np.float64]] = []
    for start in range(0, resamples, CHUNK):
        size = min(CHUNK, resamples - start)
        drawn = rng.integers(problems, size=(size, problems))  # one row of problems a resample
        scores = array[:, drawn].mean(axis=-1).T  # (size, seeds): each seed's score on them
        picks = rng.integers(seeds, size=(size, seeds))  # one row of seeds a resample
        results.append(_statistic(np.take_along_axis(scores, picks, axis=1), statistic))
    tail = (1 - confidence) / 2
    low, high = np.quantile(np.concatenate(results), [tail, 1 - tail])
    estimate = _statistic(array.mean(axis=1), statistic)
    return Interval(estimate=float(estimate), low=float(low), high=float(high))
