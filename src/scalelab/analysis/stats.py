"""The statistics every report uses: interquartile means, bootstrap confidence intervals
over seeds and eval problems, and the power of a comparison across seeds."""

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt

RESAMPLES = 10_000
CONFIDENCE = 0.95
BOOTSTRAP_SEED = 0  # fixed: the same values always give the same interval
CHUNK = 1_000  # resamples drawn at once: memory ~ CHUNK x seeds x problems x 8 bytes
ALPHA = 0.05  # m4s10: a comparison's level, two-sided
POWER = 0.8  # the chance a comparison should have of seeing a real difference
EFFECT = 0.03  # the difference in final R worth seeing between algorithms (the M4 plan)
SEEDS = 5  # seeds a side in M4's comparisons, unless the seed study says 5 can't see EFFECT

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


def detection_power(effect: float, sd: float, seeds: int, alpha: float = ALPHA) -> float:
    """The chance that a two-sided test at level `alpha` tells two variants apart when their
    means differ by `effect`, with `seeds` values a side, each side spread by `sd`: the normal
    approximation, unpaired (pairing by seed only helps), the far tail's sliver left out."""
    if effect <= 0 or sd < 0 or seeds < 1 or not 0 < alpha < 1:
        raise ValueError("power needs effect > 0, sd >= 0, seeds >= 1 and 0 < alpha < 1")
    if sd == 0:
        return 1.0  # no spread: any difference shows
    critical = NormalDist().inv_cdf(1 - alpha / 2)
    return NormalDist().cdf(effect / (sd * math.sqrt(2 / seeds)) - critical)


def seeds_for_power(effect: float, sd: float, power: float = POWER, alpha: float = ALPHA) -> int:
    """The fewest seeds a side that give a comparison `power` against `effect` (at least 2: a
    side's spread needs two values)."""
    if effect <= 0 or sd < 0 or not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("seeds need effect > 0, sd >= 0, 0 < power < 1 and 0 < alpha < 1")
    normal = NormalDist()
    z = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    return max(2, math.ceil(2 * (z * sd / effect) ** 2))
