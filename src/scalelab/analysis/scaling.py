"""Scaling fits. Each RL run's pass rate R against the compute C it has spent is fit with a
saturating sigmoid, R(C) = R_0 + (A - R_0) / (1 + (C_mid / C)^B): R_0 is the run's step-0
grading (its warm start's), measured, not fitted; A is where R levels off, C_mid the compute
where R is halfway from R_0 to A, and B how sharply it gets there. The curve is a logistic in
log C: 1 / (1 + (C_mid / C)^B) = expit(B (ln C - ln C_mid))."""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.special import expit

from scalelab.rl.gradings import Grading
from scalelab.telemetry.sink import MetricRow

# what rl.flops spends: a step's sampling, log-prob and training FLOPs, counted by A
FLOPS_METRICS = ("system/sample_flops", "system/logprobs_flops", "system/train_flops")
ALL_OPS = "+-*"  # task.ops None mixes all three operations
MIN_POINTS = 3  # A, C_mid and B: at least three gradings after step 0
B_BOUNDS = (0.01, 20.0)  # B > 0: R moves once, from R_0 towards A; 20 is already a step
DECADES = 3.0  # C_mid within 3 decades of the graded compute, either side
STARTS = (0.25, 0.5, 0.75)  # C_mid's starting points: these quantiles of the graded log C
SLOPES = (0.5, 1.0, 2.0)  # B's starting points
PARAMETERS = ("A", "C_mid", "B")  # the fitted ones, in least_squares' order
BOUND_TOLERANCE = 1e-3  # a parameter this share of its range from a bound is on it
NEAR = 0.9  # "near A": 90% of the way from R_0 (the M4 plan's budget rule, m4s9)


@dataclass(frozen=True)
class Curve:
    """R(C), one run's pass rate against its compute in FLOPs."""

    r0: float  # R at C = 0: the warm start's, measured
    a: float  # the asymptote: where R levels off as C grows
    c_mid: float  # the compute where R is halfway from R_0 to A
    b: float  # how sharply R moves around C_mid, in log C

    def __call__(self, compute: ArrayLike) -> NDArray[np.float64]:
        """R at each compute (FLOPs, 0 or more): R_0 at 0."""
        c = np.asarray(compute, dtype=np.float64)
        with np.errstate(divide="ignore"):  # log 0 = -inf, and expit(-inf) = 0: R_0
            share = expit(self.b * (np.log(c) - np.log(self.c_mid)))
        return self.r0 + (self.a - self.r0) * share

    def compute_at(self, fraction: float) -> float:
        """The compute where R has gone `fraction` of the way from R_0 to A (0 < fraction < 1):
        C_mid at 0.5."""
        if not 0 < fraction < 1:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
        return self.c_mid * (fraction / (1 - fraction)) ** (1 / self.b)


@dataclass(frozen=True)
class Fit:
    """A curve fit to a run's gradings."""

    curve: Curve
    rmse: float  # root mean square of R's residuals at the fitted gradings
    points: int  # gradings fitted: those after step 0
    bounded: tuple[str, ...]  # PARAMETERS the fit left on a bound: the gradings don't pin them


@dataclass(frozen=True)
class Point:
    """One grading of an RL run, placed on its curve."""

    step: int
    compute: float  # C: FLOPs spent by the end of this step, 0 at step 0
    rate: float  # R: greedy pass@1 on the problems of the task's operations
    n: int  # how many problems R is over


def fit_curve(compute: Sequence[float], rates: Sequence[float], r0: float) -> Fit:
    """A, C_mid and B by least squares on R, with R_0 held at the measured `r0`, given the
    gradings after step 0 as pairs: `compute` (C > 0, increasing) and `rates` (R). Starts from
    several points and keeps the best fit: such a sum of squares can have more than one
    minimum."""
    c = np.asarray(compute, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    if c.ndim != 1 or c.shape != r.shape:
        raise ValueError("compute and rates must be two lists of the same length")
    if len(c) < MIN_POINTS:
        raise ValueError(f"a fit needs {MIN_POINTS} gradings after step 0 or more, got {len(c)}")
    if c[0] <= 0 or np.any(np.diff(c) <= 0):
        raise ValueError("compute must be above 0 and increasing")
    log_c = np.log(c)

    def residuals(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        a, log_mid, b = theta
        return r0 + (a - r0) * expit(b * (log_c - log_mid)) - r

    span = DECADES * np.log(10)
    lower = [0.0, log_c[0] - span, B_BOUNDS[0]]
    upper = [1.0, log_c[-1] + span, B_BOUNDS[1]]
    a0 = float(np.clip(r[-(len(r) // 3 + 1) :].mean(), 0.0, 1.0))  # the last third's mean
    best = None
    for log_mid in np.quantile(log_c, STARTS):
        for b in SLOPES:
            found = least_squares(residuals, [a0, log_mid, b], bounds=(lower, upper))
            if best is None or found.cost < best.cost:
                best = found
    assert best is not None
    a, log_mid, b = (float(value) for value in best.x)
    curve = Curve(r0=r0, a=a, c_mid=float(np.exp(log_mid)), b=b)
    rmse = float(np.sqrt(np.mean(np.square(residuals(best.x)))))
    gap = np.minimum(best.x - lower, np.subtract(upper, best.x))
    on = gap <= BOUND_TOLERANCE * np.subtract(upper, lower)
    bounded = tuple(name for name, at in zip(PARAMETERS, on, strict=True) if at)
    return Fit(curve=curve, rmse=rmse, points=len(c), bounded=bounded)


def cumulative_compute(rows: Iterable[MetricRow]) -> dict[int, float]:
    """C after each step, from a run's metrics: the FLOPs rl.flops spends (FLOPS_METRICS),
    summed over the steps up to it. Every step from 1 on must have all of them."""
    per_step: dict[int, float] = {}
    counts: Counter[int] = Counter()
    for row in rows:
        if row.metric in FLOPS_METRICS and row.layer is None:
            per_step[row.step] = per_step.get(row.step, 0.0) + row.value
            counts[row.step] += 1
    steps = sorted(per_step)
    if steps != list(range(1, len(steps) + 1)) or any(
        counts[step] != len(FLOPS_METRICS) for step in steps
    ):
        raise ValueError("the metrics miss some steps' FLOPs")
    return dict(zip(steps, accumulate(per_step[step] for step in steps), strict=True))


def pooled_rate(grading: Grading, ops: str) -> tuple[float, int]:
    """Greedy pass@1 over the problems of `ops` (operations, such as "*"), pooled across the
    grading's eval sets: its parts named "<eval set>/<op>". Returns the rate and n."""
    parts = [
        result
        for result in grading.greedy
        if "/" in result.name and result.name.rsplit("/", 1)[1] in set(ops)
    ]
    n = sum(part.n for part in parts)
    if n == 0:
        raise ValueError(f"the grading at step {grading.step} has no {ops} problems")
    return sum(part.passed for part in parts) / n, n


def run_points(gradings: Sequence[Grading], rows: Iterable[MetricRow], ops: str) -> list[Point]:
    """A run's gradings (load_gradings) as points (C, R): C from its metrics (read_metrics),
    R pooled over the problems of `ops`."""
    compute = cumulative_compute(rows)
    points: list[Point] = []
    for grading in gradings:
        if grading.step and grading.step not in compute:
            raise ValueError(f"the metrics stop before step {grading.step}, a graded one")
        rate, n = pooled_rate(grading, ops)
        points.append(Point(grading.step, compute.get(grading.step, 0.0), rate, n))
    return points


def fit_run(points: Sequence[Point]) -> Fit:
    """The curve through a run's points (run_points): R_0 is its step-0 point's R."""
    if not points or points[0].step != 0:
        raise ValueError("a run's points must start at step 0, the warm start's grading")
    rest = points[1:]
    return fit_curve([p.compute for p in rest], [p.rate for p in rest], points[0].rate)
