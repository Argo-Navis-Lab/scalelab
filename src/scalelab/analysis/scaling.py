"""Scaling fits. Each RL run's pass rate R against the compute C it has spent is fit with a
saturating sigmoid, R(C) = R_0 + (A - R_0) / (1 + (C_mid / C)^B): R_0 is the run's step-0
grading (its warm start's), measured, not fitted; A is where R levels off, C_mid the compute
where R is halfway from R_0 to A, and B how sharply it gets there. The curve is a logistic in
log C: 1 / (1 + (C_mid / C)^B) = expit(B (ln C - ln C_mid))."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.special import expit

from scalelab.analysis.stats import BOOTSTRAP_SEED, CONFIDENCE, Interval
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


def step_reaching(compute: Mapping[int, float], target: float) -> int | None:
    """The first step whose C (cumulative_compute) is at least `target`; None if no step's
    is. Where a curve's compute_at lands in a run (m4s9: the pilot's budget rule)."""
    return next((step for step, c in sorted(compute.items()) if c >= target), None)


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


# m4s12: one family of curves across model sizes
N_REF = 1e7  # sizes enter as x = log10(N / N_REF): m1 -1.10 .. m5 0.77, m7 1.31
MIN_SIZES = 3  # each line has two numbers: a third size is the first that can disagree
LOGIT_A_LIMIT = 10.0  # A's intercept within expit(-10) .. expit(10): 0.005% .. 99.995%
FAMILY_PARAMETERS = ("a0", "a1", "c0", "c1", "b0", "b1")  # least_squares' order


@dataclass(frozen=True)
class Family:
    """Each size N's curve from six numbers, every parameter a line in x = log10(N / N_REF):
    logit A = a0 + a1 x, ln C_mid = c0 + c1 x, ln B = b0 + b1 x. The logit keeps A within
    0 and 1 and the logs keep C_mid and B above 0 at any size, m6 and m7 included."""

    a0: float
    a1: float
    c0: float
    c1: float
    b0: float
    b1: float

    def curve(self, params: int, r0: float) -> Curve:
        """The curve of a model with `params` non-embedding parameters, from its warm
        start's R_0 (measured, never fitted)."""
        x = float(np.log10(params / N_REF))
        return Curve(
            r0=r0,
            a=float(expit(self.a0 + self.a1 * x)),
            c_mid=float(np.exp(self.c0 + self.c1 * x)),
            b=float(np.exp(self.b0 + self.b1 * x)),
        )


@dataclass(frozen=True)
class SizedRun:
    """One RL run's points (run_points, from step 0 on) and its model's size N."""

    params: int
    points: tuple[Point, ...]


@dataclass(frozen=True)
class FamilyFit:
    """A family fit to runs of several sizes at once."""

    family: Family
    rmse: float  # root mean square of R's residuals over every run's gradings after step 0
    points: int  # gradings fitted, over all runs
    sizes: int  # distinct N among the runs


def _start(runs: Sequence[SizedRun]) -> NDArray[np.float64]:
    """Where the joint fit starts: each run's own curve (fit_run), its logit A, ln C_mid and
    ln B averaged per size, then a straight line through the sizes for each."""
    by_size: dict[int, list[list[float]]] = {}
    for run in runs:
        curve = fit_run(run.points).curve
        a = float(np.clip(curve.a, 0.01, 0.99))  # a run's A may sit on 0 or 1: logit inf
        by_size.setdefault(run.params, []).append(
            [float(np.log(a / (1 - a))), float(np.log(curve.c_mid)), float(np.log(curve.b))]
        )
    x = np.log10(np.array(sorted(by_size), dtype=np.float64) / N_REF)
    ys = np.array([np.mean(by_size[n], axis=0) for n in sorted(by_size)])  # sizes x 3
    lines = [np.polyfit(x, ys[:, k], 1) for k in range(3)]  # each (slope, intercept)
    return np.array([value for slope, cut in lines for value in (cut, slope)])


def fit_family(runs: Sequence[SizedRun]) -> FamilyFit:
    """The six numbers of a Family by least squares on R over every run's gradings after
    step 0 at once, each run's R_0 held at its step-0 point's R. Starts from the per-size
    lines (_start) and from flat ones through their mean, and keeps the better fit."""
    sizes = {run.params for run in runs}
    if len(sizes) < MIN_SIZES:
        raise ValueError(f"a family needs runs of {MIN_SIZES} sizes or more, got {len(sizes)}")
    for run in runs:
        if not run.points or run.points[0].step != 0:
            raise ValueError("a run's points must start at step 0, the warm start's grading")
    rest = [(run, point) for run in runs for point in run.points[1:]]
    x = np.log10(np.array([run.params for run, _ in rest], dtype=np.float64) / N_REF)
    r0 = np.array([run.points[0].rate for run, _ in rest])
    log_c = np.log(np.array([point.compute for _, point in rest]))
    r = np.array([point.rate for _, point in rest])

    def residuals(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        a0, a1, c0, c1, b0, b1 = theta
        a = expit(a0 + a1 * x)
        b = np.exp(b0 + b1 * x)
        return r0 + (a - r0) * expit(b * (log_c - (c0 + c1 * x))) - r

    span = DECADES * np.log(10)
    inf = np.inf  # the slopes are free; the intercepts keep fit_curve's ranges
    lower = [-LOGIT_A_LIMIT, -inf, log_c.min() - span, -inf, np.log(B_BOUNDS[0]), -inf]
    upper = [LOGIT_A_LIMIT, inf, log_c.max() + span, inf, np.log(B_BOUNDS[1]), inf]
    lines = np.clip(_start(runs), np.add(lower, 1e-9), np.subtract(upper, 1e-9))
    flat = lines.copy()
    flat[1::2] = 0.0  # every slope 0: the same curve at every size
    best = None
    for start in (lines, flat):
        found = least_squares(residuals, start, bounds=(lower, upper))
        if best is None or found.cost < best.cost:
            best = found
    assert best is not None
    family = Family(*(float(value) for value in best.x))
    rmse = float(np.sqrt(np.mean(np.square(residuals(best.x)))))
    return FamilyFit(family=family, rmse=rmse, points=len(rest), sizes=len(sizes))


def resample_seeds(runs: Sequence[SizedRun], rng: np.random.Generator) -> list[SizedRun]:
    """One draw of the seed bootstrap (m4s12b): size by size, as many runs as that size has,
    drawn from its own runs with replacement. Sizes never mix: every draw keeps the ladder."""
    by_size: dict[int, list[SizedRun]] = {}
    for run in runs:
        by_size.setdefault(run.params, []).append(run)
    drawn: list[SizedRun] = []
    for _, mine in sorted(by_size.items()):
        drawn.extend(mine[int(i)] for i in rng.integers(len(mine), size=len(mine)))
    return drawn


def bootstrap_families(
    runs: Sequence[SizedRun], resamples: int, seed: int = BOOTSTRAP_SEED
) -> list[Family]:
    """The family refit to `resamples` draws of the seeds (resample_seeds): how far its six
    numbers, and all that is read off them, move with which seeds happened to run. The eval
    problems stay the same frozen set: the spread says nothing about another draw of them."""
    if resamples < 1:
        raise ValueError(f"resamples must be 1 or more, got {resamples}")
    rng = np.random.default_rng(seed)
    return [fit_family(resample_seeds(runs, rng)).family for _ in range(resamples)]


def spread(estimate: float, values: Sequence[float]) -> Interval:
    """The central CONFIDENCE share of bootstrap values (percentiles), around the estimate
    the fit to all the runs gave."""
    tail = 50 * (1 - CONFIDENCE)
    low, high = np.percentile(values, [tail, 100 - tail])
    return Interval(estimate=estimate, low=float(low), high=float(high))
