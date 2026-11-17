"""Two algorithms compared at a matched point. Equal FLOPs, what rl.flops spends, is only one
way to line up two runs: an algorithm whose FLOPs leave some samples out (DAPO's dynamic
sampling drops flat groups before log-probs and training) gets more samples and more hours
for them. Here each run of one side is read at its grading nearest in samples, FLOPs or
seconds to its partner (the other side's run of the same size and seed) at the partner's
last grading, and the difference in R is bootstrapped over seeds and problems,
paired by seed and by problem."""

from collections.abc import Iterable, Mapping, Sequence
from itertools import accumulate
from typing import Literal

from scalelab.analysis.scaling import cumulative_compute
from scalelab.analysis.stats import Interval, bootstrap_interval
from scalelab.rl.gradings import Grading
from scalelab.telemetry.sink import MetricRow

Measure = Literal["samples", "flops", "seconds"]
MEASURES: tuple[Measure, ...] = ("samples", "flops", "seconds")
SECONDS_METRIC = "system/step_s"  # a step's wall-clock time; gradings and saves not in it


def cumulative_seconds(rows: Iterable[MetricRow]) -> dict[int, float]:
    """Seconds after each step, from a run's metrics: system/step_s summed over the steps up
    to it. Every step from 1 on must have it."""
    per_step = {
        row.step: row.value for row in rows if row.metric == SECONDS_METRIC and row.layer is None
    }
    steps = sorted(per_step)
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError("the metrics miss some steps' seconds")
    return dict(zip(steps, accumulate(per_step[step] for step in steps), strict=True))


def measured(
    measure: Measure, rows: Iterable[MetricRow], samples_per_step: int
) -> dict[int, float]:
    """The measure after each step, 0 at step 0. Samples are the completions sampled: every
    step samples prompts_per_step x group_size of them, kept or dropped later."""
    rows = list(rows)
    if measure == "flops":
        table = cumulative_compute(rows)
    elif measure == "seconds":
        table = cumulative_seconds(rows)
    else:
        table = {step: float(step * samples_per_step) for step in cumulative_compute(rows)}
    return {0: 0.0, **table}


def nearest(gradings: Sequence[Grading], at: Mapping[int, float], target: float) -> Grading:
    """The grading whose measure (`at` its step) is nearest `target`; of two as near, the
    earlier. Nearest, not "at most": a budget run's last step overshoots its budget a little,
    and "at most" would send its partner back a whole grading for it."""
    if not gradings:
        raise ValueError("the run has no gradings")
    for grading in gradings:
        if grading.step not in at:
            raise ValueError(f"the metrics stop before step {grading.step}, a graded one")
    return min(gradings, key=lambda grading: abs(at[grading.step] - target))


def pooled_solved(grading: Grading, ops: str) -> dict[str, tuple[bool, ...]]:
    """Each greedy part of `ops`'s problems (named "<eval set>/<op>", as pooled_rate reads
    them) and its problems' outcomes, in eval-set order."""
    parts = {
        result.name: result.solved
        for result in grading.greedy
        if "/" in result.name and result.name.rsplit("/", 1)[1] in set(ops)
    }
    if not parts:
        raise ValueError(f"the grading at step {grading.step} has no {ops} problems")
    return parts


def difference(
    base: Sequence[Mapping[str, tuple[bool, ...]]],
    other: Sequence[Mapping[str, tuple[bool, ...]]],
) -> Interval:
    """R of `other` minus R of `base`, pair k being one seed of each (pooled_solved): every
    problem's difference, seeds x problems, bootstrapped (stats.bootstrap_interval)."""
    if not base or len(base) != len(other):
        raise ValueError("one other run for each base run, and one pair or more")
    names = sorted(base[0])
    rows: list[list[int]] = []
    for mine, theirs in zip(base, other, strict=True):
        if sorted(mine) != names or sorted(theirs) != names:
            raise ValueError("the runs grade different problems")
        row: list[int] = []
        for name in names:
            if len(mine[name]) != len(theirs[name]):
                raise ValueError(f"the runs grade different problems in {name}")
            row.extend(int(b) - int(a) for a, b in zip(mine[name], theirs[name], strict=True))
        rows.append(row)
    return bootstrap_interval(rows)
