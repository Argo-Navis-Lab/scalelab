"""Reports read from what runs recorded. gains: how much RL changed each eval set's greedy
pass@1, from the warm start's grading to the last one, with confidence intervals over seeds
and eval problems."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from scalelab.analysis.stats import Interval, bootstrap_interval
from scalelab.rl.evaluation import EvalResult
from scalelab.rl.gradings import Grading

Verdict = Literal["improved", "worse", "unclear"]


@dataclass(frozen=True)
class Gain:
    """One eval set (or part), last grading against the first, over the seeds of one
    experiment: rates are means over seeds, the gain is last - first."""

    name: str  # such as "arithmetic-3/*"
    n: int  # problems graded
    first: float  # greedy pass@1 at step 0, the warm start's own
    last: float  # greedy pass@1 at each seed's last grading
    mean: Interval  # the gain: the mean over seeds, problems paired, and its 95% interval
    iqm: Interval  # the same gain as an interquartile mean over seeds

    @property
    def verdict(self) -> Verdict:
        """Improved only when the mean gain's whole interval is above 0, worse only when it is
        below 0: an interval that holds 0 can't tell."""
        if self.mean.low > 0:
            return "improved"
        if self.mean.high < 0:
            return "worse"
        return "unclear"


def _greedy(grading: Grading) -> dict[str, EvalResult]:
    return {result.name: result for result in grading.greedy}


def gains(runs: Sequence[Sequence[Grading]]) -> list[Gain]:
    """The greedy pass@1 gain of every eval set and part, given each seed's gradings in step
    order (load_gradings), in the order the gradings list them.

    Each seed's first grading must be step 0, and its last one is the end of its budget (seeds
    can stop at different steps). Each problem is compared with itself: a seed's per-problem
    differences, last - first, are what the bootstrap resamples.
    """
    if not runs:
        raise ValueError("gains needs the gradings of at least one run")
    pairs: list[tuple[dict[str, EvalResult], dict[str, EvalResult]]] = []
    for gradings in runs:
        if not gradings or gradings[0].step != 0:
            raise ValueError("each run's gradings must start at step 0, the warm start's")
        if gradings[-1].step == 0:
            raise ValueError(f"{gradings[0].run_id} has no grading after step 0")
        pairs.append((_greedy(gradings[0]), _greedy(gradings[-1])))
    sizes = {name: result.n for name, result in pairs[0][0].items()}
    for first, last in pairs:
        for results in (first, last):
            if {name: result.n for name, result in results.items()} != sizes:
                raise ValueError("every grading must grade the same eval sets and parts")
    found: list[Gain] = []
    for name in sizes:
        before = np.array([first[name].solved for first, _ in pairs], dtype=np.float64)
        after = np.array([last[name].solved for _, last in pairs], dtype=np.float64)
        differences = after - before  # seeds x problems, each -1, 0 or 1
        found.append(
            Gain(
                name=name,
                n=sizes[name],
                first=float(before.mean()),
                last=float(after.mean()),
                mean=bootstrap_interval(differences),
                iqm=bootstrap_interval(differences, statistic="iqm"),
            )
        )
    return found
