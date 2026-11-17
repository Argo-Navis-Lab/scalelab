"""Two algorithms at a matched point (m4s14): the measures, the nearest grading, and the
paired difference."""

import pytest

from scalelab.analysis.versus import cumulative_seconds, difference, measured, nearest
from scalelab.rl.evaluation import EvalResult
from scalelab.rl.gradings import Grading
from scalelab.telemetry.sink import MetricRow


def rows(steps: int, seconds: float = 2.0, flops: float = 10.0) -> list[MetricRow]:
    """Each step's three FLOPs metrics (flops each) and its seconds."""
    names = ("system/sample_flops", "system/logprobs_flops", "system/train_flops")
    every = range(1, steps + 1)
    found = [MetricRow("r", step, 0.0, name, None, flops) for step in every for name in names]
    return found + [MetricRow("r", step, 0.0, "system/step_s", None, seconds) for step in every]


def grading(step: int, solved: tuple[bool, ...] = (True,)) -> Grading:
    part = EvalResult(name="arithmetic-4/*", solved=solved)
    return Grading(run_id="r", step=step, greedy=(part,), sampled=())


def test_each_measure_adds_up_step_by_step():
    assert cumulative_seconds(rows(3)) == {1: 2.0, 2: 4.0, 3: 6.0}
    assert measured("seconds", rows(3), 8) == {0: 0.0, 1: 2.0, 2: 4.0, 3: 6.0}
    assert measured("flops", rows(2), 8) == {0: 0.0, 1: 30.0, 2: 60.0}
    assert measured("samples", rows(2, flops=99.0), 8) == {0: 0.0, 1: 8.0, 2: 16.0}
    with pytest.raises(ValueError, match="the metrics miss some steps' seconds"):
        cumulative_seconds([row for row in rows(3) if row.step != 2])


def test_the_nearest_grading_wins_and_ties_go_to_the_earlier():
    at = {0: 0.0, 10: 100.0, 20: 200.0}
    gradings = [grading(0), grading(10), grading(20)]
    assert nearest(gradings, at, 160.0).step == 20
    assert nearest(gradings, at, 140.0).step == 10
    assert nearest(gradings, at, 150.0).step == 10  # as near: the earlier
    assert nearest(gradings, at, 1e9).step == 20  # a little past the end: still the last
    with pytest.raises(ValueError, match="the metrics stop before step 30, a graded one"):
        nearest([*gradings, grading(30)], at, 0.0)
    with pytest.raises(ValueError, match="the run has no gradings"):
        nearest([], at, 0.0)


def test_the_difference_pairs_seeds_and_problems():
    base = [{"a/*": (True, False, False, False)}, {"a/*": (False, False, True, False)}]
    other = [{"a/*": (True, True, False, False)}, {"a/*": (True, False, True, True)}]
    gap = difference(base, other)
    assert gap.estimate == pytest.approx((1 / 4 + 2 / 4) / 2)
    assert gap.low <= gap.estimate <= gap.high
    assert difference(base, base).estimate == 0.0
    with pytest.raises(ValueError, match="one other run for each base run"):
        difference(base, other[:1])
    with pytest.raises(ValueError, match="the runs grade different problems"):
        difference(base, [{"b/*": (True,) * 4}, other[1]])
    with pytest.raises(ValueError, match="different problems in a/"):
        difference(base, [{"a/*": (True,)}, other[1]])
