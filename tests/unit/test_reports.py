import pytest

from scalelab.analysis.reports import Gain, gains
from scalelab.analysis.stats import Interval, bootstrap_interval
from scalelab.rl.evaluation import EvalResult
from scalelab.rl.gradings import Grading


def grading(step: int, results: dict[str, str], seed: int = 0) -> Grading:
    """A grading whose greedy results are strings of outcomes, "1" for each problem solved."""
    greedy = tuple(
        EvalResult(name=name, solved=tuple(outcome == "1" for outcome in outcomes))
        for name, outcomes in results.items()
    )
    run_id = f"0123456789abcdef-s{seed}-01234567"
    return Grading(run_id=run_id, step=step, greedy=greedy, sampled=())


def test_the_gain_pairs_each_problem_with_itself():
    runs = [
        [grading(0, {"a": "0011"}), grading(10, {"a": "1011"})],
        [grading(0, {"a": "0011"}, seed=1), grading(7, {"a": "1110"}, seed=1)],
    ]
    (gain,) = gains(runs)
    differences = [[1, 0, 0, 0], [1, 1, 0, -1]]  # last - first, problem by problem
    assert (gain.name, gain.n, gain.first, gain.last) == ("a", 4, 0.5, 0.75)
    assert gain.mean == bootstrap_interval(differences)
    assert gain.iqm == bootstrap_interval(differences, statistic="iqm")
    assert gain.mean.estimate == 0.25


def test_the_last_grading_is_compared_not_one_in_the_middle():
    runs = [[grading(0, {"a": "00"}), grading(5, {"a": "11"}), grading(10, {"a": "01"})]]
    (gain,) = gains(runs)
    assert (gain.last, gain.mean.estimate) == (0.5, 0.5)


def test_every_eval_set_and_part_in_the_order_of_the_gradings():
    results = {"arithmetic-3": "0101", "arithmetic-3/*": "01", "arithmetic-3/+": "01"}
    runs = [[grading(0, results), grading(1, results)]]
    assert [gain.name for gain in gains(runs)] == list(results)


def test_a_clear_gain_is_improved_and_a_clear_loss_worse():
    before, after = "1" * 50 + "0" * 50, "1" * 70 + "0" * 30
    improved = [[grading(0, {"a": before}, s), grading(9, {"a": after}, s)] for s in range(3)]
    worse = [[grading(0, {"a": after}, s), grading(9, {"a": before}, s)] for s in range(3)]
    assert [gain.verdict for gain in gains(improved) + gains(worse)] == ["improved", "worse"]


@pytest.mark.parametrize(
    ("low", "high", "verdict"),
    [
        (0.01, 0.05, "improved"),
        (-0.05, -0.01, "worse"),
        (-0.01, 0.05, "unclear"),
        (0, 0.05, "unclear"),
    ],
)
def test_the_verdict_needs_the_whole_interval_on_one_side(low: float, high: float, verdict: str):
    interval = Interval(estimate=(low + high) / 2, low=low, high=high)
    gain = Gain(name="a", n=100, first=0.5, last=0.53, mean=interval, iqm=interval)
    assert gain.verdict == verdict


@pytest.mark.parametrize(
    ("runs", "match"),
    [
        ([], "at least one run"),
        ([[]], "start at step 0"),
        ([[grading(5, {"a": "01"}), grading(10, {"a": "11"})]], "start at step 0"),
        ([[grading(0, {"a": "01"})]], "no grading after step 0"),
        ([[grading(0, {"a": "01"}), grading(9, {"a": "011"})]], "same eval sets"),
        ([[grading(0, {"a": "01"}), grading(9, {"b": "01"})]], "same eval sets"),
    ],
)
def test_gradings_that_cannot_be_paired_are_refused(runs: list[list[Grading]], match: str):
    with pytest.raises(ValueError, match=match):
        gains(runs)
