import numpy as np
import pytest

from scalelab.analysis.scaling import (
    Curve,
    Point,
    cumulative_compute,
    fit_curve,
    fit_run,
    pooled_rate,
    run_points,
)
from scalelab.rl.evaluation import EvalResult
from scalelab.rl.experiment import geometric_steps
from scalelab.rl.gradings import Grading
from scalelab.telemetry.sink import MetricRow

STEP_FLOPS = 1e12  # each step's compute in the made-up runs
STEPS = geometric_steps(4, 1000)  # graded as an M4 run is: 1 2 3 6 10 18 ... 562 1000
COMPUTE = [STEP_FLOPS * step for step in STEPS]
TRUE = Curve(r0=0.15, a=0.6, c_mid=1e14, b=1.5)  # halfway at step 100
PROBLEMS = 653  # the M4 task's: * at bands 4 and 5 (316 + 337)
METRICS = ("system/sample_flops", "system/logprobs_flops", "system/train_flops", "system/step_s")


def test_the_curve_starts_at_r0_is_halfway_at_c_mid_and_levels_off_at_a():
    assert TRUE(0.0) == pytest.approx(0.15)
    assert TRUE(1e14) == pytest.approx((0.15 + 0.6) / 2)
    assert TRUE(1e30) == pytest.approx(0.6)
    assert TRUE.compute_at(0.5) == pytest.approx(1e14)
    assert TRUE(TRUE.compute_at(0.9)) == pytest.approx(0.15 + 0.9 * (0.6 - 0.15))
    with pytest.raises(ValueError, match="between 0 and 1"):
        TRUE.compute_at(1.0)


@pytest.mark.parametrize(
    "true",
    [
        TRUE,
        Curve(r0=0.4, a=0.1, c_mid=3e12, b=0.8),  # RL made it worse
        Curve(r0=0.05, a=0.9, c_mid=5e14, b=3.0),  # late and sharp
        Curve(r0=0.3, a=0.35, c_mid=2e13, b=1.0),  # a small gain
    ],
)
def test_a_fit_recovers_the_curve_its_points_came_from(true: Curve):
    fit = fit_curve(COMPUTE, true(COMPUTE).tolist(), true.r0)
    assert fit.curve.r0 == true.r0
    assert fit.curve.a == pytest.approx(true.a, rel=1e-6)
    assert fit.curve.c_mid == pytest.approx(true.c_mid, rel=1e-6)
    assert fit.curve.b == pytest.approx(true.b, rel=1e-6)
    assert fit.rmse < 1e-9
    assert fit.points == len(STEPS)
    assert fit.bounded == ()


def test_a_fit_says_which_parameters_the_gradings_leave_on_a_bound():
    """A rise sharper than B's bound, all of it between two gradings: B can't follow."""
    step = Curve(r0=0.1, a=0.5, c_mid=4e13, b=50.0)
    assert fit_curve(COMPUTE, step(COMPUTE).tolist(), step.r0).bounded == ("B",)


def test_a_fit_finds_the_curve_through_binomial_noise():
    """Each grading, step 0's too, is PROBLEMS problems solved or not: the noise a real run's
    gradings have (a standard error near 2 pp). Over 1000 seeds (m4s8) the worst were: A off
    by 0.088, C_mid by 0.20 decades, B 0.64x to 1.73x, the rms 0.005 to 0.037."""
    for seed in range(20):
        rng = np.random.default_rng(seed)
        r0 = rng.binomial(PROBLEMS, TRUE.r0) / PROBLEMS
        rates = rng.binomial(PROBLEMS, TRUE(COMPUTE)) / PROBLEMS
        fit = fit_curve(COMPUTE, rates.tolist(), r0)
        assert fit.curve.a == pytest.approx(TRUE.a, abs=0.1)
        assert np.log10(fit.curve.c_mid) == pytest.approx(np.log10(TRUE.c_mid), abs=0.25)
        assert TRUE.b / 2 < fit.curve.b < TRUE.b * 2
        assert 0.003 < fit.rmse < 0.05


@pytest.mark.parametrize(
    ("compute", "rates", "message"),
    [
        ([1.0, 2.0], [0.1, 0.2], "needs 3 gradings after step 0 or more, got 2"),
        ([1.0, 2.0, 3.0], [0.1, 0.2], "the same length"),
        ([0.0, 2.0, 3.0], [0.1, 0.2, 0.3], "above 0 and increasing"),
        ([1.0, 3.0, 2.0], [0.1, 0.2, 0.3], "above 0 and increasing"),
    ],
)
def test_a_fit_refuses_points_it_cannot_fit(compute: list[float], rates: list[float], message: str):
    with pytest.raises(ValueError, match=message):
        fit_curve(compute, rates, 0.1)


def rows(flops: dict[int, tuple[float, ...]]) -> list[MetricRow]:
    """A run's metrics: each step's sampling, log-prob and training FLOPs, and its time."""
    found: list[MetricRow] = []
    for step, values in flops.items():
        for metric, value in zip(METRICS, values, strict=False):
            found.append(MetricRow("r", step, 0.0, metric, None, value))
    return found


def test_compute_adds_up_each_steps_flops():
    flops = {1: (1.0, 2.0, 3.0, 9.0), 2: (1.0, 2.0, 3.0, 9.0), 3: (2.0, 2.0, 2.0, 9.0)}
    assert cumulative_compute(rows(flops)) == {1: 6.0, 2: 12.0, 3: 18.0}
    assert cumulative_compute([]) == {}
    with pytest.raises(ValueError, match="miss some steps"):
        cumulative_compute(rows({1: flops[1], 3: flops[3]}))
    with pytest.raises(ValueError, match="miss some steps"):
        cumulative_compute(rows({1: flops[1], 2: (1.0, 2.0)}))  # no training FLOPs


def grading(step: int, results: dict[str, str]) -> Grading:
    """A grading whose results are given as outcomes, such as "0110"."""
    greedy = tuple(
        EvalResult(name=name, solved=tuple(outcome == "1" for outcome in outcomes))
        for name, outcomes in results.items()
    )
    return Grading(run_id="r", step=step, greedy=greedy, sampled=())


def test_r_pools_the_problems_of_the_tasks_operations():
    graded = grading(
        5,
        {
            "arithmetic-4": "1100",
            "arithmetic-4/+": "11",
            "arithmetic-4/*": "00",
            "arithmetic-5": "101",
            "arithmetic-5/*": "101",
        },
    )
    assert pooled_rate(graded, "*") == (2 / 5, 5)
    assert pooled_rate(graded, "+*") == (4 / 7, 7)
    with pytest.raises(ValueError, match="the grading at step 5 has no - problems"):
        pooled_rate(graded, "-")


def test_a_runs_gradings_become_points_and_its_curve():
    flops = rows({1: (1.0, 2.0, 3.0), 2: (1.0, 2.0, 3.0), 3: (2.0, 2.0, 2.0)})
    gradings = [
        grading(0, {"arithmetic-4/*": "0001"}),
        grading(1, {"arithmetic-4/*": "0011"}),
        grading(3, {"arithmetic-4/*": "0111"}),
    ]
    assert run_points(gradings, flops, "*") == [
        Point(0, 0.0, 0.25, 4),
        Point(1, 6.0, 0.5, 4),
        Point(3, 18.0, 0.75, 4),
    ]
    with pytest.raises(ValueError, match="the metrics stop before step 4, a graded one"):
        run_points([*gradings, grading(4, {"arithmetic-4/*": "1111"})], flops, "*")

    points = [Point(0, 0.0, TRUE.r0, PROBLEMS)] + [
        Point(step, c, rate, PROBLEMS)
        for step, c, rate in zip(STEPS, COMPUTE, TRUE(COMPUTE).tolist(), strict=True)
    ]
    assert fit_run(points).curve.c_mid == pytest.approx(TRUE.c_mid, rel=1e-6)
    with pytest.raises(ValueError, match="must start at step 0"):
        fit_run(points[1:])
