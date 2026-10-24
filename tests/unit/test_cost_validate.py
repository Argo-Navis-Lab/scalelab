"""The cost model's whole steps against measured ones (m3s10)."""

import pytest

from scalelab.costmodel.rlstep import Phase, PhaseTime, StepPrediction
from scalelab.costmodel.validate import PARTS, check_step, summarize
from scalelab.rl.costing import measured_step
from scalelab.telemetry.sink import MetricRow


def predicted(generate: float, logprobs: float, train: float) -> StepPrediction:
    """A step whose phases take these seconds."""
    phases: dict[Phase, float] = {"generate": generate, "logprobs": logprobs, "train": train}
    return StepPrediction(
        hardware="cpu",
        phases={
            name: PhaseTime(by_bound={"compute": seconds}, by_class={"gemm": seconds}, flops=0.0)
            for name, seconds in phases.items()
        },
    )


def measured(generate: float, logprobs: float, train: float, step: float) -> dict[str, float]:
    return {"generate": generate, "logprobs": logprobs, "train": train, "step": step}


def row(step: int, metric: str, value: float) -> MetricRow:
    return MetricRow(run_id="r", step=step, wall_time=0.0, metric=metric, layer=None, value=value)


def test_a_check_compares_each_phase_and_the_whole_step() -> None:
    check = check_step("r", predicted(0.3, 0.25, 0.6), 9, measured(0.4, 0.25, 0.5, 1.25))
    assert check.steps == 9
    assert check.predicted == pytest.approx(measured(0.3, 0.25, 0.6, 1.15))
    assert check.errors == pytest.approx(measured(-0.25, 0.0, 0.2, 1.15 / 1.25 - 1))


def test_every_part_must_be_measured() -> None:
    with pytest.raises(ValueError, match="r: no measured logprobs, step"):
        check_step("r", predicted(1.0, 1.0, 1.0), 3, {"generate": 1.0, "train": 1.0})


def test_the_summary_counts_each_run_once_in_each_part() -> None:
    a = check_step("a", predicted(0.3, 0.2, 0.6), 9, measured(0.4, 0.2, 0.5, 1.0))
    b = check_step("b", predicted(0.5, 0.2, 0.4), 9, measured(0.5, 0.25, 0.5, 1.375))
    # errors: a -25%, 0%, +20%, +10%; b 0%, -20%, -20%, -20%
    summary = summarize([a, b])
    assert list(summary) == list(PARTS)
    expected = {  # MAPE, bias, worst (the first of the largest size)
        "generate": (0.125, -0.125, -0.25),
        "logprobs": (0.1, -0.1, -0.2),
        "train": (0.2, 0.0, 0.2),
        "step": (0.15, -0.05, -0.2),
    }
    for part, (mape, bias, worst) in expected.items():
        errors = summary[part]
        assert errors.samples == 2
        assert (errors.mape, errors.bias, errors.worst) == pytest.approx((mape, bias, worst))


def test_measured_steps_leave_the_first_out_and_average_the_rest() -> None:
    scales = {"generate": 1.0, "logprobs": 2.0, "train": 3.0, "step": 7.0}
    rows = [
        row(step, f"system/{part}_s", step * scales[part]) for step in (1, 2, 3) for part in PARTS
    ]
    rows += [row(2, "system/data_s", 5.0), row(3, "optim/loss", 9.0)]  # neither is timed
    steps, means = measured_step(rows)
    assert steps == 2
    assert means == pytest.approx(measured(2.5, 5.0, 7.5, 17.5))
    assert measured_step(rows, skip=0) == (3, pytest.approx(measured(2.0, 4.0, 6.0, 14.0)))
    assert measured_step([row(1, "system/step_s", 1.0)]) == (0, {})
