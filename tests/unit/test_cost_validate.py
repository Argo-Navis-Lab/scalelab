"""The cost model's whole steps against measured ones (m3s10)."""

import pytest

from scalelab.costmodel.calibrate import Errors, errors_between
from scalelab.costmodel.rlstep import Phase, PhaseTime, StepPrediction
from scalelab.costmodel.validate import (
    BAND_MIN_RUNS,
    PARTS,
    Validation,
    band,
    check_step,
    summarize,
)
from scalelab.rl.costing import measured_step
from scalelab.telemetry.detectors import MAD_TO_STD
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


# --- added in m3s11: a warm start's step, train alone ---------------------------------------


def warm_start(train: float) -> StepPrediction:
    """A warm start's step: a train phase alone."""
    phase = PhaseTime(by_bound={"compute": train}, by_class={"gemm": train}, flops=0.0)
    return StepPrediction(hardware="cpu", phases={"train": phase})


def test_a_warm_start_is_checked_on_its_train_phase_and_step() -> None:
    check = check_step("w", warm_start(0.6), 9, {"train": 0.5, "step": 0.8, "generate": 1.0})
    assert check.measured == {"train": 0.5, "step": 0.8}  # only the parts it predicts
    assert check.errors == pytest.approx({"train": 0.2, "step": -0.25})
    with pytest.raises(ValueError, match="w: no measured step"):
        check_step("w", warm_start(0.6), 9, {"train": 0.5})


def test_each_part_is_summed_up_over_the_runs_that_time_it() -> None:
    rl = check_step("a", predicted(0.3, 0.2, 0.6), 9, measured(0.4, 0.2, 0.5, 1.0))
    warm = check_step("w", warm_start(0.6), 9, {"train": 0.5, "step": 0.8})
    summary = summarize([rl, warm])
    samples = {part: errors.samples for part, errors in summary.items()}
    assert samples == {"generate": 1, "logprobs": 1, "train": 2, "step": 2}
    assert summary["step"].mape == pytest.approx(0.175)  # step: +10% and -25%
    assert list(summarize([warm])) == ["train", "step"]


# m3s25: the errors' median and spread, for the throughput detector.


def test_errors_keep_a_median_and_spread_that_a_wild_run_barely_moves() -> None:
    errors = errors_between([1.1, 0.9, 1.0], [1.0, 1.0, 1.0])  # errors +10%, -10%, 0
    assert errors.median == pytest.approx(0.0, abs=1e-12)
    assert errors.spread == pytest.approx(MAD_TO_STD * 0.1)
    wild = errors_between([1.1, 0.9, 1.0, 0.1], [1.0] * 4)  # and a run at -90%
    assert wild.median == pytest.approx(-0.05)
    assert wild.spread == pytest.approx(MAD_TO_STD * 0.1)  # the MAD of 0.15, 0.05, 0.05, 0.85
    old = '{"samples": 28, "mape": 0.04, "bias": 0.006, "worst": -0.101}'  # saved before m3s25
    assert Errors.model_validate_json(old).spread is None


def test_the_band_is_the_runtimes_own_step_errors_given_enough_runs() -> None:
    def errors(runs: int, median: float | None, spread: float | None) -> Errors:
        return Errors(samples=runs, mape=0.1, bias=-0.1, worst=-0.2, median=median, spread=spread)

    validation = Validation(
        hardware="t4",
        errors={
            "step": errors(20, 0.01, 0.05),
            "step, fp16": errors(BAND_MIN_RUNS, -0.13, 0.04),
            "step, fp32 cut": errors(BAND_MIN_RUNS - 1, 0.0, 0.01),
        },
    )
    assert band(validation, "fp16") == ("step, fp16", -0.13, 0.04)
    assert band(validation, "fp32 cut") == ("step", 0.01, 0.05)  # too few runs of its own
    assert band(validation, "fp32") == ("step", 0.01, 0.05)
    assert band(Validation(hardware="t4", errors={"step": errors(20, None, None)}), "fp32") is None
