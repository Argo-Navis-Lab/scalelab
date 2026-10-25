"""DecisionReport (m3s15): a change's step against the baseline's, phase by phase, judged against
the model's validated error on that device."""

import pytest

from scalelab.costmodel.calibrate import Errors
from scalelab.costmodel.report import DecisionReport, Measurement
from scalelab.costmodel.rlstep import PhaseTime, StepPrediction

ERRORS = Errors(samples=28, mape=0.04, bias=0.006, worst=-0.101)  # the T4's at m3s12


def step(generate: float, train: float) -> StepPrediction:
    return StepPrediction(
        hardware="t4",
        phases={
            "generate": PhaseTime(
                by_bound={"memory": generate}, by_class={"gemv-decode": generate}, flops=1e12
            ),
            "train": PhaseTime(
                by_bound={"compute": train}, by_class={"gemm": train}, flops=4e12 * train
            ),
        },
    )


def report(candidate: StepPrediction, validated: Errors | None) -> DecisionReport:
    return DecisionReport(
        change="padding-cut",
        experiment="rl",
        config_hash="47b519376e33a2b5",
        hardware="t4",
        baseline=step(1.0, 2.0),
        candidate=candidate,
        validated=validated,
    )


def test_a_report_shows_each_phase_and_the_whole_step() -> None:
    decision = report(step(1.0, 1.0), ERRORS)
    assert decision.speedup == 1.5
    assert decision.saving == pytest.approx(1 / 3)
    assert decision.worth_building
    text = decision.markdown()
    assert text.startswith("# DecisionReport: padding-cut, rl on t4\n")
    assert "| generate | 1.0000 | 1.0000 | +0.0% | memory | gemv-decode |\n" in text
    assert "| train | 2.0000 | 1.0000 | -50.0% | compute | gemm |\n" in text
    assert "| step | 3.0000 | 2.0000 | -33.3% | | |\n" in text
    assert "- speedup: 1.50x, 1,200 -> 1,800 steps an hour\n" in text
    assert "- FLOPs a step: 9e+12 -> 5e+12 (-44.4%)" in text
    assert "(bias +0.6%, worst -10.1%, 28 runs): the candidate's step 1.9200-2.0800 s\n" in text
    assert text.endswith("- verdict: worth building: the saving, 33.3%, is above the error\n")


def test_a_saving_within_the_error_is_not_worth_building() -> None:
    decision = report(step(1.0, 1.9), ERRORS)  # it saves 3.3%; whole steps miss by 4% on average
    assert not decision.worth_building
    assert "not worth building: the saving, 3.3%, is within the error\n" in decision.markdown()
    unvalidated = report(step(1.0, 1.0), None)  # saves a third, on a device nothing validated
    assert not unvalidated.worth_building
    expected = "- error band: none, t4 has no validation (cost validate saves one)\n"
    assert expected in unvalidated.markdown()


def test_a_report_reads_back_from_its_json() -> None:
    decision = report(step(1.0, 1.5), ERRORS)
    assert DecisionReport.model_validate_json(decision.model_dump_json()) == decision


def test_a_change_slower_by_more_than_the_error_says_so() -> None:
    # m3s16: the A100's m1 warm-start cut (+123.8%) read "the saving, -123.8%, is within"
    decision = report(step(1.0, 3.0), ERRORS)  # 4 s a step instead of 3
    assert not decision.worth_building
    assert "not worth building: it is 33.3% slower, beyond the error\n" in decision.markdown()


# --- added in m3s17: a change, measured -----------------------------------------------------


def measured(validated: Errors | None, without: float, with_cut: float) -> Measurement:
    """The report above (3.0 -> 2.0 s predicted, 1.50x), and two runs' steps as measured."""
    return Measurement(
        report=report(step(1.0, 1.0), validated),
        steps=20,
        baseline={"generate": 1.1, "train": without - 1.3, "step": without},
        candidate={"generate": 1.1, "train": with_cut - 1.3, "step": with_cut},
    )


def test_a_measurement_sets_each_part_beside_its_prediction() -> None:
    measurement = measured(ERRORS, 3.5, 2.5)
    assert measurement.speedup == pytest.approx(1.4)
    assert measurement.saving == pytest.approx(2 / 7)
    assert measurement.error == pytest.approx(1.5 / 1.4 - 1)  # +7.1%: the model expected more
    assert not measurement.held  # beyond the 4% band
    text = measurement.markdown()
    assert text.startswith("# Measured: padding-cut, rl on t4\n")
    assert "| generate | 1.0000 | 1.0000 | +0.0% | 1.1000 | 1.1000 | +0.0% |\n" in text
    assert "| train | 2.0000 | 1.0000 | -50.0% | 2.2000 | 1.2000 | -45.5% |\n" in text
    assert "| step | 3.0000 | 2.0000 | -33.3% | 3.5000 | 2.5000 | -28.6% |\n" in text
    assert "- speedup: predicted 1.50x, measured 1.40x, so the prediction is off by +7.1%\n" in text
    assert text.endswith("- verdict: the prediction missed: beyond the 4.0% error band\n")
    assert Measurement.model_validate_json(measurement.model_dump_json()) == measurement


def test_a_prediction_within_the_error_band_held() -> None:
    measurement = measured(ERRORS, 3.1, 2.1)  # 1.48x measured: 1.50x is off by +1.6%
    assert measurement.held
    assert "- verdict: the prediction held: within the 4.0% error band\n" in measurement.markdown()
    unvalidated = measured(None, 3.1, 2.1)
    assert not unvalidated.held
    expected = "- verdict: unjudged: t4 has no validation (cost validate saves one)\n"
    assert unvalidated.markdown().endswith(expected)


def test_an_fp16_report_names_its_change() -> None:
    decision = report(step(1.0, 1.0), ERRORS).model_copy(update={"change": "fp16"})
    text = decision.markdown()
    assert text.startswith("# DecisionReport: fp16, rl on t4\n")
    assert "The change: autocast float16 (m3s18's graph): matmuls, attention, " in text
    assert "(-44.4%): the same matmuls (casts do none), so budgets buy the same steps\n" in text
    assert "padding" not in text
