"""DecisionReport (m3s15): a change's step against the baseline's, phase by phase, judged against
the model's validated error on that device."""

import pytest

from scalelab.costmodel.calibrate import Errors
from scalelab.costmodel.report import DecisionReport
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
