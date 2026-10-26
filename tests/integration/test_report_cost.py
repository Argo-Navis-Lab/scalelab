from pathlib import Path

import pytest

from scalelab.analysis.costreport import (
    CHANGES_FIGURE,
    PHASES_FIGURE,
    RL_PHASES,
    STEPS_FIGURE,
    TABLES,
)
from scalelab.cli import main
from scalelab.costmodel.calibrate import Errors
from scalelab.costmodel.report import DecisionReport, Measurement
from scalelab.costmodel.rlstep import Phase, PhaseTime, StepPrediction
from scalelab.costmodel.validate import StepCheck, Validation

BAND = Errors(samples=5, mape=0.04, bias=0.0, worst=0.1, median=0.0, spread=0.03)
PARTS = {"generate": 1.0, "logprobs": 0.2, "train": 0.4, "step": 1.6}


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path))
    return tmp_path


def save_validation(home: Path, hardware: str, runs: bool = True) -> None:
    check = StepCheck(run_id="rl-run", steps=10, predicted=PARTS, measured=PARTS)
    validation = Validation(
        hardware=hardware, errors={"step": BAND}, checks=[check] if runs else []
    )
    folder = home / "bench" / hardware
    folder.mkdir(parents=True)
    (folder / "validation.json").write_text(validation.model_dump_json())


def save_measurement(home: Path, hardware: str) -> None:
    phases: dict[Phase, PhaseTime] = {
        name: PhaseTime(by_bound={"launch": PARTS[name]}, by_class={"gemm": PARTS[name]}, flops=1e9)
        for name in RL_PHASES
    }
    prediction = StepPrediction(hardware=hardware, phases=phases)
    report = DecisionReport(
        change="fp16",
        experiment="configs/experiments/rl.yaml",
        config_hash="a" * 16,
        hardware=hardware,
        baseline=prediction,
        candidate=prediction,
        validated=BAND,
    )
    measurement = Measurement(report=report, steps=20, baseline=PARTS, candidate=PARTS)
    folder = home / "reports"
    folder.mkdir(exist_ok=True)
    (folder / f"fp16-{'a' * 16}-{hardware}-measured.json").write_text(measurement.model_dump_json())


def test_report_cost_draws_each_devices_runs_and_measured_changes(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    save_validation(home, "t4")
    save_measurement(home, "t4")
    save_measurement(home, "a100-40g")  # a device not asked for
    out = home / "figs"
    assert main(["report", "cost", "--hardware", "t4", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert f"validation:    {home / 'bench' / 't4' / 'validation.json'} (1 runs)\n" in text
    assert f"measured:      1 changes in {home / 'reports'}\n" in text
    names = [STEPS_FIGURE, PHASES_FIGURE, CHANGES_FIGURE, TABLES]
    assert sorted(path.name for path in out.iterdir()) == sorted(names)


def test_report_cost_needs_a_device_and_its_runs_steps(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["report", "cost"]) == 2
    assert "report cost needs --hardware" in capsys.readouterr().err
    assert main(["report", "cost", "--hardware", "t4"]) == 2
    assert "run scalelab cost validate --hardware t4 first" in capsys.readouterr().err
    save_validation(home, "t4", runs=False)  # as cost validate saved it before m3s26
    assert main(["report", "cost", "--hardware", "t4"]) == 2
    assert "rerun scalelab cost validate --hardware t4" in capsys.readouterr().err
