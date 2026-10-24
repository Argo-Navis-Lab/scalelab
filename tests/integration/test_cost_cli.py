"""`scalelab cost calibrate` and `scalelab cost predict` end to end, on the CPU (m3s9)."""

from pathlib import Path
from typing import get_args

import pytest

from scalelab.cli import main
from scalelab.core.config import load_config
from scalelab.costmodel.calibrate import Calibration, load_calibration, save_calibration
from scalelab.costmodel.roofline import Efficiency, Group
from scalelab.costmodel.specs import CPU
from scalelab.rl.costing import predict_rl_run

REPO = Path(__file__).resolve().parents[2]
RL = REPO / "configs" / "experiments" / "rl.yaml"
SMALL = ["task.ops=*", "task.difficulties=[2]", "rl.max_steps=10"]


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fresh SCALELAB_HOME, run from the repo's folder (where configs/models is)."""
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path))
    monkeypatch.chdir(REPO)
    return tmp_path


def test_calibrate_fits_the_bench_results_and_saves_beside_them(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = home / "bench" / "cpu"
    assert main(["cost", "calibrate", "--hardware", "cpu"]) == 2
    assert f"no benchmark results in {folder}" in capsys.readouterr().err
    assert main(["bench", "--hardware", "cpu", "--models", "ladder_s/m1", "--suites", "copy"]) == 0
    assert main(["cost", "calibrate", "--hardware", "cpu"]) == 0
    calibration = load_calibration(folder / "calibration.json")
    assert calibration.hardware == "cpu"
    assert list(calibration.errors) == ["copy"]
    copy = calibration.efficiencies["copy"]
    out = capsys.readouterr().out
    assert f"samples:       8 from {folder}\n" in out
    assert f"{'copy':<15}{copy.compute:8.3f}{copy.memory:8.3f}" in out
    assert f"saved:         {folder / 'calibration.json'}\n" in out


def test_predict_without_a_calibration_uses_the_datasheet(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cost", "predict", str(RL), *SMALL, "--hardware", "cpu"]) == 0
    run = predict_rl_run(load_config(RL, SMALL), CPU)
    out = capsys.readouterr().out
    assert "calibration:   none, so the datasheet's peaks and free launches" in out
    assert f"{'step':<15}{run.step.seconds:8.3f}" in out
    assert "run:           10 steps, " in out
    assert f"(rl.max_steps: the FLOPs budget buys {run.budget_steps})\n" in out
    assert "session:       fits in one of 12 h" in out


def test_predict_uses_the_calibration_and_says_when_it_is_stale(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    slow = Efficiency(compute=0.01, memory=0.01, launch_s=1e-3)
    efficiencies = dict.fromkeys(get_args(Group), slow)
    path = home / "bench" / "cpu" / "calibration.json"
    path.parent.mkdir(parents=True)
    save_calibration(
        path,
        Calibration(
            hardware="cpu", versions={"torch": "0.0.1"}, efficiencies=efficiencies, errors={}
        ),
    )
    argv = ["cost", "predict", str(RL), *SMALL, "--hardware", "cpu", "--session-hours", "0.001"]
    assert main(argv) == 0
    run = predict_rl_run(load_config(RL, SMALL), CPU, efficiencies)
    out = capsys.readouterr().out
    assert f"calibration:   {path}\n" in out
    assert "warning:       calibration.json was fitted on torch 0.0.1, this is " in out
    assert f"{'step':<15}{run.step.seconds:8.3f}" in out
    assert f"session:       needs {run.seconds / 3.6:.1f} of 0.001 h" in out


def test_a_warm_start_has_nothing_to_predict(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    warm_start = REPO / "configs" / "experiments" / "warmstart.yaml"
    assert main(["cost", "predict", str(warm_start), "--hardware", "cpu"]) == 2
    assert "cost predict times RL steps, and" in capsys.readouterr().err
