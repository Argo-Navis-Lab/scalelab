"""`scalelab cost calibrate`, `cost predict` and `cost compare` (m3s15) end to end, on the CPU
(m3s9)."""

from pathlib import Path
from typing import get_args

import pytest

from scalelab.cli import main
from scalelab.core.config import load_config
from scalelab.core.hashing import config_hash
from scalelab.costmodel.calibrate import Calibration, Errors, load_calibration, save_calibration
from scalelab.costmodel.report import DecisionReport
from scalelab.costmodel.roofline import Efficiency, Group
from scalelab.costmodel.specs import CPU
from scalelab.costmodel.validate import Validation
from scalelab.rl.costing import padding_cut, predict_rl_run, predict_step

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


def test_compare_prints_and_saves_a_decision_report(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cost", "compare", str(RL), *SMALL, "--hardware", "cpu"]) == 0
    config = load_config(RL, SMALL)
    stem = home / "reports" / f"padding-cut-{config_hash(config)}-cpu"
    report = DecisionReport.model_validate_json(stem.with_suffix(".json").read_text())
    assert report.baseline == predict_step(config, CPU)
    assert report.candidate == padding_cut(config, CPU)
    assert report.validated is None  # a fresh home: nothing validated
    out = capsys.readouterr().out
    assert "validation:    none (cost validate saves one)\n" in out
    assert stem.with_suffix(".md").read_text() in out
    assert f"saved:         {stem}.md and .json\n" in out
    path = home / "bench" / "cpu" / "validation.json"  # as cost validate saves it
    errors = Errors(samples=15, mape=0.299, bias=-0.299, worst=-0.862)
    path.parent.mkdir(parents=True)
    path.write_text(Validation(hardware="cpu", errors={"step": errors}).model_dump_json())
    assert main(["cost", "compare", str(RL), *SMALL, "--hardware", "cpu"]) == 0
    assert f"validation:    {path}\n" in capsys.readouterr().out
    report = DecisionReport.model_validate_json(stem.with_suffix(".json").read_text())
    assert report.validated == errors
    assert main(["cost", "compare", str(RL), "--hardware", "cpu", "--micro-batch", "0"]) == 2
    assert "--micro-batch must be at least 1, not 0" in capsys.readouterr().err


def test_compare_fp16_needs_fp16_efficiencies(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["cost", "compare", str(RL), *SMALL, "--hardware", "cpu", "--change", "fp16"]
    assert main(argv) == 0  # no calibration: the datasheet's peaks
    config = load_config(RL, SMALL)
    stem = home / "reports" / f"fp16-{config_hash(config)}-cpu"
    report = DecisionReport.model_validate_json(stem.with_suffix(".json").read_text())
    assert report.change == "fp16"
    assert report.baseline == predict_step(config, CPU)
    assert report.candidate == predict_step(config, CPU, precision="fp16")
    assert stem.with_suffix(".md").read_text() in capsys.readouterr().out
    assert main([*argv, "--micro-batch", "32"]) == 2
    assert "--micro-batch sizes the padding cut's pieces, not fp16's" in capsys.readouterr().err
    fitted = Efficiency(compute=0.5, memory=0.5, launch_s=1e-5)
    fp32 = {name: fitted for name in get_args(Group) if not name.endswith(".fp16")}
    path = home / "bench" / "cpu" / "calibration.json"  # fitted before any fp16 bench
    path.parent.mkdir(parents=True)
    save_calibration(path, Calibration(hardware="cpu", versions={}, efficiencies=fp32, errors={}))
    assert main(argv) == 2
    assert (
        "the cpu calibration has no attention.fp16, gemm.fp16, gemv-decode.fp16: run `scalelab "
        "bench --hardware cpu --precision fp16`, then `scalelab cost calibrate --hardware cpu`"
    ) in capsys.readouterr().err
