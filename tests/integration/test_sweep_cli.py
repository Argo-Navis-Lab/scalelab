"""`scalelab sweep plan` end to end (m4s4), on the datasheet's cost model."""

from pathlib import Path

import pytest
import yaml

from scalelab.cli import main
from scalelab.core.config import RuntimeConfig, load_config
from scalelab.core.hashing import config_hash
from scalelab.costmodel.specs import CPU
from scalelab.rl.costing import predict_job

REPO = Path(__file__).resolve().parents[2]
RL = REPO / "configs" / "experiments" / "rl.yaml"
SMALL = ["task.ops=*", "task.difficulties=[2]", "rl.max_steps=10"]


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fresh SCALELAB_HOME (no calibrations), run from the repo's folder."""
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(REPO)
    return tmp_path / "home"


def write_sweep(folder: Path, **fields: object) -> Path:
    """A sweep of 10-step RL runs of 2-digit multiplication, with `fields` on top."""
    path = folder / "sweep.yaml"
    sweep = {"name": "try", "experiment": str(RL), "overrides": SMALL, **fields}
    path.write_text(yaml.safe_dump(sweep))
    return path


def test_plan_prices_every_job_in_order(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_sweep(tmp_path, axes={"algo": ["algo=grpo", "algo=rloo"]}, seeds=[0, 1])
    assert main(["sweep", "plan", str(path), "--hardware", "cpu"]) == 0
    out = capsys.readouterr().out
    assert f"sweep:         try: 4 jobs of {RL}\n" in out
    assert "calibration:   none, so the datasheet's peaks and free launches" in out
    total = 0.0
    for number, (algo, seed) in enumerate([("grpo", 0), ("rloo", 0), ("grpo", 1), ("rloo", 1)], 1):
        config = load_config(RL, [*SMALL, f"algo={algo}"])
        job = predict_job(config, RuntimeConfig(), CPU)
        total += job.seconds
        assert (
            f"{number:>3}  {config_hash(config):<18}{seed:>4}  fp32{10:>13}{job.step:>8.3f}" in out
        )
    assert (
        f"total:         {total / 3600:.1f} h predicted, {total * 1.35 / 3600:.1f} h at 1.35x\n"
        in out
    )
    assert "sessions:      every job fits in one 12 h session at 1.35x\n" in out
    assert "memory:        every job fits in the " in out


def test_plan_refuses_two_jobs_that_are_the_same_run(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # runtime settings are never hashed: the same config and seed in fp16 is the same run
    axes = {"precision": ["runtime.precision=fp32", "runtime.precision=fp16"]}
    assert main(["sweep", "plan", str(write_sweep(tmp_path, axes=axes)), "--hardware", "cpu"]) == 2
    config = load_config(RL, SMALL)
    err = capsys.readouterr().err
    assert f"jobs 1 and 2 are the same run: config {config_hash(config)}, seed 0\n" in err


def test_plan_names_the_job_that_does_not_compose(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_sweep(tmp_path, axes={"model": ["model=ladder_s/m1", "model=ladder_s/m9"]})
    assert main(["sweep", "plan", str(path), "--hardware", "cpu"]) == 2
    err = capsys.readouterr().err
    assert f"job 2 ({' '.join(SMALL)} model=ladder_s/m9 seed=0): " in err


def test_plan_refuses_a_bad_sweep_file(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_sweep(tmp_path, seeds=[0, 0])
    assert main(["sweep", "plan", str(path), "--hardware", "cpu"]) == 2
    assert f"{path}: 1 validation error for Sweep" in capsys.readouterr().err


def test_a_job_too_big_for_the_device_exits_1(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_sweep(tmp_path, axes={"width": ["model.d_model=128", "model.d_model=4096"]})
    assert main(["sweep", "plan", str(path), "--hardware", "t4"]) == 1
    out = capsys.readouterr().out
    assert out.count("too big\n") == 1
    assert "memory:        job 2 needs more than the 13.4 GiB usable\n" in out


def test_every_sweep_file_plans_on_the_t4(home: Path) -> None:
    paths = sorted((REPO / "configs" / "sweeps").glob("*.yaml"))
    assert [path.stem for path in paths] == ["local", "warmstarts"]
    for path in paths:
        assert main(["sweep", "plan", str(path), "--hardware", "t4"]) == 0
