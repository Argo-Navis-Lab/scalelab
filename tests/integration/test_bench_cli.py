"""`scalelab bench` end to end: a suite on the CPU, a dry run, and the GPU checks."""

from pathlib import Path

import pytest
import torch

from scalelab.bench.elementwise import COPY_SIZES, copy_op
from scalelab.bench.suites import load_env, load_results
from scalelab.bench.timing import Measurement
from scalelab.cli import main
from scalelab.core.envcapture import capture_git

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fresh SCALELAB_HOME, run from the repo's folder (where configs/models is)."""
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path))
    monkeypatch.chdir(REPO)
    return tmp_path


def test_bench_saves_each_suite_it_runs(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "--hardware", "cpu", "--models", "ladder_s/m1", "--suites", "copy"]) == 0
    folder = home / "bench" / "cpu"
    assert [path.name for path in folder.iterdir()] == ["copy.parquet"]
    results = load_results(folder / "copy.parquet", Measurement)
    assert [result.op for result in results] == [copy_op(size) for size in COPY_SIZES]
    assert load_env(folder / "copy.parquet").git == capture_git()
    out = capsys.readouterr().out
    assert f"results:       {folder}\n" in out
    assert "copy:          8 cases: 8 timings in " in out


def test_a_dry_run_times_nothing(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["bench", "--hardware", "t4", "--models", "ladder_s/m6", "ladder_s/m7", "--dry-run"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "(dry run: nothing is timed or saved)\n" in out
    assert "decode:        28 cases (4 more too big for t4)\n" in out
    assert not (home / "bench").exists()


def test_the_gpu_must_be_the_one_named(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["bench", "--hardware", "t4", "--suites", "copy"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(argv) == 2
    assert "--hardware t4 needs its GPU; this machine has none" in capsys.readouterr().err
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "Tesla P100-PCIE-16GB")
    assert main(argv) == 2
    assert "but this machine's GPU is a Tesla P100-PCIE-16GB" in capsys.readouterr().err
    assert not (home / "bench").exists()


def test_an_unknown_model_is_a_usage_error(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "--hardware", "cpu", "--models", "ladder_s/m9", "--dry-run"]) == 2
    assert "no model config 'ladder_s/m9'" in capsys.readouterr().err
