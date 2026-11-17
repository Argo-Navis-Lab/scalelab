"""scalelab export (m4s17): runs copied as a dataset, with nothing that names a machine."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scalelab.analysis.export import (
    MANIFEST,
    ExportError,
    export_runs,
    file_leaks,
    leaks,
    sha256,
)
from scalelab.cli import main
from scalelab.core.config import ExperimentConfig, RuntimeConfig
from scalelab.core.envcapture import EnvInfo, GitInfo, HardwareInfo, SoftwareInfo
from scalelab.core.registry import EvalRecord, Job, Registry
from scalelab.core.storage import registry_path

ENV = EnvInfo(
    git=GitInfo(sha="0123456789abcdef0123456789abcdef01234567", dirty=False),
    software=SoftwareInfo(python="3.12.14", os="test-os", packages={"scalelab": "0.1.0"}),
    hardware=HardwareInfo(cpu_model="Test CPU", cpu_count=4),
)
# What a Colab worker records about itself: all of it must stay behind.
COLAB = RuntimeConfig(local_registry="/content/registry.db", worker_name="vm-7", precision="fp16")


def finished_run(registry: Registry, config: ExperimentConfig, seed: int) -> str:
    """A done run with one eval row, started with COLAB's runtime settings."""
    run_id = registry.add_run(config, seed, ENV, runtime=COLAB)
    registry.record_eval(
        EvalRecord(
            run_id=run_id,
            step=0,
            task="arithmetic-2",
            metric="greedy_pass@1",
            value=0.5,
            n=1000,
            stderr=0.0158,
        )
    )
    registry.finish_run(run_id, "done")
    return run_id


def leased_job(run_id: str) -> Job:
    return Job(
        job_id="main-0123abcd-1",
        sweep_id="main-0123abcd",
        number=1,
        run_id=run_id,
        experiment="configs/experiments/rl.yaml",
        overrides=("seed=0",),
        hardware="t4",
        seconds=60.0,
        state="done",
        lease_owner="bf0ac6044717-54088",
        attempts=1,
    )


@pytest.fixture
def home(tmp_path: Path, config: ExperimentConfig) -> Path:
    """A home with two done runs: the first with a grading, metrics and a sweep job."""
    home = tmp_path / "home"
    registry = Registry.sqlite(registry_path(home))
    first = finished_run(registry, config, seed=0)
    finished_run(registry, config, seed=1)
    registry.enqueue("main-0123abcd", "{}", ENV.git.sha, [leased_job(first)])
    registry.engine.dispose()
    grading = home / "gradings" / first / "grading-000000000.json"
    grading.parent.mkdir(parents=True)
    grading.write_text(json.dumps({"run_id": first, "step": 0}))
    part = home / "metrics" / first / "part-000000.parquet"
    part.parent.mkdir(parents=True)
    rows = {"run_id": [first], "step": [0], "metric": ["optim/loss"], "value": [0.1]}
    pq.write_table(pa.table(rows), part)
    return home


def first_run(home: Path) -> str:
    return next((home / "gradings").iterdir()).name


def test_leaks_finds_absolute_paths_and_machine_names() -> None:
    text = 'reg "/content/registry.db", /root/x, C:\\Users\\me, /workspace/home on box-17'
    assert leaks(text, ["box-17"]) == ["/content/", "/root/", "C:\\", "/workspace/", "box-17"]


def test_leaks_passes_relative_paths_and_metric_names() -> None:
    text = "configs/experiments/rl.yaml runs optim/grad_norm system/train_s gradings/x/y.json"
    assert leaks(text, ["box-17"]) == []


def test_file_leaks_reads_parquet_text_columns(tmp_path: Path) -> None:
    part = tmp_path / "part.parquet"
    pq.write_table(pa.table({"metric": ["optim/loss", "/home/me/x"], "value": [1.0, 2.0]}), part)
    assert file_leaks(part, []) == ["/home/"]


def test_export_copies_the_runs_without_machine_details(home: Path, tmp_path: Path) -> None:
    out = tmp_path / "dataset"
    run_id = first_run(home)
    registry = Registry.sqlite(registry_path(home))
    exported = export_runs(home, registry, [run_id], out)
    copy = Registry.sqlite(registry_path(out))
    assert [run.run_id for run in copy.list_runs()] == [run_id]  # the other seed stays home
    assert len(copy.list_evals(run_id)) == 1
    runtime = copy.get_runtime(run_id)
    assert runtime == RuntimeConfig(precision="fp16")  # the machine's fields back to defaults
    assert [job.lease_owner for job in copy.list_jobs("main-0123abcd")] == [None]
    assert exported.runs == 1
    assert exported.files == 3  # registry.db, the grading, the metrics part
    manifest = json.loads((out / MANIFEST).read_text())
    assert manifest["runs"] == [run_id]
    grading = f"gradings/{run_id}/grading-000000000.json"
    assert manifest["files"][grading] == sha256(out / grading)
    assert (out / f"metrics/{run_id}/part-000000.parquet").is_file()


def test_a_leak_stops_the_export_and_leaves_nothing(home: Path, tmp_path: Path) -> None:
    run_id = first_run(home)
    grading = home / "gradings" / run_id / "grading-000000000.json"
    grading.write_text(json.dumps({"saved_from": "/content/drive/x"}))
    out = tmp_path / "dataset"
    with pytest.raises(ExportError, match="grading-000000000.json: /content/"):
        export_runs(home, Registry.sqlite(registry_path(home)), [run_id], out)
    assert not out.exists()


def test_export_refuses_a_folder_with_files(home: Path, tmp_path: Path) -> None:
    out = tmp_path / "dataset"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    with pytest.raises(ExportError, match="not empty"):
        export_runs(home, Registry.sqlite(registry_path(home)), [first_run(home)], out)
    assert (out / "notes.txt").read_text() == "mine"


def test_the_command_checks_its_runs_then_exports(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SCALELAB_HOME", str(home))
    run_id = first_run(home)
    out = tmp_path / "dataset"
    assert main(["export", "nope-s0-01234567", "--out", str(out)]) == 2
    assert main(["export", run_id, run_id, "--out", str(out)]) == 2
    assert main(["export", run_id, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "exported 1 runs (1 given, 0 warm starts)" in printed
    assert (out / MANIFEST).is_file()
