"""`scalelab sweep plan` (m4s4) and `sweep submit` (m4s5) end to end, on the datasheet's cost
model."""

from pathlib import Path

import pytest
import yaml

from scalelab import cli
from scalelab.cli import main
from scalelab.core.config import RuntimeConfig, load_config
from scalelab.core.envcapture import EnvInfo, capture_env
from scalelab.core.hashing import config_hash, run_id
from scalelab.core.registry import Registry
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
    assert [path.stem for path in paths] == ["local", "pilot", "warmstarts"]
    for path in paths:
        assert main(["sweep", "plan", str(path), "--hardware", "t4"]) == 0


@pytest.fixture
def clean(monkeypatch: pytest.MonkeyPatch) -> EnvInfo:
    """This checkout's environment, as if it had no uncommitted changes (m4s5)."""
    env = capture_env()
    git = env.git.model_copy(update={"dirty": False, "diff_sha256": None})
    env = env.model_copy(update={"git": git})
    monkeypatch.setattr(cli, "capture_env", lambda: env)
    return env


def test_submit_queues_every_job_in_plan_order(
    home: Path, tmp_path: Path, clean: EnvInfo, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_sweep(tmp_path, axes={"algo": ["algo=grpo", "algo=rloo"]}, seeds=[0, 1])
    assert main(["sweep", "submit", str(path), "--hardware", "cpu"]) == 0
    sweep_id = f"try-{clean.git.sha[:8]}"
    out = capsys.readouterr().out
    assert f"submitted:     sweep {sweep_id}, 4 jobs\njobs:          4 queued\n" in out
    jobs = Registry.sqlite(home / "registry.db").list_jobs(sweep_id)
    order = [("grpo", 0), ("rloo", 0), ("grpo", 1), ("rloo", 1)]
    assert [job.overrides for job in jobs] == [
        (*SMALL, f"algo={algo}", f"seed={seed}") for algo, seed in order
    ]
    for job, (algo, seed) in zip(jobs, order, strict=True):
        config = load_config(RL, [*SMALL, f"algo={algo}"])
        assert job.run_id == run_id(config, seed, clean.git.sha)  # what `scalelab run` records
        assert job.seconds == predict_job(config, RuntimeConfig(), CPU).seconds
        assert (job.state, job.hardware, job.experiment) == ("queued", "cpu", str(RL))
    assert main(["sweep", "submit", str(path), "--hardware", "cpu"]) == 0
    assert f"submitted:     already (sweep {sweep_id}): nothing added\n" in capsys.readouterr().out
    assert len(Registry.sqlite(home / "registry.db").list_jobs(sweep_id)) == 4


def test_submit_records_a_job_whose_run_is_done_as_done(
    home: Path, tmp_path: Path, clean: EnvInfo, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = Registry.sqlite(home / "registry.db")
    finished = registry.add_run(load_config(RL, [*SMALL, "algo=grpo"]), 1, clean)
    registry.finish_run(finished, "done")
    registry.engine.dispose()
    path = write_sweep(tmp_path, axes={"algo": ["algo=grpo"]}, seeds=[0, 1])
    assert main(["sweep", "submit", str(path), "--hardware", "cpu"]) == 0
    assert "jobs:          1 queued, 1 done\n" in capsys.readouterr().out
    jobs = Registry.sqlite(home / "registry.db").list_jobs(f"try-{clean.git.sha[:8]}")
    assert [(job.overrides[-1], job.state) for job in jobs] == [
        ("seed=0", "queued"),
        ("seed=1", "done"),
    ]
    assert jobs[1].run_id == finished


def test_submit_refuses_uncommitted_changes(
    home: Path,
    tmp_path: Path,
    clean: EnvInfo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dirty = clean.model_copy(update={"git": clean.git.model_copy(update={"dirty": True})})
    monkeypatch.setattr(cli, "capture_env", lambda: dirty)
    assert main(["sweep", "submit", str(write_sweep(tmp_path)), "--hardware", "cpu"]) == 2
    assert "error: commit your changes first" in capsys.readouterr().err
    assert not (home / "registry.db").exists()


def test_submit_records_nothing_if_a_job_is_too_big(
    home: Path, tmp_path: Path, clean: EnvInfo, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_sweep(tmp_path, axes={"width": ["model.d_model=128", "model.d_model=4096"]})
    assert main(["sweep", "submit", str(path), "--hardware", "t4"]) == 1
    assert capsys.readouterr().out.endswith("submitted:     nothing\n")
    assert not (home / "registry.db").exists()
