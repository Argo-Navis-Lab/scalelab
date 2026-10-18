"""The M0 exit criterion: one command records a complete run row."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from scalelab.cli import main
from scalelab.core.envcapture import EnvInfo
from scalelab.core.registry import Registry

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"


@pytest.fixture
def home(monkeypatch, tmp_path):
    """Point SCALELAB_HOME at a fresh temporary folder for this test."""
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path))
    return tmp_path.resolve()


def last_loss(output: str) -> str:
    return output.strip().splitlines()[-1].split("last loss ")[1]


def test_one_command_records_a_complete_run_row(home, capsys):
    assert main(["run", str(HELLO), "seed=3"]) == 0

    registry = Registry.sqlite(home / "registry.db")
    (run,) = registry.list_runs()
    assert run.run_id.startswith("3f7ffa68d30146bd-s3-")
    assert run.config_hash == "3f7ffa68d30146bd"
    assert run.seed == 3
    assert run.status == "done"
    assert run.deterministic is True  # tests run on CPUs, where it is on by default
    assert run.executor == "local"
    assert run.hardware == "cpu"
    assert run.ended_at is not None
    assert run.started_at <= run.ended_at
    assert EnvInfo.model_validate_json(run.env_json).git.sha.startswith(run.run_id[-8:])
    assert registry.experiment_json(run.config_hash) is not None
    assert "done: ran 50 steps" in capsys.readouterr().out


def test_running_it_again_does_nothing(home, capsys):
    main(["run", str(HELLO)])
    assert main(["run", str(HELLO)]) == 0
    assert "already done" in capsys.readouterr().out
    assert len(Registry.sqlite(home / "registry.db").list_runs()) == 1


def test_runtime_settings_do_not_change_the_run(home, capsys):
    main(["run", str(HELLO)])
    main(["run", str(HELLO), "runtime.worker_name=colab-1"])  # same run_id: already done
    assert "already done" in capsys.readouterr().out


def test_overrides_make_a_different_experiment(home):
    main(["run", str(HELLO), "model=ladder_s/m2"])
    (run,) = Registry.sqlite(home / "registry.db").list_runs()
    assert run.config_hash != "3f7ffa68d30146bd"


@pytest.mark.parametrize(
    "bad",
    ["model.n_layer=6", "nonsense", "model=ladder_s/m9", "seed=-1", "runtime.colour=red"],
)
def test_bad_input_is_one_clear_error_and_records_nothing(home, capsys, bad):
    assert main(["run", str(HELLO), bad]) == 2
    assert capsys.readouterr().err.startswith("scalelab: error: ")
    assert not (home / "registry.db").exists()


def test_old_registry_file_is_reported_clearly(home, capsys):
    registry = Registry.sqlite(home / "registry.db")
    with registry.engine.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE runs DROP COLUMN deterministic")
    assert main(["run", str(HELLO)]) == 2
    assert "created by an older version of ScaleLab" in capsys.readouterr().err


def test_interrupted_run_resumes_to_the_same_result(home, tmp_path, capsys, monkeypatch):
    main(["run", str(HELLO), "seed=4"])  # the uninterrupted reference, in `home`
    expected = last_loss(capsys.readouterr().out)

    other_home = tmp_path / "interrupted"
    job = subprocess.Popen(
        [sys.executable, "-u", "-m", "scalelab.cli", "run", str(HELLO), "seed=4"]
        + ["--step-seconds", "0.05"],
        env={**os.environ, "SCALELAB_HOME": str(other_home)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert job.stdout is not None  # we asked for it with stdout=subprocess.PIPE
    for line in job.stdout:  # wait until the run is recorded and training
        if line.startswith("started"):
            break
    job.send_signal(signal.SIGTERM)
    output, _ = job.communicate(timeout=60)
    assert job.returncode == 130, output
    assert "stopped early, checkpoint saved" in output
    (run,) = Registry.sqlite(other_home / "registry.db").list_runs()
    assert run.status == "running"

    monkeypatch.setenv("SCALELAB_HOME", str(other_home))  # now resume it, in-process
    assert main(["run", str(HELLO), "seed=4"]) == 0
    output = capsys.readouterr().out
    assert "resuming (it was running)" in output
    assert last_loss(output) == expected
    (run,) = Registry.sqlite(other_home / "registry.db").list_runs()
    assert run.status == "done"
