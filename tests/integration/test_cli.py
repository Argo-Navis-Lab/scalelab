"""The m0 exit criterion: one command records a complete run row."""

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
    assert run.run_id.startswith("c6b6e17b3e7b116b-s3-")
    assert run.config_hash == "c6b6e17b3e7b116b"
    assert run.seed == 3
    assert run.status == "done"
    assert run.deterministic is True  # tests run on CPUs, where it is on by default
    assert run.executor == "local"
    assert run.hardware == "cpu"
    assert run.ended_at is not None
    assert run.started_at <= run.ended_at
    assert EnvInfo.model_validate_json(run.env_json).git.sha.startswith(run.run_id[-8:])
    assert registry.experiment_json(run.config_hash) is not None
    assert "done: ran 16 steps" in capsys.readouterr().out
    graded = [(e.step, e.task, e.metric, e.n) for e in registry.list_evals(run.run_id)]
    assert graded == [(16, f"arithmetic-{d}", "greedy_pass@1", 20) for d in (2, 3, 4, 5)]


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
    assert run.config_hash != "c6b6e17b3e7b116b"


@pytest.mark.parametrize(
    "bad",
    [
        "model.n_layer=6",
        "nonsense",
        "model=ladder_s/m9",
        "seed=-1",
        "runtime.colour=red",
        "task.difficulties=[13]",
    ],
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
    longer = ["seed=4", "warm_start.tokens=50000"]  # a few seconds: time for the signal to land
    main(["run", str(HELLO), *longer])  # the uninterrupted reference, in `home`
    expected = last_loss(capsys.readouterr().out)

    other_home = tmp_path / "interrupted"
    job = subprocess.Popen(
        [sys.executable, "-u", "-m", "scalelab.cli", "run", str(HELLO), *longer],
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
    assert main(["run", str(HELLO), *longer]) == 0
    output = capsys.readouterr().out
    assert "resuming (it was running)" in output
    assert last_loss(output) == expected
    (run,) = Registry.sqlite(other_home / "registry.db").list_runs()
    assert run.status == "done"


# --- added in m1s22: a registry on the machine's own disk, copied home ----------------------


def on_disk_of(machine: Path) -> str:
    """The override that keeps the live registry on `machine`'s own disk."""
    return f"runtime.local_registry={machine / 'registry.db'}"


def test_a_local_registry_is_copied_home(home: Path, tmp_path: Path):
    vm = tmp_path / "vm"  # the VM's own disk, while `home` plays Google Drive
    assert main(["run", str(HELLO), on_disk_of(vm)]) == 0
    assert (home / "registry.db").read_bytes() == (vm / "registry.db").read_bytes()
    copy = Registry.sqlite(home / "registry.db")
    (run,) = copy.list_runs()
    assert run.status == "done"
    assert len(copy.list_evals(run.run_id)) == 4  # the final grade on each eval set


def test_a_new_machine_carries_on_from_the_copy_at_home(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    main(["run", str(HELLO), on_disk_of(tmp_path / "vm1")])
    assert main(["run", str(HELLO), on_disk_of(tmp_path / "vm2")]) == 0
    assert "already done" in capsys.readouterr().out  # vm2 began from vm1's copy at home
