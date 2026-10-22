"""Kill a real run the way a Colab VM dies (SIGKILL: no warning, no last save), then run the
same command again: a warm start and an RL run both finish exactly as if never killed."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from scalelab.cli import main
from scalelab.core.checkpoint import latest_checkpoint, load_checkpoint
from scalelab.core.registry import Registry
from scalelab.telemetry.sink import read_metrics

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"
HELLO, RL = EXPERIMENTS / "hello.yaml", EXPERIMENTS / "rl.yaml"
# hello made small: 2-digit problems and a 64-token context, 32 quick steps. Also RL's warm
# start, for RL with 4 problems x 2 answers a step, about 13 steps, graded only at the start
# and the end.
SMALL = ["model.context=64", "task.difficulties=[2]"]
SMALL_RL = [
    *SMALL,
    "rl.prompts_per_step=4",
    "rl.group_size=2",
    "rl.flops=6e10",
    "eval.every=0",
    "eval.problems=5",
]


def run_here(monkeypatch: pytest.MonkeyPatch, home: Path, experiment: Path, *overrides: str):
    """`scalelab run`, in this process, with `home` as SCALELAB_HOME."""
    monkeypatch.setenv("SCALELAB_HOME", str(home))
    assert main(["run", str(experiment), *overrides]) == 0


def start(home: Path, experiment: Path, *overrides: str) -> tuple[subprocess.Popen[str], str]:
    """`scalelab run` in a process of its own. Returns it and its run_id once it has started
    training."""
    command = [sys.executable, "-u", "-m", "scalelab.cli", "run", str(experiment), *overrides]
    environment = {**os.environ, "SCALELAB_HOME": str(home)}
    job = subprocess.Popen(
        command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert job.stdout is not None  # asked for with stdout=PIPE
    run_id = ""
    for line in job.stdout:
        if line.startswith("run:"):
            run_id = line.split()[1]
        if line.startswith("started"):
            break
    return job, run_id


def kill_once_saved(job: subprocess.Popen[str], folder: Path, step: int) -> int:
    """SIGKILL the job as soon as `folder` holds a checkpoint at `step` or later. Returns the
    step of the last checkpoint it saved."""
    deadline = time.monotonic() + 60
    while checkpoint_step(folder) < step:
        assert job.poll() is None, "the job ended before it could be killed"
        assert time.monotonic() < deadline, f"no checkpoint at step {step} after 60 seconds"
        time.sleep(0.01)
    job.kill()  # SIGKILL: no warning and no last save, like a Colab VM vanishing
    job.wait()
    assert job.stdout is not None
    job.stdout.close()
    return checkpoint_step(folder)


def checkpoint_step(folder: Path) -> int:
    latest = latest_checkpoint(folder)
    return 0 if latest is None else int(latest.stem.removeprefix("step-"))


def final_weights(home: Path, run_id: str) -> dict[str, torch.Tensor]:
    latest = latest_checkpoint(home / "checkpoints" / run_id)
    assert latest is not None
    return load_checkpoint(latest, home)[1]["model"]


def same_weights(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


def test_a_killed_warm_start_resumes_to_the_same_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    straight, killed = tmp_path / "straight", tmp_path / "killed"
    run_here(monkeypatch, straight, HELLO, *SMALL)

    job, run_id = start(killed, HELLO, *SMALL, "runtime.checkpoint_every=5")
    saved = kill_once_saved(job, killed / "checkpoints" / run_id, step=10)
    assert 10 <= saved < checkpoint_step(straight / "checkpoints" / run_id)  # killed halfway
    (run,) = Registry.sqlite(killed / "registry.db").list_runs()
    assert run.status == "running"  # it had no chance to say otherwise

    run_here(monkeypatch, killed, HELLO, *SMALL)  # the same command again: it resumes
    assert same_weights(final_weights(killed, run_id), final_weights(straight, run_id))
    (run,) = Registry.sqlite(killed / "registry.db").list_runs()
    assert run.status == "done"


def test_a_killed_rl_run_resumes_to_the_same_weights_and_grades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    straight, killed = tmp_path / "straight", tmp_path / "killed"
    for home in (straight, killed):
        run_here(monkeypatch, home, HELLO, *SMALL)  # the same warm start in both homes
    (warm_start,) = Registry.sqlite(straight / "registry.db").list_runs()
    rl = [*SMALL_RL, f"rl.from_run={warm_start.run_id}"]
    run_here(monkeypatch, straight, RL, *rl)

    job, run_id = start(killed, RL, *rl, "runtime.checkpoint_every=2")
    saved = kill_once_saved(job, killed / "checkpoints" / run_id, step=4)
    assert 4 <= saved < checkpoint_step(straight / "checkpoints" / run_id)

    run_here(monkeypatch, killed, RL, *rl)
    assert same_weights(final_weights(killed, run_id), final_weights(straight, run_id))
    grades = [
        [
            (e.step, e.task, e.metric, e.value)
            for e in Registry.sqlite(h / "registry.db").list_evals(run_id)
        ]
        for h in (straight, killed)
    ]
    assert grades[0] == grades[1]  # step 0 from before the kill, the end from after it
    measured = [
        [
            (row.step, row.metric, None if row.metric.endswith("_s") else row.value)
            for row in read_metrics(h, run_id)
        ]
        for h in (straight, killed)
    ]
    # And the same metrics (m2s28), values included but for seconds and per-second rates
    # (m2s29): the resumed run read its own clock.
    assert measured[0] == measured[1]


def test_periodic_saving_keeps_only_the_last_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_here(monkeypatch, tmp_path, HELLO, "runtime.checkpoint_every=5")  # 16 steps
    (run,) = Registry.sqlite(tmp_path / "registry.db").list_runs()
    saved = sorted(path.name for path in (tmp_path / "checkpoints" / run.run_id).iterdir())
    assert saved == ["step-000000015.json", "step-000000016.json"]  # 5 and 10 were pruned
    tensors = [path for path in (tmp_path / "artifacts").rglob("*") if path.is_file()]
    assert len(tensors) == 2  # and their tensors with them


def test_a_checkpoint_interval_of_0_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path))
    assert main(["run", str(HELLO), "runtime.checkpoint_every=0"]) == 2
