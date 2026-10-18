"""Interrupt a real running job, the ways Colab and you can, and check it resumes exactly."""

import signal
import subprocess
import sys
import time
from pathlib import Path

from scalelab.core.checkpoint import CheckpointPolicy, latest_checkpoint
from scalelab.core.dummy_job import run_dummy_job

STEPS = 50


def start_job(home: Path) -> subprocess.Popen:
    """Start the dummy job in a separate process: a checkpoint every 5 steps, 0.05 s a step."""
    command = [sys.executable, "-m", "scalelab.core.dummy_job", str(home)]
    command += ["--every-steps", "5", "--step-seconds", "0.05"]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def saved_step(home: Path) -> int:
    latest = latest_checkpoint(home / "checkpoints")
    return 0 if latest is None else int(latest.stem.removeprefix("step-"))


def wait_until_saved(job: subprocess.Popen, home: Path, step: int) -> None:
    deadline = time.monotonic() + 60
    while saved_step(home) < step:
        if job.poll() is not None:
            raise RuntimeError(f"the job ended too early:\n{job.stdout.read()}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"no checkpoint at step {step} after 60 seconds")
        time.sleep(0.02)


def test_hard_kill_then_resume_is_bit_identical(tmp_path):
    uninterrupted = run_dummy_job(0, STEPS, tmp_path / "straight", home=tmp_path)

    home = tmp_path / "killed"
    job = start_job(home)
    wait_until_saved(job, home, step=10)
    job.kill()  # SIGKILL: no warning and no last save, like a Colab VM vanishing
    job.wait()

    saved = saved_step(home)
    assert 10 <= saved < STEPS
    policy = CheckpointPolicy(every_steps=5)
    resumed = run_dummy_job(0, STEPS, home / "checkpoints", home=home, policy=policy)
    assert resumed == uninterrupted[saved:]  # steps after the checkpoint, bit for bit


def test_sigterm_saves_and_stops_cleanly(tmp_path):
    uninterrupted = run_dummy_job(0, STEPS, tmp_path / "straight", home=tmp_path)

    home = tmp_path / "terminated"
    job = start_job(home)
    wait_until_saved(job, home, step=10)
    job.send_signal(signal.SIGTERM)  # "please stop": what a polite shutdown sends
    output, _ = job.communicate(timeout=60)

    assert job.returncode == 0  # it finished its step, saved and exited normally
    assert "stopped early, checkpoint saved" in output
    saved = saved_step(home)
    assert 10 <= saved < STEPS
    resumed = run_dummy_job(0, STEPS, home / "checkpoints", home=home)
    assert resumed == uninterrupted[saved:]


def test_periodic_saving_keeps_only_two(tmp_path):
    policy = CheckpointPolicy(every_steps=5)
    run_dummy_job(0, STEPS, tmp_path / "run", home=tmp_path, policy=policy)
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == [
        "step-000000045.json",
        "step-000000050.json",
    ]
    assert len([p for p in (tmp_path / "artifacts").rglob("*") if p.is_file()]) == 2
