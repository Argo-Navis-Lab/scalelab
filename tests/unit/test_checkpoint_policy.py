"""Tests for when checkpoints are saved, and for keeping only the newest ones."""

import os
import signal

import pytest
import torch

from scalelab.core.checkpoint import (
    CheckpointPolicy,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)


class FakeClock:
    """A clock the test moves by hand, so a 15-minute rule can be tested instantly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# --- when a save is due ------------------------------------------------------------------


def test_every_n_steps():
    policy = CheckpointPolicy(every_steps=5, every_seconds=None)
    assert [step for step in range(1, 16) if policy.due(step)] == [5, 10, 15]


def test_every_15_minutes_of_wall_clock_time():
    clock = FakeClock()
    policy = CheckpointPolicy(clock=clock)  # the default: every 15 minutes
    clock.now = 14 * 60 + 59
    assert not policy.due(step=1)
    clock.now = 15 * 60
    assert policy.due(step=2)
    policy.saved()  # the clock restarts at the save
    clock.now = 25 * 60
    assert not policy.due(step=3)
    clock.now = 30 * 60
    assert policy.due(step=4)


def test_whichever_comes_first():
    clock = FakeClock()
    policy = CheckpointPolicy(every_steps=1000, every_seconds=60, clock=clock)
    clock.now = 61
    assert policy.due(step=3)  # time won, long before step 1000


def test_no_triggers_means_no_saves():
    policy = CheckpointPolicy(every_steps=None, every_seconds=None)
    assert not any(policy.due(step) for step in range(1, 1000))


def test_a_stop_request_makes_a_save_due_at_once():
    policy = CheckpointPolicy(every_steps=None, every_seconds=None)
    policy.request_stop()
    assert policy.stop_requested
    assert policy.due(step=3)


def test_keep_must_be_at_least_one():
    with pytest.raises(ValueError, match="keep must be at least 1"):
        CheckpointPolicy(keep=0)


@pytest.fixture
def restore_signal_handlers():
    """stop_on_signals changes how this whole Python process reacts to signals. Undo that."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "Ctrl+C"])
def test_signals_ask_for_a_save_and_stop(restore_signal_handlers, sig):
    policy = CheckpointPolicy()
    policy.stop_on_signals()
    os.kill(os.getpid(), sig)  # send the signal to ourselves; without the handler, we'd die
    assert policy.stop_requested


# --- keeping only the newest checkpoints -----------------------------------------------


def save_steps(directory, home, steps):
    for step in steps:
        save_checkpoint(directory, step, {"w": torch.full((3,), float(step))}, home)


def stored_files(home):
    return [p for p in (home / "artifacts").rglob("*") if p.is_file()]


def test_prune_keeps_the_newest(tmp_path):
    save_steps(tmp_path / "run", tmp_path, [5, 10, 15, 20])
    prune_checkpoints(tmp_path / "run", keep=2, home=tmp_path)
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == [
        "step-000000015.json",
        "step-000000020.json",
    ]
    assert len(stored_files(tmp_path)) == 2  # the old tensor files are gone too


def test_prune_with_nothing_to_delete(tmp_path):
    save_steps(tmp_path / "run", tmp_path, [5])
    prune_checkpoints(tmp_path / "run", keep=2, home=tmp_path)
    assert len(list((tmp_path / "run").iterdir())) == 1


def test_tensor_files_shared_with_another_run_survive(tmp_path):
    same = {"w": torch.ones(3)}  # two runs that happen to hold identical tensors
    save_checkpoint(tmp_path / "run_a", 1, same, tmp_path)
    other_run = save_checkpoint(tmp_path / "run_b", 1, same, tmp_path)
    save_steps(tmp_path / "run_a", tmp_path, [2])

    prune_checkpoints(tmp_path / "run_a", keep=1, home=tmp_path)  # drops run_a's step 1

    _, state = load_checkpoint(other_run, tmp_path)  # run_b still loads
    assert torch.equal(state["w"], torch.ones(3))


def test_prune_refuses_folders_outside_home(tmp_path):
    with pytest.raises(ValueError, match="must be inside"):
        prune_checkpoints(tmp_path / "elsewhere", keep=2, home=tmp_path / "home")


def test_the_kept_checkpoint_still_loads(tmp_path):
    save_steps(tmp_path / "run", tmp_path, [1, 2, 3])
    prune_checkpoints(tmp_path / "run", keep=1, home=tmp_path)
    (manifest,) = (tmp_path / "run").glob("step-*.json")
    step, state = load_checkpoint(manifest, tmp_path)  # fails if its tensors were deleted
    assert step == 3
    assert torch.equal(state["w"], torch.full((3,), 3.0))
