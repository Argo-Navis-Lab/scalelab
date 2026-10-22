"""StepTimer tests: phases against a made-up clock, a new step starting afresh, and on a
pretend GPU, waiting for the device at every reading and reading its peak memory."""

import pytest
import torch

from scalelab.telemetry.timing import StepTimer


def test_each_phase_runs_from_the_previous_lap():
    readings = iter([0.0, 1.0, 3.0, 3.5, 6.0, 7.0])
    timer = StepTimer(torch.device("cpu"), clock=readings.__next__)  # the step begins at 0
    timer.lap("data")  # at 1
    timer.lap("generate")  # at 3
    timer.lap("score")  # at 3.5
    timer.lap("data")  # at 6: a phase that comes back adds up
    assert timer.phases == {"data": 3.5, "generate": 2.0, "score": 0.5}
    assert timer.seconds() == 7.0  # read at 7
    assert timer.peak_memory() is None  # PyTorch doesn't track a CPU's


def test_start_begins_a_new_step():
    readings = iter([0.0, 1.0, 5.0, 5.5, 6.0])
    timer = StepTimer(torch.device("cpu"), clock=readings.__next__)
    timer.lap("data")
    timer.start()  # at 5
    timer.lap("generate")  # at 5.5
    assert timer.phases == {"generate": 0.5}
    assert timer.seconds() == 1.0  # read at 6


def test_on_a_gpu_every_reading_waits_for_the_device(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def synchronize(device: torch.device) -> None:
        calls.append("synchronize")

    def reset_peak_memory_stats(device: torch.device) -> None:
        calls.append("reset")

    def max_memory_allocated(device: torch.device) -> int:
        return 3 * 2**30

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", max_memory_allocated)
    readings = iter([0.0, 2.0])
    timer = StepTimer(torch.device("cuda"), clock=readings.__next__)  # no GPU needed: faked
    timer.lap("generate")
    assert calls == ["reset", "synchronize", "synchronize"]  # the start, then the lap
    assert timer.phases == {"generate": 2.0}
    assert timer.peak_memory() == 3 * 2**30
