"""Where a training step's time goes, phase by phase, and the most device memory it needed."""

import time
from collections.abc import Callable

import torch


class StepTimer:
    """Times the phases of a step: start() as the step begins, then lap(name) as each phase
    ends. A phase's time runs from the previous lap (or the start) to its own lap, and a
    name that laps twice adds up.

    On a GPU, Python runs ahead of the work it queues: each reading of the clock first waits
    for the device to finish, so a phase's time is its own work and not the next one's.
    """

    def __init__(self, device: torch.device, clock: Callable[[], float] = time.perf_counter):
        self.device = device
        self.clock = clock
        self.start()

    def start(self) -> None:
        """A step begins: forget the last step's phases and, on a GPU, its peak memory."""
        self.phases: dict[str, float] = {}  # seconds by phase
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.began = self.last = self.now()

    def lap(self, name: str) -> None:
        """Phase `name` has just ended."""
        now = self.now()
        self.phases[name] = self.phases.get(name, 0.0) + now - self.last
        self.last = now

    def seconds(self) -> float:
        """The time since the step began."""
        return self.now() - self.began

    def peak_memory(self) -> int | None:
        """The most bytes of device memory the step's tensors held at once, on a GPU; None on
        a CPU, where PyTorch doesn't track it."""
        if self.device.type != "cuda":
            return None
        return torch.cuda.max_memory_allocated(self.device)

    def now(self) -> float:
        """The clock, once the device has finished the work queued so far."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return self.clock()
