"""What telemetry costs: the same RL steps, timed without it and with it."""

import copy
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from scalelab.core.checkpoint import CheckpointPolicy
from scalelab.core.config import ExperimentConfig
from scalelab.models.transformer import Transformer
from scalelab.rl.experiment import run_rl_experiment
from scalelab.rl.loop import RLStepMetrics
from scalelab.telemetry.anomalies import AnomalyEvent
from scalelab.telemetry.sink import MetricsSink

OVERHEAD_BUDGET = 0.05  # telemetry may add at most 5% to a step's time


@dataclass(frozen=True)
class Timing:
    """One of the two runs: the mean seconds of a timed step, and what the run did."""

    seconds: float
    metrics: list[RLStepMetrics]
    events: list[AnomalyEvent]  # the detectors' (none without telemetry)


@dataclass(frozen=True)
class Overhead:
    """The same steps without telemetry (off) and with it (on)."""

    off: Timing
    on: Timing

    @property
    def overhead(self) -> float:
        """What telemetry adds to a step, as a share of the step without it."""
        return self.on.seconds / self.off.seconds - 1


def measure_overhead(
    config: ExperimentConfig,
    seed: int,
    start: Transformer,
    deterministic: bool | None = None,
    steps: int = 30,
    warmup: int = 5,
    clock: Callable[[], float] = time.perf_counter,
) -> Overhead:
    """Run `warmup + steps` RL steps of `config` from `start` twice, first without telemetry,
    then with it (collectors, sink and detectors), and time the last `steps` of each: from
    the end of step `warmup` to the end of the last step. Each run trains its own copy of
    `start` with the same seed, so both do the same work (exactly the same on CPU), and only
    the step count ends them. Nothing is graded, and everything is written to a temporary
    folder, so nothing is recorded.
    """
    if config.rl is None:
        raise ValueError(f"{config.name} is not an RL experiment: it has no rl section")
    if steps < 1 or warmup < 1:
        raise ValueError(f"steps and warmup must be at least 1, got {steps} and {warmup}")
    last = warmup + steps
    rl = config.rl.model_copy(update={"max_steps": last, "flops": math.inf})
    config = config.model_copy(update={"rl": rl})

    def timed(home: Path, telemetry: bool) -> Timing:
        ended: dict[int, float] = {}  # step -> the clock as it ended
        events: list[AnomalyEvent] = []

        def on_step(metrics: RLStepMetrics) -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # the step's GPU work done, not just queued
            ended[metrics.step] = clock()

        metrics = run_rl_experiment(
            config,
            seed,
            copy.deepcopy(start),
            home / "checkpoints",
            home,
            CheckpointPolicy(every_seconds=None),  # saved once, after the last step
            deterministic,
            sink=MetricsSink(home, "overhead") if telemetry else None,
            on_anomaly=events.append,
            on_step=on_step,
        )
        return Timing((ended[last] - ended[warmup]) / steps, metrics, events)

    with tempfile.TemporaryDirectory() as folder:
        off = timed(Path(folder) / "off", telemetry=False)
        on = timed(Path(folder) / "on", telemetry=True)
    return Overhead(off, on)
