"""A change, measured: the same steps run twice, without it and with it, timed by telemetry."""

import copy
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from scalelab.core.checkpoint import CheckpointPolicy
from scalelab.core.config import ExperimentConfig, Precision
from scalelab.costmodel.report import Change
from scalelab.models.transformer import Transformer
from scalelab.rl.costing import measured_step
from scalelab.rl.experiment import run_experiment, run_rl_experiment
from scalelab.telemetry.sink import MetricsSink, read_metrics

RUN_ID = "measure"  # the metrics' folder inside each run's temporary home


@dataclass(frozen=True)
class Timed:
    """The same steps without the change (off: fp32, uncut) and with it (on): each part's mean
    seconds a step (generate, logprobs, train, step; a warm start: train, step), as cost
    validate reads them from real runs."""

    off: dict[str, float]
    on: dict[str, float]


def measure_change(
    config: ExperimentConfig,
    seed: int,
    change: Change,
    start: Transformer | None = None,
    deterministic: bool | None = None,
    steps: int = 20,
    warmup: int = 3,
) -> Timed:
    """Run the first `warmup + steps` steps of `config` twice, in fp32 without the padding cut
    and then with `change` (runtime.padding_cut, m3s17; runtime.precision fp16, m3s24), and
    average each part's laps over the last `steps` of each. An RL experiment trains its own copy
    of `start` (its warm start) each time; a warm start begins from its seed (`start` None).
    Both runs draw the same data; the cut does the same math, to rounding (m3s16), while fp16
    rounds more, so its answers (and their lengths) can drift from fp32's. Telemetry is on, as
    real runs have it. Nothing is graded and everything is written to a temporary folder, so
    nothing is recorded.
    """
    if (start is None) != (config.rl is None):
        raise ValueError("an RL experiment starts from its warm start, and a warm start from none")
    if steps < 1 or warmup < 1:
        raise ValueError(f"steps and warmup must be at least 1, got {steps} and {warmup}")
    last = warmup + steps
    if config.rl is not None:
        rl = config.rl.model_copy(update={"max_steps": last, "flops": math.inf})
        config = config.model_copy(update={"rl": rl})

    def timed(home: Path, applied: Change | None) -> dict[str, float]:
        sink = MetricsSink(home, RUN_ID)
        policy = CheckpointPolicy(every_seconds=None)  # saved once, after the last step
        checkpoints = home / "checkpoints"
        padding_cut = applied == "padding-cut"
        precision: Precision = "fp16" if applied == "fp16" else "fp32"
        if start is None:
            run_experiment(
                config,
                seed,
                checkpoints,
                home,
                policy,
                deterministic,
                sink=sink,
                padding_cut=padding_cut,
                stop_after=last,
                precision=precision,
            )
        else:
            run_rl_experiment(
                config,
                seed,
                copy.deepcopy(start),
                checkpoints,
                home,
                policy,
                deterministic,
                sink=sink,
                padding_cut=padding_cut,
                precision=precision,
            )
        count, seconds = measured_step(read_metrics(home, RUN_ID), skip=warmup)
        if count < steps:  # the sink's rows reach disk with the save after the last step
            raise ValueError(
                f"{config.name} ran {warmup + count} steps, not {last}: its warm_start.tokens "
                "ran out first"
            )
        return seconds

    with tempfile.TemporaryDirectory() as folder:
        off = timed(Path(folder) / "off", None)
        on = timed(Path(folder) / "on", change)
    return Timed(off, on)
