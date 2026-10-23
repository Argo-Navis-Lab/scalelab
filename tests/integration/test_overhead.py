"""m2s36: the same RL steps timed without and with telemetry, which changes nothing about them."""

from pathlib import Path
from typing import Any

import pytest
import torch

from scalelab.core.config import ExperimentConfig, load_config
from scalelab.models.transformer import Transformer
from scalelab.rl import overhead
from scalelab.rl.experiment import run_rl_experiment
from scalelab.rl.loop import RLStepMetrics
from scalelab.rl.overhead import measure_overhead

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"
# rl.yaml made small, as in test_rl_experiment: a tiny model, 4 problems x 2 answers a step.
SMALL = [
    "model.name=tiny",
    "model.n_layers=2",
    "model.d_model=64",
    "model.n_heads=1",
    "model.context=32",
    "task.difficulties=[2]",
    "rl.prompts_per_step=4",
    "rl.group_size=2",
    "rl.flops=2e9",
]


def config() -> ExperimentConfig:
    return load_config(EXPERIMENTS / "rl.yaml", SMALL)


def warm_start(config: ExperimentConfig) -> Transformer:
    """The same weights every time, made without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(config.model)


def test_telemetry_changes_nothing_about_the_steps():
    small = config()
    # 11 steps, so step 10's detailed collectors (layers, Adam, activations) run too
    result = measure_overhead(small, 0, warm_start(small), True, steps=10, warmup=1)
    assert len(result.off.metrics) == 11  # the step count alone ends a run, not rl.flops
    assert result.on.metrics == result.off.metrics
    assert result.off.seconds > 0
    assert result.on.seconds > 0


def test_each_run_trains_its_own_copy_of_the_warm_start(monkeypatch: pytest.MonkeyPatch):
    # A tiny untrained model never answers right, so its gradients are 0 and the metrics
    # above can't tell whether the second run started where the first one ended: look.
    given: list[Transformer] = []

    def recording(
        config: ExperimentConfig, seed: int, start: Transformer, *args: Any, **kwargs: Any
    ) -> list[RLStepMetrics]:
        given.append(start)
        return run_rl_experiment(config, seed, start, *args, **kwargs)

    monkeypatch.setattr(overhead, "run_rl_experiment", recording)
    small = config()
    start = warm_start(small)
    measure_overhead(small, 0, start, True, steps=1, warmup=1)
    assert len(given) == 2
    assert given[0] is not start
    assert given[1] is not start
    assert given[0] is not given[1]


def test_only_the_steps_after_the_warmup_are_timed():
    # The clock is read once as each step ends: steps 1 to 3 without telemetry, then with it.
    clock = iter([10.0, 11.0, 13.0, 100.0, 103.0, 106.0]).__next__
    small = config()
    result = measure_overhead(small, 0, warm_start(small), True, steps=2, warmup=1, clock=clock)
    assert result.off.seconds == 1.5  # (13 - 10) / 2
    assert result.on.seconds == 3.0  # (106 - 100) / 2
    assert result.overhead == 1.0  # telemetry doubled the step time


@pytest.mark.parametrize(("steps", "warmup"), [(0, 5), (30, 0)])
def test_no_steps_to_time_is_refused(steps: int, warmup: int):
    small = config()
    with pytest.raises(ValueError, match="steps and warmup must be at least 1"):
        measure_overhead(small, 0, warm_start(small), True, steps, warmup)


def test_a_warm_start_experiment_is_refused():
    hello = load_config(EXPERIMENTS / "hello.yaml")
    with pytest.raises(ValueError, match="is not an RL experiment"):
        measure_overhead(hello, 0, warm_start(hello), True)
