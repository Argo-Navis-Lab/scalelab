"""m3s17: a change, measured: the same steps timed without it and with it (fp16 too: m3s24)."""

from pathlib import Path
from typing import Any

import pytest
import torch

from scalelab.core.config import ExperimentConfig, Precision, load_config
from scalelab.costmodel.report import Change
from scalelab.models.transformer import Transformer
from scalelab.rl import measure
from scalelab.rl.loop import RLStepMetrics
from scalelab.rl.measure import measure_change
from scalelab.rl.warmstart_loop import StepMetrics

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"
# hello made small, as in test_cli: 2-digit problems, a 64-token context, 16 steps of 16 rows
HELLO = ["model.context=64", "task.difficulties=[2]"]
# rl.yaml made small, as in test_overhead: a tiny model, 4 problems x 2 answers a step
TINY_RL = [
    "model.name=tiny",
    "model.n_layers=2",
    "model.d_model=64",
    "model.n_heads=1",
    "model.context=32",
    "task.difficulties=[2]",
    "rl.prompts_per_step=4",
    "rl.group_size=2",
]
# m3s24: the runtime each run gets (padding_cut, precision): fp32 uncut, then the change alone
RUNTIMES: list[tuple[Change, list[tuple[bool, Precision]]]] = [
    ("padding-cut", [(False, "fp32"), (True, "fp32")]),
    ("fp16", [(False, "fp32"), (False, "fp16")]),
]


def tiny_rl() -> ExperimentConfig:
    return load_config(EXPERIMENTS / "rl.yaml", TINY_RL)


def warm_start(config: ExperimentConfig) -> Transformer:
    """The same weights every time, made without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(config.model)


@pytest.mark.parametrize(("change", "expected"), RUNTIMES)
def test_a_warm_start_is_timed_without_the_change_then_with_it(
    monkeypatch: pytest.MonkeyPatch, change: Change, expected: list[tuple[bool, Precision]]
):
    runtimes: list[tuple[bool, Precision]] = []
    run_experiment = measure.run_experiment

    def recording(
        *args: Any, padding_cut: bool, precision: Precision, **kwargs: Any
    ) -> list[StepMetrics]:
        runtimes.append((padding_cut, precision))
        return run_experiment(*args, padding_cut=padding_cut, precision=precision, **kwargs)

    monkeypatch.setattr(measure, "run_experiment", recording)
    config = load_config(EXPERIMENTS / "hello.yaml", HELLO)
    timed = measure_change(config, 0, change, None, True, steps=3, warmup=1)
    assert runtimes == expected
    assert set(timed.off) == set(timed.on) == {"train", "step"}  # a warm start's laps
    assert all(seconds > 0 for seconds in [*timed.off.values(), *timed.on.values()])


@pytest.mark.parametrize(("change", "expected"), RUNTIMES)
def test_rl_trains_its_own_copy_of_the_warm_start_each_time(
    monkeypatch: pytest.MonkeyPatch, change: Change, expected: list[tuple[bool, Precision]]
):
    given: list[tuple[Transformer, bool, Precision]] = []
    run_rl_experiment = measure.run_rl_experiment

    def recording(
        config: ExperimentConfig,
        seed: int,
        start: Transformer,
        *args: Any,
        padding_cut: bool,
        precision: Precision,
        **kwargs: Any,
    ) -> list[RLStepMetrics]:
        given.append((start, padding_cut, precision))
        metrics = run_rl_experiment(
            config, seed, start, *args, padding_cut=padding_cut, precision=precision, **kwargs
        )
        assert len(metrics) == 3  # the step count alone ends a run, not rl.flops
        return metrics

    monkeypatch.setattr(measure, "run_rl_experiment", recording)
    config = tiny_rl()
    start = warm_start(config)
    timed = measure_change(config, 0, change, start, True, steps=2, warmup=1)
    assert [(cut, precision) for _, cut, precision in given] == expected
    (off, _, _), (on, _, _) = given
    assert off is not start and on is not start and off is not on
    assert set(timed.off) == set(timed.on) == {"generate", "logprobs", "train", "step"}


def test_a_warm_start_that_ends_before_the_last_step_is_an_error():
    config = load_config(EXPERIMENTS / "hello.yaml", [*HELLO, "warm_start.tokens=2000"])
    with pytest.raises(ValueError, match=r"hello ran \d+ steps, not 6: its warm_start.tokens"):
        measure_change(config, 0, "padding-cut", None, True, steps=5, warmup=1)


def test_rl_needs_its_warm_start_and_a_warm_start_none():
    config = tiny_rl()
    with pytest.raises(ValueError, match="starts from its warm start"):
        measure_change(config, 0, "padding-cut", None, True)
    hello = load_config(EXPERIMENTS / "hello.yaml", HELLO)
    with pytest.raises(ValueError, match="starts from its warm start"):
        measure_change(hello, 0, "padding-cut", warm_start(config), True)
