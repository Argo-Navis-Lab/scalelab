"""Warm-start loop tests: the learning-rate schedule, the optimizer, the budget, the FLOP
ledger, the final checkpoint and the step times."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from scalelab.core.checkpoint import latest_checkpoint, load_checkpoint
from scalelab.core.config import ModelConfig, WarmStartConfig
from scalelab.costmodel.ops import count_flops
from scalelab.costmodel.specs import ModelSpec, WorkloadSpec
from scalelab.models.transformer import Transformer
from scalelab.rl.warmstart import make_batch, sample_examples
from scalelab.rl.warmstart_loop import StepMetrics, learning_rate, make_optimizer, warm_start
from scalelab.tasks.arithmetic import ArithmeticTask
from scalelab.telemetry.sink import MetricsSink, read_metrics

TASK = ArithmeticTask()
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1)
SMALL = WarmStartConfig(tokens=20_000, batch_size=16, lr=3e-3, warmup=0.1)


def make() -> Transformer:
    """The same starting weights every time, without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(TINY)


def train(home: Path) -> tuple[Transformer, list[StepMetrics]]:
    """A tiny warm start: 2-digit arithmetic until 20,000 positions have been read."""
    model = make()
    metrics = warm_start(model, TASK, [2], SMALL, np.random.default_rng(0), home / "ckpt", home)
    return model, metrics


# --- the config -----------------------------------------------------------------------------


def test_the_default_budget_is_100_million_tokens():
    assert WarmStartConfig().tokens == 100_000_000


@pytest.mark.parametrize(
    "fields",
    [{"tokens": 0}, {"warmup": 1.0}, {"decay_to": 1.5}, {"grad_clip": 0.0}],
    ids=["no-tokens", "all-warmup", "decay-upward", "no-gradient"],
)
def test_impossible_settings_are_refused(fields: dict[str, object]):
    with pytest.raises(ValidationError):
        WarmStartConfig.model_validate(fields)


# --- the learning-rate schedule -------------------------------------------------------------


def test_the_schedule_by_hand():
    config = WarmStartConfig(lr=1e-3, warmup=0.1, decay_to=0.1)
    assert learning_rate(config, 0.05) == pytest.approx(5e-4)  # halfway up the ramp
    assert learning_rate(config, 0.1) == pytest.approx(1e-3)  # the peak
    assert learning_rate(config, 0.55) == pytest.approx(5.5e-4)  # halfway down the cosine
    assert learning_rate(config, 1.0) == pytest.approx(1e-4)  # decay_to * lr at the end


def test_the_schedule_rises_during_the_warmup_then_falls():
    config = WarmStartConfig(warmup=0.1)
    rates = [learning_rate(config, i / 100) for i in range(101)]
    assert rates[:11] == sorted(rates[:11])  # up to the peak at 10%...
    assert rates[10:] == sorted(rates[10:], reverse=True)  # ...then down to the end


def test_without_a_warmup_the_schedule_starts_at_the_peak():
    assert learning_rate(WarmStartConfig(lr=1e-3, warmup=0.0), 0.0) == 1e-3


# --- the optimizer --------------------------------------------------------------------------


def test_only_the_matrices_get_weight_decay():
    optimizer = make_optimizer(make(), WarmStartConfig(weight_decay=0.1))
    decayed, kept = optimizer.param_groups
    counts = ModelSpec.from_config(TINY).param_counts()
    assert decayed["weight_decay"] == 0.1
    assert sum(p.numel() for p in decayed["params"]) == counts.total - counts.norms
    assert kept["weight_decay"] == 0.0
    assert sum(p.numel() for p in kept["params"]) == counts.norms


# --- training -------------------------------------------------------------------------------


def test_training_stops_once_the_budget_is_spent(tmp_path: Path):
    _, metrics = train(tmp_path)
    assert [m.step for m in metrics] == list(range(1, len(metrics) + 1))
    assert metrics[-2].tokens < SMALL.tokens <= metrics[-1].tokens  # the last step crossed it


def test_the_loss_falls(tmp_path: Path):
    _, metrics = train(tmp_path)
    assert metrics[-1].loss < 0.75 * metrics[0].loss


def test_every_step_uses_the_scheduled_learning_rate(tmp_path: Path):
    _, metrics = train(tmp_path)
    for m in metrics:
        assert m.lr == learning_rate(SMALL, min(1.0, m.tokens / SMALL.tokens))


def test_the_flop_ledger_counts_what_the_cost_model_predicts(tmp_path: Path):
    _, metrics = train(tmp_path)
    # The first step's batch, drawn again from the same seed: its padded shape is the work done.
    first = make_batch(sample_examples(TASK, np.random.default_rng(0), 16, [2], 256))
    sequences, width = first.inputs.shape
    workload = WorkloadSpec(phase="train", batch=sequences, tokens=width)
    assert metrics[0].flops == count_flops(ModelSpec.from_config(TINY), workload).total
    assert all(a.flops < b.flops for a, b in zip(metrics, metrics[1:], strict=False))


def test_on_step_sees_every_step(tmp_path: Path):
    seen: list[StepMetrics] = []
    metrics = warm_start(
        make(),
        TASK,
        [2],
        SMALL,
        np.random.default_rng(0),
        tmp_path / "ckpt",
        tmp_path,
        on_step=seen.append,  # each step's metrics are added to `seen` as they come
    )
    assert seen == metrics


def test_the_last_step_is_always_saved(tmp_path: Path):
    model, metrics = train(tmp_path)
    last = tmp_path / "ckpt" / f"step-{metrics[-1].step:09d}.json"
    assert latest_checkpoint(tmp_path / "ckpt") == last
    step, state = load_checkpoint(last, tmp_path)
    assert step == metrics[-1].step
    assert (state["tokens"], state["flops"]) == (metrics[-1].tokens, metrics[-1].flops)
    for name, tensor in model.state_dict().items():
        assert torch.equal(state["model"][name], tensor), name


# --- added in m3s11: the step times ---------------------------------------------------------


def test_with_a_sink_each_step_records_its_laps_and_saving_writes_them(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run", every_seconds=math.inf)  # it writes when saving alone
    ckpt = tmp_path / "ckpt"
    metrics = warm_start(
        make(), TASK, [2], SMALL, np.random.default_rng(0), ckpt, tmp_path, sink=sink
    )
    rows = read_metrics(tmp_path, "run")  # on disk: the last step's save wrote them
    for m in metrics:
        laps = {row.metric: row.value for row in rows if row.step == m.step}
        assert laps.keys() == {"system/data_s", "system/train_s", "system/step_s"}
        assert 0 < laps["system/data_s"] + laps["system/train_s"] <= laps["system/step_s"]
    assert metrics == train(tmp_path / "plain")[1]  # timing changes nothing about training
