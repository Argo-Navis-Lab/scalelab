"""Metrics sink tests: rows come back as recorded, part files are written every so often and
at checkpoints, a resumed run carries on its parts and its newest rows win, and a real RL run
writes its metrics at every checkpoint."""

import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from scalelab.core.checkpoint import CheckpointPolicy
from scalelab.core.config import AlgoConfig, ModelConfig, RLConfig
from scalelab.core.determinism import seed_everything
from scalelab.models.transformer import Transformer
from scalelab.rl.loop import rl_train
from scalelab.tasks.arithmetic import ArithmeticTask
from scalelab.telemetry.hooks import CheckpointEvent, Hook, RolloutEvent
from scalelab.telemetry.sink import MetricsSink, metrics_folder, read_metrics

RLOO = Path(__file__).resolve().parents[2] / "configs" / "algos" / "rloo.yaml"
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1, context=32)


def parts(home: Path, run_id: str) -> list[str]:
    return sorted(path.name for path in metrics_folder(home, run_id).glob("*"))


def test_rows_come_back_as_they_were_recorded(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    before = time.time()
    sink.record(1, "reward", 0.25)
    sink.record(1, "grad_norm", 2.0, layer=3)
    sink.flush()
    rows = read_metrics(tmp_path, "run-a")
    assert [(r.run_id, r.step, r.metric, r.layer, r.value) for r in rows] == [
        ("run-a", 1, "reward", None, 0.25),
        ("run-a", 1, "grad_norm", 3, 2.0),
    ]
    assert all(before <= r.wall_time <= time.time() for r in rows)


def test_the_files_have_the_blueprints_columns(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    sink.record(4, "kl", 0.5)
    sink.flush()
    table = pq.read_table(metrics_folder(tmp_path, "run-a") / "part-000000.parquet")
    assert table.column_names == ["run_id", "step", "wall_time", "metric", "layer", "value"]
    assert [str(t) for t in table.schema.types] == [
        "string",
        "int64",
        "double",
        "string",
        "int64",
        "double",
    ]
    assert table.column("layer").to_pylist() == [None]  # a whole-model metric has no layer


def test_nothing_is_written_until_a_checkpoint_or_30_seconds(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a", clock=lambda: 0.0)  # time stands still
    sink.record(1, "reward", 0.5)
    assert parts(tmp_path, "run-a") == []  # only in memory so far
    sink.on_checkpoint(CheckpointEvent(step=1))
    assert parts(tmp_path, "run-a") == ["part-000000.parquet"]


def test_rows_are_written_every_30_seconds_as_they_arrive(tmp_path: Path):
    now = [0.0]
    sink = MetricsSink(tmp_path, "run-a", clock=lambda: now[0])
    for seconds, step in [(10, 1), (31, 2), (40, 3), (62, 4)]:
        now[0] = seconds
        sink.record(step, "reward", 0.0)
    # Written at 31 s (steps 1 and 2) and at 62 s (steps 3 and 4).
    assert parts(tmp_path, "run-a") == ["part-000000.parquet", "part-000001.parquet"]
    folder = metrics_folder(tmp_path, "run-a")
    first = pq.read_table(folder / "part-000000.parquet").column("step").to_pylist()
    assert first == [1, 2]


def test_an_empty_flush_writes_nothing(tmp_path: Path):
    MetricsSink(tmp_path, "run-a").flush()
    assert parts(tmp_path, "run-a") == []


def test_a_resumed_run_carries_on_its_parts_and_its_newest_rows_win(tmp_path: Path):
    first = MetricsSink(tmp_path, "run-a")
    for step in (1, 2, 3):
        first.record(step, "reward", step / 10)
    first.flush()  # step 3 was written, then the run died before its next checkpoint
    again = MetricsSink(tmp_path, "run-a")  # the resumed session redoes step 3
    again.record(3, "reward", 0.33)
    again.record(4, "reward", 0.4)
    again.flush()
    assert parts(tmp_path, "run-a") == ["part-000000.parquet", "part-000001.parquet"]
    rows = read_metrics(tmp_path, "run-a")
    assert [(r.step, r.value) for r in rows] == [(1, 0.1), (2, 0.2), (3, 0.33), (4, 0.4)]


def test_a_real_run_writes_its_metrics_at_every_checkpoint(tmp_path: Path):
    class Rewards(Hook):
        """A one-metric collector: each step's mean reward."""

        def __init__(self, sink: MetricsSink) -> None:
            self.sink = sink

        def on_rollout(self, event: RolloutEvent) -> None:
            self.sink.record(event.step, "reward", event.rewards.mean().item())

    sink = MetricsSink(tmp_path, "run-a")
    seeds = seed_everything(0, deterministic=True)
    metrics = rl_train(
        Transformer(TINY),
        ArithmeticTask(),
        [2],
        AlgoConfig.model_validate(yaml.safe_load(RLOO.read_text())),
        RLConfig(from_run="0123456789abcdef-s0-01234567", flops=1e15, max_steps=5, lr=1e-2),
        np.random.default_rng(seeds.data),
        torch.Generator().manual_seed(seeds.sampling),
        tmp_path / "ckpt",
        tmp_path,
        policy=CheckpointPolicy(every_steps=2),  # checkpoints at steps 2, 4 and 5 (the last)
        hooks=[Rewards(sink), sink],
    )
    folder = metrics_folder(tmp_path, "run-a")
    steps = [
        pq.read_table(folder / name).column("step").to_pylist() for name in parts(tmp_path, "run-a")
    ]
    assert steps == [[1, 2], [3, 4], [5]]  # one part per checkpoint
    rows = read_metrics(tmp_path, "run-a")
    assert [r.value for r in rows] == [m.reward for m in metrics]


# --- added in m2s35: each step's values, for the detectors ------------------------------------


def test_values_are_the_newest_steps_metrics_with_no_layer(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run")
    sink.record(1, "optim/loss", 0.5)
    sink.record(2, "optim/loss", 0.25)
    sink.record(2, "optim/layer_grad_norm", 3.0, layer=1)
    sink.flush()  # writing the rows keeps the values
    sink.record(2, "policy/entropy", 0.75)
    assert sink.values(2) == {"optim/loss": 0.25, "policy/entropy": 0.75}
    assert sink.values(1) == {}  # an older step's are gone
    assert sink.values(3) == {}
