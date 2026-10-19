"""The resume contract for warm starts: interrupted and resumed on CPU, a warm start is
bit-identical to one that ran straight through."""

from pathlib import Path

import numpy as np
import torch

from scalelab.core.checkpoint import CheckpointPolicy, latest_checkpoint, load_checkpoint
from scalelab.core.config import ModelConfig, WarmStartConfig
from scalelab.core.determinism import seed_everything
from scalelab.models.transformer import Transformer
from scalelab.rl.warmstart_loop import StepMetrics, warm_start
from scalelab.tasks.arithmetic import ArithmeticTask

TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1)
CONFIG = WarmStartConfig(tokens=20_000, batch_size=16, lr=3e-3, warmup=0.1)  # 25 steps


def session(home: Path, folder: str, stop_after: int | None = None) -> list[StepMetrics]:
    """One session of a run, as a fresh process would start it: everything built again from
    the seed, then the latest checkpoint (if any) picks up where the last session stopped."""
    seeds = seed_everything(0, deterministic=True)
    return warm_start(
        Transformer(TINY),
        ArithmeticTask(),
        [2, 3],
        CONFIG,
        np.random.default_rng(seeds.data),
        home / folder,
        home=home,
        policy=CheckpointPolicy(every_steps=10),
        stop_after=stop_after,
    )


def final_weights(home: Path, folder: str) -> dict[str, torch.Tensor]:
    latest = latest_checkpoint(home / folder)
    assert latest is not None
    return load_checkpoint(latest, home)[1]["model"]


def test_stop_and_resume_is_bit_identical(tmp_path: Path):
    uninterrupted = session(tmp_path, "a")
    before = session(tmp_path, "b", stop_after=13)
    after = session(tmp_path, "b")
    assert (len(before), len(after)) == (13, len(uninterrupted) - 13)
    assert before + after == uninterrupted  # every loss, every gradient norm, exactly
    a, b = final_weights(tmp_path, "a"), final_weights(tmp_path, "b")
    assert all(torch.equal(a[name], b[name]) for name in a)


def test_interrupted_twice_is_still_bit_identical(tmp_path: Path):
    uninterrupted = session(tmp_path, "a")
    first = session(tmp_path, "b", stop_after=4)
    second = session(tmp_path, "b", stop_after=17)
    third = session(tmp_path, "b")
    assert first + second + third == uninterrupted


def test_a_finished_warm_start_has_nothing_left_to_do(tmp_path: Path):
    assert len(session(tmp_path, "a")) == 25
    assert session(tmp_path, "a") == []
