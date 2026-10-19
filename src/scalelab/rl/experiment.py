"""One run of an experiment: the model, task and data built from its config and seed."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from scalelab.core.checkpoint import CheckpointPolicy
from scalelab.core.config import ExperimentConfig
from scalelab.core.determinism import seed_everything
from scalelab.models.transformer import Transformer
from scalelab.rl.warmstart import eval_prompts
from scalelab.rl.warmstart_loop import StepMetrics, warm_start
from scalelab.tasks.arithmetic import ArithmeticProblem, ArithmeticTask
from scalelab.tasks.base import Task
from scalelab.tasks.countdown import CountdownProblem, CountdownTask


def make_task(
    name: Literal["arithmetic", "countdown"],
) -> tuple[Task[Any], frozenset[tuple[int, ...]]]:
    """The task, and the prompts of its eval sets, which training must never use."""
    if name == "arithmetic":
        return ArithmeticTask(), eval_prompts(ArithmeticTask(), ArithmeticProblem)
    return CountdownTask(), eval_prompts(CountdownTask(), CountdownProblem)


def run_experiment(
    config: ExperimentConfig,
    seed: int,
    checkpoint_dir: str | Path,
    home: Path | None = None,
    policy: CheckpointPolicy | None = None,
    deterministic: bool | None = None,
) -> list[StepMetrics]:
    """Warm-start the config's model on its task, resuming from the latest checkpoint in
    `checkpoint_dir` if there is one. Returns the metrics of every step this call ran.

    The seed's "init" stream picks the starting weights and its "data" stream the training
    problems. The model trains on the GPU when there is one, otherwise on the CPU.
    """
    seeds = seed_everything(seed, deterministic)  # PyTorch's generator now follows seeds.init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(config.model).to(device)
    task, excluded = make_task(config.task.name)
    return warm_start(
        model,
        task,
        config.task.difficulties,
        config.warm_start,
        np.random.default_rng(seeds.data),
        checkpoint_dir,
        home,
        policy,
        excluded,
    )
