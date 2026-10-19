"""One run of an experiment: the model, task and data built from its config and seed."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from scalelab.core.checkpoint import CheckpointPolicy
from scalelab.core.config import ExperimentConfig, StrictModel
from scalelab.core.determinism import seed_everything
from scalelab.models.transformer import Transformer
from scalelab.rl.evaluation import EvalResult, evaluate
from scalelab.rl.warmstart import eval_prompts
from scalelab.rl.warmstart_loop import StepMetrics, warm_start
from scalelab.tasks.arithmetic import ArithmeticProblem, ArithmeticTask
from scalelab.tasks.base import Task
from scalelab.tasks.countdown import CountdownProblem, CountdownTask
from scalelab.tasks.evalsets import EVAL_SETS, eval_set_name, read_eval_set

# Called at each grading with the step's metrics and one result per eval set.
Report = Callable[[StepMetrics, list[EvalResult]], None]


def make_task(name: Literal["arithmetic", "countdown"]) -> tuple[Task[Any], type[StrictModel]]:
    """The task, and the type of its problems (what its eval set files hold)."""
    if name == "arithmetic":
        return ArithmeticTask(), ArithmeticProblem
    return CountdownTask(), CountdownProblem


def load_eval_sets[P: StrictModel](
    task: Task[P], problem_type: type[P], difficulties: Sequence[int], problems: int
) -> dict[str, list[P]]:
    """The first `problems` of each frozen eval set at the training difficulties."""
    sets: dict[str, list[P]] = {}
    for difficulty in difficulties:
        name = eval_set_name(task.name, difficulty)
        if name in EVAL_SETS:  # some levels have none, such as 9-digit arithmetic
            sets[name] = read_eval_set(name, problem_type)[:problems]
    return sets


def run_experiment(
    config: ExperimentConfig,
    seed: int,
    checkpoint_dir: str | Path,
    home: Path | None = None,
    policy: CheckpointPolicy | None = None,
    deterministic: bool | None = None,
    report: Report | None = None,
) -> list[StepMetrics]:
    """Warm-start the config's model on its task, resuming from the latest checkpoint in
    `checkpoint_dir` if there is one. Returns the metrics of every step this call ran.

    The seed's "init" stream picks the starting weights and its "data" stream the training
    problems. The model trains on the GPU when there is one, otherwise on the CPU. With a
    `report`, the model is graded every config.eval.every steps and after the last one, and
    each grading goes to `report`. Grading changes nothing about training.
    """
    seeds = seed_everything(seed, deterministic)  # PyTorch's generator now follows seeds.init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(config.model).to(device)
    task, problem_type = make_task(config.task.name)
    eval_sets = load_eval_sets(task, problem_type, config.task.difficulties, config.eval.problems)

    def grade(metrics: StepMetrics) -> None:
        last = metrics.tokens >= config.warm_start.tokens
        due = config.eval.every > 0 and metrics.step % config.eval.every == 0
        if report is not None and (last or due):
            results = [evaluate(model, task, eval_sets[name], name) for name in eval_sets]
            report(metrics, results)

    return warm_start(
        model,
        task,
        config.task.difficulties,
        config.warm_start,
        np.random.default_rng(seeds.data),
        checkpoint_dir,
        home,
        policy,
        eval_prompts(task, problem_type),
        on_step=grade,
    )
