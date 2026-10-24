"""One run of an experiment, a warm start or RL: the model, task and data built from its config
and seed."""

import copy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from scalelab.core.checkpoint import CheckpointPolicy, latest_checkpoint
from scalelab.core.config import ArithmeticOps, ExperimentConfig, StrictModel
from scalelab.core.determinism import seed_everything
from scalelab.models.transformer import Transformer
from scalelab.rl.evaluation import (
    EvalResult,
    SampledResult,
    evaluate,
    evaluate_by,
    evaluate_sampled,
)
from scalelab.rl.loop import RLStepMetrics, budget_spent, rl_train
from scalelab.rl.warmstart import eval_prompts
from scalelab.rl.warmstart_loop import StepMetrics, warm_start
from scalelab.tasks.arithmetic import ArithmeticProblem, ArithmeticTask
from scalelab.tasks.base import Task
from scalelab.tasks.countdown import CountdownProblem, CountdownTask
from scalelab.tasks.evalsets import EVAL_SETS, eval_set_name, read_eval_set
from scalelab.telemetry.anomalies import AnomalyEvent, AnomalyMonitor, ignore
from scalelab.telemetry.collectors import Cadence, make_collectors
from scalelab.telemetry.hooks import Hook
from scalelab.telemetry.sink import MetricsSink

# Called at each grading with the step's metrics and one result per eval set.
Report = Callable[[StepMetrics, list[EvalResult]], None]

# Called at each grading during RL with the step's metrics (None at step 0, the warm start's
# own grade), the greedy results and the sampled results.
RLReport = Callable[[RLStepMetrics | None, list[EvalResult], list[SampledResult]], None]

# Every sampled grading draws from a fresh generator seeded with this. So the same model
# always gets the same sampled grade, whatever the run, its seed or the step.
GRADING_SEED = 0


def make_task(
    name: Literal["arithmetic", "countdown"], ops: ArithmeticOps | None = None
) -> tuple[Task[Any], type[StrictModel]]:
    """The task, and the type of its problems (what its eval set files hold). `ops` narrows
    arithmetic to some of its operations; None keeps all three."""
    if name == "arithmetic":
        return ArithmeticTask() if ops is None else ArithmeticTask(ops), ArithmeticProblem
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


def operation(problem: ArithmeticProblem) -> str:
    """Which part of an arithmetic eval set a problem belongs to: its operation."""
    return problem.op


def grade_by_part(
    model: Transformer, task: Task[Any], eval_sets: dict[str, list[Any]]
) -> list[EvalResult]:
    """Greedy pass@1 on each eval set: arithmetic's whole and then by operation, Countdown's
    whole (it has no parts)."""
    results: list[EvalResult] = []
    for name, problems in eval_sets.items():
        if task.name == "arithmetic":
            results += evaluate_by(model, task, problems, name, key=operation)
        else:
            results.append(evaluate(model, task, problems, name))
    return results


def grade_rl(
    model: Transformer, task: Task[Any], eval_sets: dict[str, list[Any]]
) -> tuple[list[EvalResult], list[SampledResult]]:
    """RL's grading: greedy pass@1 on each eval set (grade_by_part), and 4 sampled answers at
    T 1 to each problem, split the same way. The samples come from a fresh generator seeded
    with GRADING_SEED, never from training's."""
    generator = torch.Generator(model.embed.weight.device).manual_seed(GRADING_SEED)
    sampled: list[SampledResult] = []
    for name, problems in eval_sets.items():
        result = evaluate_sampled(model, task, problems, name, generator)
        if task.name == "arithmetic":
            sampled += result.by_part(problems, key=operation)
        else:
            sampled.append(result)
    return grade_by_part(model, task, eval_sets), sampled


def run_experiment(
    config: ExperimentConfig,
    seed: int,
    checkpoint_dir: str | Path,
    home: Path | None = None,
    policy: CheckpointPolicy | None = None,
    deterministic: bool | None = None,
    report: Report | None = None,
    sink: MetricsSink | None = None,
) -> list[StepMetrics]:
    """Warm-start the config's model on its task, resuming from the latest checkpoint in
    `checkpoint_dir` if there is one. Returns the metrics of every step this call ran.

    The seed's "init" stream picks the starting weights and its "data" stream the training
    problems. The model trains on the GPU when there is one, otherwise on the CPU. With a
    `report`, the model is graded every config.eval.every steps and after the last one, and
    each grading goes to `report`. Grading changes nothing about training. Each step's laps
    go to `sink`, if there is one (m3s11).
    """
    seeds = seed_everything(seed, deterministic)  # PyTorch's generator now follows seeds.init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(config.model).to(device)
    task, problem_type = make_task(config.task.name, config.task.ops)
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
        sink=sink,
    )


def run_rl_experiment(
    config: ExperimentConfig,
    seed: int,
    start: Transformer,
    checkpoint_dir: str | Path,
    home: Path | None = None,
    policy: CheckpointPolicy | None = None,
    deterministic: bool | None = None,
    report: RLReport | None = None,
    sink: MetricsSink | None = None,
    on_anomaly: Callable[[AnomalyEvent], None] = ignore,
    on_step: Callable[[RLStepMetrics], None] | None = None,
) -> list[RLStepMetrics]:
    """RL on the config's task, starting from `start`: the model rl.from_run ended with,
    which is also the KL penalty's reference (a copy is kept; `start` itself is trained).
    Resumes from the latest checkpoint in `checkpoint_dir` if there is one. Returns the
    metrics of every step this call ran.

    The seed's "data" stream picks the problems and its "sampling" stream the answers. With
    a `report`, the model is graded before the first step (step 0: the warm start's own
    grade, R_0), every config.eval.every steps and after the last one. Grading changes
    nothing about training, and its FLOPs aren't part of the budget. With a `sink`, every
    collector measures every step and records it there (telemetry), and the online
    detectors judge what they measured: each event goes to `on_anomaly`. `on_step`, if
    given, is called with each step's metrics before that step is graded.
    """
    if config.algo is None or config.rl is None:
        raise ValueError(f"{config.name} is not an RL experiment: it has no algo and rl sections")
    if start.config != config.model:
        raise ValueError(
            f"the warm start is model {start.config.name}, the config's is {config.model.name}"
        )
    algo, rl = config.algo, config.rl
    seeds = seed_everything(seed, deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = start.to(device)
    reference = copy.deepcopy(model) if algo.kl_coef > 0 else None  # the warm start, kept
    task, problem_type = make_task(config.task.name, config.task.ops)
    eval_sets = load_eval_sets(task, problem_type, config.task.difficulties, config.eval.problems)
    hooks: list[Hook] = []
    monitor: AnomalyMonitor | None = None
    if sink is not None:
        cadence = Cadence()  # m2s35: the detectors' events make the detailed collectors measure
        hooks = [*make_collectors(sink, algo, cadence), sink]  # the sink last: it writes them
        monitor = AnomalyMonitor(sink, cadence, on_anomaly)

    def grade(metrics: RLStepMetrics | None) -> None:
        if report is None:
            return
        if metrics is not None:  # step 0 is always graded; later steps when due, and the last
            last = budget_spent(rl, metrics.step, metrics.flops)
            due = config.eval.every > 0 and metrics.step % config.eval.every == 0
            if not (last or due):
                return
        report(metrics, *grade_rl(model, task, eval_sets))

    def after_step(metrics: RLStepMetrics) -> None:
        if on_step is not None:
            on_step(metrics)  # m2s36: before grading, so a step's time leaves grading out
        grade(metrics)

    if latest_checkpoint(checkpoint_dir) is None:  # a new run: grade the warm start first
        grade(None)
    return rl_train(
        model,
        task,
        config.task.difficulties,
        algo,
        rl,
        np.random.default_rng(seeds.data),
        torch.Generator(device).manual_seed(seeds.sampling),
        checkpoint_dir,
        home,
        policy,
        reference,
        eval_prompts(task, problem_type),
        on_step=after_step,
        hooks=hooks,
        monitor=monitor,
    )
