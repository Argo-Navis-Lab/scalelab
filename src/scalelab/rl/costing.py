"""What an RL experiment asks of the cost model: its step, built from its config the way the loop
runs it, and its whole run: how many steps the FLOPs budget buys and how long they take."""

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from scalelab.core.config import ExperimentConfig, StrictModel
from scalelab.costmodel.rlstep import RLStepSpec, StepPrediction, predict_rl_step
from scalelab.costmodel.roofline import Efficiency, Group
from scalelab.costmodel.specs import HardwareSpec, ModelSpec
from scalelab.costmodel.validate import PARTS
from scalelab.rl.experiment import make_task
from scalelab.rl.loop import GENERATION_BATCH, sample_problems
from scalelab.tasks.tokenizer import encode
from scalelab.telemetry.sink import MetricRow

ANSWER_SAMPLES = 1000  # problems drawn per difficulty, for the answers' lengths
SPEC_SEED = 0  # they're drawn from their own generator: a config always gets the same step
SKIPPED_STEPS = 1  # step 1 also warms up (first allocations, kernel choices): left out (m3s10)
TIMED = {f"system/{part}_s": part for part in PARTS}  # telemetry's laps of a step (m2s29)


def rl_step_spec(config: ExperimentConfig) -> RLStepSpec:
    """An RL experiment's step as the cost model sees it. Its prompts: the first
    prompts_per_step of ANSWER_SAMPLES problems per difficulty, drawn as the loop draws them.
    Its answers: those problems' worked solutions, what the warm start was taught to write (its
    own answers are as long on average, and their tail runs a little longer: m3s8)."""
    if config.algo is None or config.rl is None:
        raise ValueError(f"{config.name} is a warm start: only RL experiments have RL steps")
    task, _ = make_task(config.task.name, config.task.ops)
    difficulties = config.task.difficulties
    count = max(ANSWER_SAMPLES * len(difficulties), config.rl.prompts_per_step)
    problems = sample_problems(task, np.random.default_rng(SPEC_SEED), count, difficulties)
    prompts = [len(task.format_prompt(problem)) for problem in problems]
    answers: dict[int, list[int]] = defaultdict(list)  # prompt length -> its answers' lengths
    for prompt, problem in zip(prompts, problems, strict=True):
        answers[prompt].append(len(encode(task.worked_solution(problem))) + 1)  # then <eos>
    return RLStepSpec(
        prompt_lengths=tuple(prompts[: config.rl.prompts_per_step]),
        answer_lengths={prompt: tuple(lengths) for prompt, lengths in answers.items()},
        group_size=config.rl.group_size,
        generation_batch=GENERATION_BATCH,
        micro_batch=config.rl.micro_batch,
        epochs=config.rl.epochs,
        reference=config.algo.kl_coef > 0,
    )


class RunPrediction(StrictModel):
    """A whole RL run on one device: its step, and how many steps it takes."""

    spec: RLStepSpec
    step: StepPrediction
    budget_steps: int  # the steps its FLOPs budget buys, at the expected step's FLOPs
    steps: int  # the steps it takes: budget_steps, or rl.max_steps when that is fewer

    @property
    def seconds(self) -> float:
        """The steps alone: gradings, checkpoints and start-up come on top."""
        return self.step.seconds * self.steps


def predict_rl_run(
    config: ExperimentConfig,
    hardware: HardwareSpec,
    efficiencies: Mapping[Group, Efficiency] | None = None,
) -> RunPrediction:
    """How long an RL experiment's steps take on `hardware`. The loop stops once the FLOPs it
    has spent reach the budget, so a budget of F buys ceil(F / f) steps of f FLOPs."""
    spec = rl_step_spec(config)
    step = predict_rl_step(ModelSpec.from_config(config.model), spec, hardware, efficiencies)
    assert config.rl is not None  # rl_step_spec checked
    budget_steps = math.ceil(config.rl.flops / step.flops)
    steps = budget_steps if config.rl.max_steps is None else min(budget_steps, config.rl.max_steps)
    return RunPrediction(spec=spec, step=step, budget_steps=budget_steps, steps=steps)


def measured_step(
    rows: Iterable[MetricRow], skip: int = SKIPPED_STEPS
) -> tuple[int, dict[str, float]]:
    """A run's measured seconds a step: each part's mean over its steps after the first `skip`,
    and how many steps those are. Runs from before m2s29, or with runtime.telemetry=false,
    timed none: (0, {})."""
    seconds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.metric in TIMED and row.step > skip:
            seconds[TIMED[row.metric]].append(row.value)
    means = {part: statistics.fmean(values) for part, values in seconds.items()}
    return len(seconds.get("step", [])), means
