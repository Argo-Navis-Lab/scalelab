"""What an experiment asks of the cost model: an RL experiment's step, built from its config the
way the loop runs it, and its whole run: how many steps the FLOPs budget buys and how long they
take; a warm start's step, built the same way."""

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from scalelab.core.config import ExperimentConfig, RuntimeConfig, StrictModel
from scalelab.costmodel.memory import generation_memory, training_memory
from scalelab.costmodel.rlstep import (
    RLStepSpec,
    StepPrediction,
    WarmStartStepSpec,
    generation_time,
    predict_rl_step,
    predict_warm_start_step,
)
from scalelab.costmodel.roofline import Efficiency, Group
from scalelab.costmodel.specs import HardwareSpec, ModelSpec, Precision
from scalelab.costmodel.validate import PARTS
from scalelab.rl.experiment import grading_steps, load_eval_sets, make_task
from scalelab.rl.loop import GENERATION_BATCH, sample_problems
from scalelab.rl.warmstart import WARM_START_PIECES, sample_examples
from scalelab.tasks.tokenizer import encode
from scalelab.telemetry.sink import MetricRow

ANSWER_SAMPLES = 1000  # problems drawn per difficulty, for the answers' lengths
SPEC_SEED = 0  # they're drawn from their own generator: a config always gets the same step
SKIPPED_STEPS = 1  # step 1 also warms up (first allocations, kernel choices): left out (m3s10)
TIMED = {f"system/{part}_s": part for part in PARTS}  # telemetry's laps of a step (m2s29)


def rl_step_spec(config: ExperimentConfig, precision: Precision = "fp32") -> RLStepSpec:
    """An RL experiment's step as the cost model sees it. Its prompts: the first
    prompts_per_step of ANSWER_SAMPLES problems per difficulty, drawn as the loop draws them.
    Its answers: those problems' worked solutions, what the warm start was taught to write (its
    own answers are as long on average, and their tail runs a little longer: m3s8). Its precision:
    what its steps compute in (m3s20: fp16 is autocast's, as m3s18's op graph runs it)."""
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
        precision=precision,
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
    precision: Precision = "fp32",
) -> RunPrediction:
    """How long an RL experiment's steps take on `hardware`. The loop stops once the FLOPs it
    has spent reach the budget, so a budget of F buys ceil(F / f) steps of f FLOPs."""
    spec = rl_step_spec(config, precision)
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


def warm_start_step_spec(
    config: ExperimentConfig, precision: Precision = "fp32"
) -> WarmStartStepSpec:
    """A warm start's step as the cost model sees it (m3s11): its batch, and how long examples
    are: ANSWER_SAMPLES worked examples per difficulty, drawn as the loop draws them (without
    skipping the eval sets' prompts, as the loop does: they are as long as the rest)."""
    if config.rl is not None:
        raise ValueError(f"{config.name} is an RL experiment: its steps are RL steps")
    task, _ = make_task(config.task.name, config.task.ops)
    difficulties = config.task.difficulties
    count = ANSWER_SAMPLES * len(difficulties)
    rng = np.random.default_rng(SPEC_SEED)
    examples = sample_examples(task, rng, count, difficulties, config.model.context)
    return WarmStartStepSpec(
        lengths=tuple(len(example.prompt) + len(example.answer) for example in examples),
        batch_size=config.warm_start.batch_size,
        precision=precision,
    )


def predict_step(
    config: ExperimentConfig,
    hardware: HardwareSpec,
    efficiencies: Mapping[Group, Efficiency] | None = None,
    precision: Precision = "fp32",
) -> StepPrediction:
    """Any experiment's expected step on `hardware` in `precision`: an RL step, or a warm start's
    (m3s11)."""
    if config.rl is not None:
        return predict_rl_run(config, hardware, efficiencies, precision).step
    model = ModelSpec.from_config(config.model)
    spec = warm_start_step_spec(config, precision)
    return predict_warm_start_step(model, spec, hardware, efficiencies)


def padding_cut(
    config: ExperimentConfig,
    hardware: HardwareSpec,
    efficiencies: Mapping[Group, Efficiency] | None = None,
    micro_batch: int | None = None,
    precision: Precision = "fp32",
) -> StepPrediction:
    """An experiment's step under the padding cut (m3s15): its rows sorted by length, then run
    micro_batch at a time (default: RL's rl.micro_batch, a WARM_START_PIECES-th of a warm
    start's batch), each piece padded only to its own longest row. The same rows and the same
    update: only padding goes. In `precision` (m3s23: fp16 runs can take the cut too)."""
    model = ModelSpec.from_config(config.model)
    if config.rl is not None:
        rl = rl_step_spec(config, precision)
        update = {"sort_rows": True, "micro_batch": micro_batch or rl.micro_batch}
        return predict_rl_step(model, rl.model_copy(update=update), hardware, efficiencies)
    warm = warm_start_step_spec(config, precision)
    rows = micro_batch or math.ceil(warm.batch_size / WARM_START_PIECES)
    cut = warm.model_copy(update={"micro_batch": rows})
    return predict_warm_start_step(model, cut, hardware, efficiencies)


def predict_run_step(
    config: ExperimentConfig,
    runtime: RuntimeConfig,
    hardware: HardwareSpec,
    efficiencies: Mapping[Group, Efficiency] | None = None,
) -> StepPrediction:
    """The step a run of `config` took with these runtime settings (m3s23): in its precision,
    and under the padding cut if it was on."""
    if runtime.padding_cut:
        return padding_cut(config, hardware, efficiencies, precision=runtime.precision)
    return predict_step(config, hardware, efficiencies, runtime.precision)


# m4s3: how gradings generate, as the cost model prices them (a test checks each against the
# default it copies: generate_greedy's batch_size, rollout.generate's, evaluate_sampled's samples)
GREEDY_BATCH = 128
SAMPLED_BATCH = 512
GRADING_SAMPLES = 4


class GradingPrediction(StrictModel):
    """An RL run's gradings on one device (m4s3), in fp32 as gradings always run (m3s22): greedy
    answers to every eval problem at each graded step, and GRADING_SAMPLES sampled answers to
    each at the first and the last only."""

    steps: tuple[int, ...]  # the steps graded, 0 and the last included
    greedy: float  # seconds: one grading's greedy answers
    sampled: float  # seconds: one grading's sampled answers

    @property
    def seconds(self) -> float:
        """All of them: greedy answers at every grading, sampled ones at the first and last."""
        return self.greedy * len(self.steps) + self.sampled * 2


def predict_gradings(
    config: ExperimentConfig,
    steps: int,
    hardware: HardwareSpec,
    efficiencies: Mapping[Group, Efficiency] | None = None,
) -> GradingPrediction:
    """How long an RL run of `steps` steps spends grading on `hardware`. Each eval set is
    answered on its own, as grade_rl answers them, and its answers are its problems' own worked
    solutions (as rl_step_spec's are the training problems')."""
    if config.rl is None:
        raise ValueError(f"{config.name} is a warm start: only RL runs are priced gradings")
    task, problem_type = make_task(config.task.name, config.task.ops)
    eval_sets = load_eval_sets(task, problem_type, config.task.difficulties, config.eval.problems)
    model = ModelSpec.from_config(config.model)
    greedy = sampled = 0.0
    for problems in eval_sets.values():
        prompts = [len(task.format_prompt(problem)) for problem in problems]
        answers: dict[int, list[int]] = defaultdict(list)  # prompt length -> its answers' lengths
        for prompt, problem in zip(prompts, problems, strict=True):
            answers[prompt].append(len(encode(task.worked_solution(problem))) + 1)  # then <eos>
        rows = [prompt for prompt in prompts for _ in range(GRADING_SAMPLES)]  # group by group
        greedy += generation_time(
            model, prompts, answers, GREEDY_BATCH, hardware, efficiencies
        ).seconds
        sampled += generation_time(
            model, rows, answers, SAMPLED_BATCH, hardware, efficiencies
        ).seconds
    return GradingPrediction(
        steps=tuple(grading_steps(config, steps)), greedy=greedy, sampled=sampled
    )


class JobPrediction(StrictModel):
    """One run on one device (m4s4), as `sweep plan` prices it: its steps, each one's seconds
    in the run's precision and with its padding cut, its gradings, and the memory it needs."""

    steps: int
    step: float  # seconds
    gradings: float  # seconds: an RL run's (a warm start's gradings come on top)
    memory: int  # bytes: training or generating, whichever needs more

    @property
    def seconds(self) -> float:
        return self.step * self.steps + self.gradings


def predict_job(
    config: ExperimentConfig,
    runtime: RuntimeConfig,
    hardware: HardwareSpec,
    efficiencies: Mapping[Group, Efficiency] | None = None,
) -> JobPrediction:
    """How long a run of `config` takes on `hardware` with these runtime settings, and the
    memory its training and its generating each need alone, by A's memory model (16-bit
    numbers: M4 runs in fp16). A warm start takes as many steps as its token budget buys: a
    batch reads every position of its rows but the last."""
    step = predict_run_step(config, runtime, hardware, efficiencies).seconds
    model = ModelSpec.from_config(config.model)
    context = config.model.context
    if config.rl is None:
        spec = warm_start_step_spec(config, runtime.precision)
        read = spec.batch_size * statistics.fmean(length - 1 for length in spec.lengths)
        steps = math.ceil(config.warm_start.tokens / read)
        rows = spec.batch_size
        if runtime.padding_cut:
            rows = math.ceil(rows / WARM_START_PIECES)
        memory = training_memory(model, rows, context).total
        return JobPrediction(steps=steps, step=step, gradings=0.0, memory=memory)
    run = predict_rl_run(config, hardware, efficiencies, runtime.precision)
    gradings = predict_gradings(config, run.steps, hardware, efficiencies)
    memory = max(
        training_memory(model, config.rl.micro_batch, context).total,
        generation_memory(model, GENERATION_BATCH, context).total,
    )
    return JobPrediction(steps=run.steps, step=step, gradings=gradings.seconds, memory=memory)
