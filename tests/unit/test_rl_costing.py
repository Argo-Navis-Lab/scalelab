"""An RL experiment's step and whole run, as the cost model sees them (m3s9)."""

from pathlib import Path

import numpy as np
import pytest

from scalelab.core.config import ExperimentConfig, load_config
from scalelab.costmodel.rlstep import predict_warm_start_step
from scalelab.costmodel.specs import CPU, ModelSpec
from scalelab.rl.costing import (
    ANSWER_SAMPLES,
    predict_rl_run,
    predict_step,
    rl_step_spec,
    warm_start_step_spec,
)
from scalelab.rl.experiment import make_task
from scalelab.rl.loop import GENERATION_BATCH, sample_problems
from scalelab.rl.warmstart import sample_examples
from scalelab.tasks.tokenizer import encode

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"


def rl_config(*overrides: str) -> ExperimentConfig:
    """rl.yaml on 2-digit multiplication, as the try-its run it, with `overrides` after."""
    return load_config(EXPERIMENTS / "rl.yaml", ["task.ops=*", "task.difficulties=[2]", *overrides])


def test_a_step_is_the_loops_prompts_with_their_worked_solutions() -> None:
    config = rl_config()
    spec = rl_step_spec(config)
    task, _ = make_task(config.task.name, config.task.ops)
    problems = sample_problems(task, np.random.default_rng(0), ANSWER_SAMPLES, [2])
    worked = tuple(len(encode(task.worked_solution(problem))) + 1 for problem in problems)
    assert spec.prompt_lengths == (7,) * 64  # <bos>ab*cd=
    assert spec.answer_lengths == {7: worked}  # <eos> included
    assert (spec.group_size, spec.micro_batch, spec.epochs) == (8, 64, 1)
    assert spec.generation_batch == GENERATION_BATCH
    assert spec.reference  # GRPO's KL penalty needs the reference's log-probs
    assert not rl_step_spec(rl_config("algo=rloo")).reference


def test_mixed_difficulties_keep_each_prompt_lengths_own_answers() -> None:
    config = rl_config("task.difficulties=[3,4]")
    spec = rl_step_spec(config)
    assert set(spec.prompt_lengths) == {9, 11}  # one step of 64 holds both
    assert spec.answer_lengths.keys() == {9, 11}
    assert sum(len(answers) for answers in spec.answer_lengths.values()) == 2 * ANSWER_SAMPLES
    assert rl_step_spec(config) == spec  # drawn from their own generator: always the same


def test_a_warm_start_has_no_rl_step() -> None:
    with pytest.raises(ValueError, match="is a warm start"):
        rl_step_spec(load_config(EXPERIMENTS / "warmstart.yaml", []))


def test_the_budget_buys_whole_steps_and_max_steps_can_stop_sooner() -> None:
    step = predict_rl_run(rl_config(), CPU).step
    assert step.flops == sum(phase.flops for phase in step.phases.values())
    budget = f"rl.flops={10.5 * step.flops}"
    run = predict_rl_run(rl_config(budget), CPU)
    assert (run.budget_steps, run.steps) == (11, 11)  # 10 steps leave FLOPs unspent: an 11th
    run = predict_rl_run(rl_config(budget, "rl.max_steps=3"), CPU)
    assert (run.budget_steps, run.steps) == (11, 3)
    assert run.step == step  # the budget sets how many steps, not what a step is
    assert run.seconds == pytest.approx(3 * step.seconds)


# --- added in m3s11: a warm start's step ----------------------------------------------------


def test_a_warm_start_step_is_its_batch_of_the_loops_worked_examples() -> None:
    config = load_config(EXPERIMENTS / "warmstart.yaml", ["task.difficulties=[2, 3]"])
    spec = warm_start_step_spec(config)
    assert spec.batch_size == config.warm_start.batch_size == 64
    task, _ = make_task(config.task.name, config.task.ops)
    rng = np.random.default_rng(0)
    examples = sample_examples(task, rng, 2 * ANSWER_SAMPLES, [2, 3], config.model.context)
    assert spec.lengths == tuple(len(e.prompt) + len(e.answer) for e in examples)
    assert warm_start_step_spec(config) == spec  # drawn from their own generator: always the same


def test_either_kind_of_experiment_has_a_step() -> None:
    warm = load_config(EXPERIMENTS / "warmstart.yaml", ["task.difficulties=[2]"])
    step = predict_step(warm, CPU)
    model = ModelSpec.from_config(warm.model)
    assert step == predict_warm_start_step(model, warm_start_step_spec(warm), CPU)
    assert list(step.phases) == ["train"]
    assert predict_step(rl_config(), CPU) == predict_rl_run(rl_config(), CPU).step
    with pytest.raises(ValueError, match="is an RL experiment"):
        warm_start_step_spec(rl_config())
