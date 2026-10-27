"""An RL experiment's step and whole run, as the cost model sees them (m3s9)."""

import math
import statistics
from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from scalelab.core.config import ExperimentConfig, RuntimeConfig, load_config
from scalelab.costmodel.memory import generation_memory, training_memory
from scalelab.costmodel.rlstep import generation_time, predict_rl_step, predict_warm_start_step
from scalelab.costmodel.specs import CPU, HARDWARE, ModelSpec
from scalelab.rl.costing import (
    ANSWER_SAMPLES,
    GRADING_SAMPLES,
    GREEDY_BATCH,
    SAMPLED_BATCH,
    padding_cut,
    predict_gradings,
    predict_job,
    predict_rl_run,
    predict_run_step,
    predict_step,
    rl_step_spec,
    warm_start_step_spec,
)
from scalelab.rl.evaluation import evaluate_sampled, generate_greedy
from scalelab.rl.experiment import load_eval_sets, make_task
from scalelab.rl.loop import GENERATION_BATCH, sample_problems
from scalelab.rl.rollout import generate
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


def test_the_padding_cut_keeps_the_rows_and_drops_padding() -> None:
    rl = rl_config("task.difficulties=[2, 3]")  # 2- and 3-digit prompts, so lengths differ
    before, cut = predict_step(rl, CPU), padding_cut(rl, CPU)
    assert cut.phases["generate"] == before.phases["generate"]  # decoding never pads
    assert cut.phases["logprobs"].flops < before.phases["logprobs"].flops
    assert cut.phases["train"].flops < before.phases["train"].flops
    warm = load_config(EXPERIMENTS / "warmstart.yaml", ["task.difficulties=[2, 3]"])
    assert padding_cut(warm, CPU).flops < predict_step(warm, CPU).flops
    whole = padding_cut(warm, CPU, micro_batch=64)  # one piece: the batch as it comes
    assert whole.seconds == pytest.approx(predict_step(warm, CPU).seconds, rel=1e-9)


def test_specs_take_a_precision() -> None:
    rl, warm = rl_config(), load_config(EXPERIMENTS / "warmstart.yaml")
    assert rl_step_spec(rl).precision == "fp32"  # what every run so far computed in
    assert rl_step_spec(rl, "fp16") == rl_step_spec(rl).model_copy(update={"precision": "fp16"})
    assert warm_start_step_spec(warm, "fp16").precision == "fp16"
    t4 = HARDWARE["t4"]
    for config in (rl, warm):  # m3s20: at the T4's datasheet peaks, fp16 steps are faster
        fp32, fp16 = predict_step(config, t4), predict_step(config, t4, precision="fp16")
        assert fp16.flops == fp32.flops  # the same matmuls: budgets buy the same steps
        assert fp16.seconds < fp32.seconds


# --- added in m3s23: a run's step, as its runtime ran it -----------------------------------------


def test_a_run_is_predicted_in_its_precision_and_with_its_cut() -> None:
    t4 = HARDWARE["t4"]
    for config in (rl_config(), load_config(EXPERIMENTS / "warmstart.yaml")):
        assert predict_run_step(config, RuntimeConfig(), t4) == predict_step(config, t4)
        fp16 = predict_run_step(config, RuntimeConfig(precision="fp16"), t4)
        assert fp16 == predict_step(config, t4, precision="fp16")
        cut = predict_run_step(config, RuntimeConfig(padding_cut=True), t4)
        assert cut == padding_cut(config, t4)
        both = predict_run_step(config, RuntimeConfig(precision="fp16", padding_cut=True), t4)
        assert both == padding_cut(config, t4, precision="fp16")
        assert both.seconds < cut.seconds  # the cut in fp16 is still fp16...
        assert both.seconds < fp16.seconds  # ...and still cut


# --- added in m4s3: gradings priced ---------------------------------------------------------


def test_gradings_are_priced_with_the_batches_and_samples_they_use() -> None:
    assert signature(generate_greedy).parameters["batch_size"].default == GREEDY_BATCH
    assert signature(generate).parameters["batch_size"].default == SAMPLED_BATCH
    assert signature(evaluate_sampled).parameters["samples"].default == GRADING_SAMPLES


def test_a_step_prices_its_generation_as_generation_time_does() -> None:
    config = rl_config()
    spec = rl_step_spec(config)
    model = ModelSpec.from_config(config.model)
    alone = generation_time(model, spec.rows, spec.answer_lengths, GENERATION_BATCH, CPU)
    assert predict_rl_step(model, spec, CPU).phases["generate"] == alone


def test_greedy_answers_are_one_a_problem_as_long_as_its_worked_solution() -> None:
    config = rl_config("eval.problems=200")
    task, problem_type = make_task(config.task.name, config.task.ops)
    (problems,) = load_eval_sets(task, problem_type, [2], 200).values()
    lengths = [len(encode(task.worked_solution(problem))) + 1 for problem in problems]
    model = ModelSpec.from_config(config.model)
    greedy = generation_time(model, [7] * 200, {7: lengths}, GREEDY_BATCH, CPU)
    sampled = generation_time(model, [7] * 800, {7: lengths}, SAMPLED_BATCH, CPU)
    run = predict_gradings(config, 1, CPU)
    assert run.greedy == pytest.approx(greedy.seconds, rel=1e-12)
    assert run.sampled == pytest.approx(sampled.seconds, rel=1e-12)


def test_a_run_grades_greedy_at_every_grading_and_samples_at_the_ends() -> None:
    run = predict_gradings(rl_config("eval.every=0"), 100, CPU)
    assert run.steps == (0, 1, 2, 3, 6, 10, 18, 32, 56, 100)
    assert run.seconds == pytest.approx(10 * run.greedy + 2 * run.sampled, rel=1e-12)
    assert 0 < run.greedy < run.sampled  # 4 answers to a problem cost more than 1
    with pytest.raises(ValueError, match="is a warm start"):
        predict_gradings(load_config(EXPERIMENTS / "warmstart.yaml", []), 100, CPU)


def test_an_rl_job_is_its_steps_in_its_runtime_and_its_gradings() -> None:  # m4s4
    config, t4 = rl_config("rl.max_steps=10"), HARDWARE["t4"]
    runtime = RuntimeConfig(precision="fp16", padding_cut=True)
    job = predict_job(config, runtime, t4)
    assert job.steps == predict_rl_run(config, t4, precision="fp16").steps == 10
    assert job.step == predict_run_step(config, runtime, t4).seconds
    assert job.gradings == predict_gradings(config, 10, t4).seconds
    assert job.seconds == pytest.approx(10 * job.step + job.gradings, rel=1e-12)
    model = ModelSpec.from_config(config.model)
    training = training_memory(model, 64, 256).total  # rl.micro_batch rows of the whole context
    assert job.memory == max(training, generation_memory(model, GENERATION_BATCH, 256).total)


def test_a_warm_start_job_takes_the_steps_its_token_budget_buys() -> None:  # m4s4
    config = load_config(EXPERIMENTS / "warmstart.yaml", ["warm_start.tokens=1000000"])
    lengths = warm_start_step_spec(config).lengths
    whole = predict_job(config, RuntimeConfig(), CPU)
    assert whole.steps == math.ceil(1_000_000 / (64 * statistics.fmean(n - 1 for n in lengths)))
    assert whole.gradings == 0
    cut = predict_job(config, RuntimeConfig(padding_cut=True), CPU)
    assert cut.steps == whole.steps
    assert cut.step == padding_cut(config, CPU).seconds
    model = ModelSpec.from_config(config.model)
    assert whole.memory == training_memory(model, 64, 256).total
    assert cut.memory == training_memory(model, 16, 256).total  # the cut's pieces of 16 rows
