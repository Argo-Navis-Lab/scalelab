"""Arithmetic task tests."""

import numpy as np
import pytest
from pydantic import ValidationError

from scalelab.tasks.arithmetic import MAX_DIGITS, MIN_DIGITS, ArithmeticProblem, ArithmeticTask, Op
from scalelab.tasks.base import Task
from scalelab.tasks.tokenizer import BOS, decode, encode


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- one problem ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "op", "b", "question", "answer"),
    [
        (347, "+", 852, "347+852=", 1199),
        (85, "-", 347, "85-347=", -262),
        (12, "*", 34, "12*34=", 408),
    ],
)
def test_question_and_answer(a: int, op: Op, b: int, question: str, answer: int):
    problem = ArithmeticProblem(a=a, op=op, b=b)
    assert problem.question == question
    assert problem.answer == answer


@pytest.mark.parametrize(
    "fields",
    [{"a": 1, "op": "/", "b": 2}, {"a": -5, "op": "+", "b": 3}, {"a": 1, "op": "+"}],
    ids=["unknown-op", "negative-number", "missing-number"],
)
def test_bad_problems_are_refused(fields: dict[str, object]):
    with pytest.raises(ValidationError):
        ArithmeticProblem.model_validate(fields)


def test_problem_survives_a_json_round_trip():
    problem = ArithmeticProblem(a=347, op="*", b=85)
    assert ArithmeticProblem.model_validate_json(problem.model_dump_json()) == problem


def test_prompt_is_bos_then_the_question():
    prompt = ArithmeticTask().format_prompt(ArithmeticProblem(a=12, op="+", b=34))
    assert prompt == [BOS, *encode("12+34=")]
    assert decode(prompt) == "<bos>12+34="


# --- sampling ------------------------------------------------------------------------------


def test_sampling_is_pinned():
    # Golden values: the problems seed 0 gives. If NumPy or this code ever changes them, every
    # dataset drawn from a seed silently changes. This test makes that loud.
    assert ArithmeticTask().sample(rng(0), 3, difficulty=3) == [
        ArithmeticProblem(a=865, op="-", b=673),
        ArithmeticProblem(a=342, op="+", b=377),
        ArithmeticProblem(a=167, op="+", b=114),
    ]


def test_same_seed_same_problems():
    task = ArithmeticTask()
    assert task.sample(rng(7), 50, 4) == task.sample(rng(7), 50, 4)


def test_different_seeds_give_different_problems():
    task = ArithmeticTask()
    assert task.sample(rng(0), 50, 4) != task.sample(rng(1), 50, 4)


def test_asking_for_more_only_adds_to_the_end():
    task = ArithmeticTask()
    assert task.sample(rng(0), 20, 4)[:10] == task.sample(rng(0), 10, 4)


@pytest.mark.parametrize("difficulty", [MIN_DIGITS, 5, MAX_DIGITS])
def test_every_number_has_the_requested_digits(difficulty: int):
    for problem in ArithmeticTask().sample(rng(0), 200, difficulty):
        assert len(str(problem.a)) == len(str(problem.b)) == difficulty


def test_the_default_mixes_all_three_operations():
    problems = ArithmeticTask().sample(rng(0), 300, 3)
    assert {p.op for p in problems} == {"+", "-", "*"}


def test_only_the_chosen_operations_appear():
    problems = ArithmeticTask("+*").sample(rng(0), 300, 3)
    assert {p.op for p in problems} == {"+", "*"}


def test_the_order_of_ops_does_not_matter():
    assert ArithmeticTask("*+").sample(rng(0), 50, 3) == ArithmeticTask("+*").sample(rng(0), 50, 3)


@pytest.mark.parametrize("difficulty", range(MIN_DIGITS, MAX_DIGITS + 1))
def test_every_prompt_decodes_back_and_has_2d_plus_3_tokens(difficulty: int):
    task = ArithmeticTask()
    for problem in task.sample(rng(0), 50, difficulty):
        prompt = task.format_prompt(problem)
        assert decode(prompt) == "<bos>" + problem.question
        assert len(prompt) == 2 * difficulty + 3  # <bos>, two numbers, the operation and "="


# --- mistakes are caught -------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", [MIN_DIGITS - 1, MAX_DIGITS + 1])
def test_difficulty_out_of_range_is_refused(difficulty: int):
    with pytest.raises(ValueError, match="number of digits"):
        ArithmeticTask().sample(rng(0), 1, difficulty)


@pytest.mark.parametrize(
    "ops", ["", "/", "+x", "++"], ids=["empty", "division", "unknown", "repeated"]
)
def test_bad_ops_are_refused(ops: str):
    with pytest.raises(ValueError, match="ops must be"):
        ArithmeticTask(ops)


# --- the Task interface --------------------------------------------------------------------


def test_arithmetic_task_is_a_task():
    # The real check happens before any test runs: pyright compares ArithmeticTask with the
    # Task protocol on this line, and fails (in VS Code and in CI) if anything doesn't match.
    task: Task[ArithmeticProblem] = ArithmeticTask()
    assert task.name == "arithmetic"
