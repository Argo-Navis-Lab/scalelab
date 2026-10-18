"""Worked-solution tests: the working a perfect model writes for an arithmetic problem."""

import re

import numpy as np
import pytest

from scalelab.tasks.arithmetic import MAX_DIGITS, MIN_DIGITS, ArithmeticProblem, ArithmeticTask, Op
from scalelab.tasks.tokenizer import encode

TASK = ArithmeticTask()
CONTEXT = 256  # how many tokens a Track S model sees at once (blueprint Section 6.1)

# One step of working, such as "2*3+1*4+0=10,08": expression, "=", value, ",", result so far.
STEP = re.compile(r"(?P<expression>[-+*0-9]+)=(?P<value>-?\d+),(?P<result>\d+)")


def solve(a: int, op: Op, b: int) -> str:
    return TASK.worked_solution(ArithmeticProblem(a=a, op=op, b=b))


def sample(op: Op, difficulty: int, n: int = 100) -> list[ArithmeticProblem]:
    return ArithmeticTask(op).sample(np.random.default_rng(difficulty), n, difficulty)


def total_tokens(problem: ArithmeticProblem) -> int:
    """Everything the model reads or writes: the prompt, the working and answer, and <eos>."""
    return len(TASK.format_prompt(problem)) + len(encode(TASK.worked_solution(problem))) + 1


def evaluate(expression: str) -> int:
    """Work out an expression such as "2*3+1*4+0" or "8-6-1". The test has its own small
    calculator, so it doesn't trust the code it checks (and it never calls Python's eval)."""
    total = 0
    for sign, product in re.findall(r"([+-]?)(\d+(?:\*\d+)*)", expression):
        value = 1
        for factor in product.split("*"):
            value *= int(factor)
        total += -value if sign == "-" else value
    return total


# --- the format, pinned by hand-checked examples ----------------------------------------------


@pytest.mark.parametrize(
    ("a", "op", "b", "working"),
    [
        (347, "+", 852, "7+2+0=9,9;4+5+0=9,99;3+8+0=11,199;#1199"),
        (865, "-", 673, "5-3+0=2,2;6-7+0=-1,92;8-6-1=1,192;#192"),
        (673, "-", 865, "-;5-3+0=2,2;6-7+0=-1,92;8-6-1=1,192;#-192"),
        (865, "-", 863, "5-3+0=2,2;6-6+0=0,02;8-8+0=0,002;#2"),
        (12, "*", 34, "2*4+0=8,8;2*3+1*4+0=10,08;1*3+1=4,408;#408"),
        (1, "+", 999, "1+9+0=10,0;0+9+1=10,00;0+9+1=10,000;#1000"),
    ],
    ids=["add", "subtract", "negative-answer", "leading-zeros", "multiply", "different-lengths"],
)
def test_golden_worked_solutions(a: int, op: Op, b: int, working: str):
    assert solve(a, op, b) == working


# --- the working is correct, for every operation and difficulty -------------------------------


@pytest.mark.parametrize("op", ["+", "-", "*"])
@pytest.mark.parametrize("difficulty", range(MIN_DIGITS, MAX_DIGITS + 1))
def test_the_working_ends_with_the_right_answer(op: Op, difficulty: int):
    for problem in sample(op, difficulty):
        assert TASK.worked_solution(problem).endswith(f";#{problem.answer}")


@pytest.mark.parametrize("op", ["+", "-", "*"])
def test_every_step_is_a_true_equation_that_extends_the_result(op: Op):
    for problem in sample(op, 5):
        *steps, _answer = TASK.worked_solution(problem).split(";")
        if steps[0] == "-":
            steps = steps[1:]
        carry, result = 0, ""
        for step in steps:
            match = STEP.fullmatch(step)
            assert match is not None, step
            value = int(match["value"])
            assert match["expression"].endswith(f"{carry:+d}")  # the carry comes last
            assert evaluate(match["expression"]) == value  # a true equation
            result = str(value % 10) + result
            assert match["result"] == result  # the value's last digit joins the front
            carry = value // 10


def test_only_negative_answers_start_with_a_minus_step():
    problems = sample("-", 4, 300)
    for problem in problems:
        assert TASK.worked_solution(problem).startswith("-;") == (problem.answer < 0)
    assert {problem.answer < 0 for problem in problems} == {True, False}  # both kinds checked


# --- what fits in the model's context ---------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "op", "b", "tokens"),
    [
        (999999999999, "+", 999999999999, 240),  # the longest addition: every column carries
        (199999999999, "-", 900000000000, 251),  # the longest subtraction: many -10 columns
        (99999, "*", 99999, 238),  # the longest 5-digit multiplication: all nines
        (100000, "*", 100000, 293),  # the shortest 6-digit multiplication: already too long
    ],
)
def test_the_extremes(a: int, op: Op, b: int, tokens: int):
    assert total_tokens(ArithmeticProblem(a=a, op=op, b=b)) == tokens


@pytest.mark.parametrize(("op", "fits_up_to"), [("+", 12), ("-", 12), ("*", 5)])
def test_what_fits_in_the_context(op: Op, fits_up_to: int):
    for difficulty in range(MIN_DIGITS, fits_up_to + 1):
        assert max(total_tokens(p) for p in sample(op, difficulty)) <= CONTEXT
    for difficulty in range(fits_up_to + 1, MAX_DIGITS + 1):
        assert min(total_tokens(p) for p in sample(op, difficulty)) > CONTEXT
