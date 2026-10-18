"""Countdown tests: problems, prompts, worked solutions and scoring."""

import re

import numpy as np
import pytest
from pydantic import ValidationError

from scalelab.tasks.base import Task
from scalelab.tasks.countdown import (
    LARGEST_NUMBER,
    MAX_NUMBERS,
    MAX_VALUE,
    MIN_NUMBERS,
    CountdownProblem,
    CountdownTask,
)
from scalelab.tasks.rewards import Reward
from scalelab.tasks.tokenizer import BOS, encode

TASK = CountdownTask()
PROBLEM = CountdownProblem(numbers=(19, 36, 55, 7), target=65, solution="55+36-19-7")
CONTEXT = 256  # how many tokens a Track S model sees at once (blueprint Section 6.1)
STEP = re.compile(r"(\d+)[-+*/](\d+)=(\d+)")  # one step of working, such as "91-19=72"


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def score(answer: str) -> Reward:
    return TASK.score(PROBLEM, "#" + answer)


# --- one problem ---------------------------------------------------------------------------


def test_question_and_prompt():
    assert PROBLEM.question == "19,36,55,7:65="
    assert TASK.format_prompt(PROBLEM) == [BOS, *encode("19,36,55,7:65=")]


@pytest.mark.parametrize(
    "fields",
    [
        {"numbers": (19, 36, 55, 7), "target": 65, "solution": "55+36-19"},
        {"numbers": (19, 36, 55, 7), "target": 65, "solution": "55+36-19-7+7-7"},
        {"numbers": (19, 36, 55, 7), "target": 66, "solution": "55+36-19-7"},
        {"numbers": (19, 36, 55, 7), "target": 65, "solution": "55+36-19-(7"},
        {"numbers": (2, 3, 6), "target": 4, "solution": "2/3*6"},
        {"numbers": (2, 2, 6), "target": 0, "solution": "6/(2-2)"},
    ],
    ids=[
        "leaves-a-number-out",
        "uses-a-number-twice",
        "misses-the-target",
        "malformed",
        "fraction-on-the-way",
        "divides-by-zero",
    ],
)
def test_inconsistent_problems_are_refused(fields: dict[str, object]):
    with pytest.raises(ValidationError):
        CountdownProblem.model_validate(fields)


def test_problem_survives_a_json_round_trip():
    assert CountdownProblem.model_validate_json(PROBLEM.model_dump_json()) == PROBLEM


@pytest.mark.parametrize(
    ("numbers", "target", "solution", "working"),
    [
        ((19, 36, 55, 7), 65, "55+36-19-7", "55+36=91;91-19=72;72-7=65;#55+36-19-7"),
        ((25, 3, 7, 3), 157, "(25-3)*7+3", "25-3=22;22*7=154;154+3=157;#(25-3)*7+3"),
        ((2, 3, 10, 4), 12, "2*3+(10-4)", "2*3=6;10-4=6;6+6=12;#2*3+(10-4)"),
    ],
    ids=["one-chain", "brackets", "two-branches"],
)
def test_golden_worked_solutions(
    numbers: tuple[int, ...], target: int, solution: str, working: str
):
    problem = CountdownProblem(numbers=numbers, target=target, solution=solution)
    assert TASK.worked_solution(problem) == working


# --- sampling ------------------------------------------------------------------------------


def test_sampling_is_pinned():
    # Golden values: the problems seed 0 gives. If they change, every dataset drawn from a
    # seed silently changes. This test makes that loud.
    assert TASK.sample(rng(0), 2, difficulty=4) == [
        CountdownProblem(numbers=(85, 64, 51, 27), target=125, solution="85+64-(51-27)"),
        CountdownProblem(numbers=(50, 61, 97, 73), target=87, solution="73+(50-(97-61))"),
    ]


def test_same_seed_same_problems_and_more_only_adds_to_the_end():
    assert TASK.sample(rng(7), 20, 5) == TASK.sample(rng(7), 20, 5)
    assert TASK.sample(rng(7), 20, 5)[:10] == TASK.sample(rng(7), 10, 5)
    assert TASK.sample(rng(7), 20, 5) != TASK.sample(rng(8), 20, 5)


@pytest.mark.parametrize("difficulty", range(MIN_NUMBERS, MAX_NUMBERS + 1))
def test_every_generated_problem_is_solvable_and_fits(difficulty: int):
    for problem in TASK.sample(rng(difficulty), 200, difficulty):
        assert len(problem.numbers) == difficulty
        assert all(1 <= number <= LARGEST_NUMBER for number in problem.numbers)
        working = TASK.worked_solution(problem)
        *steps, _answer = working.split(";")
        assert len(steps) == difficulty - 1  # each step joins two numbers into one
        for step in steps:
            match = STEP.fullmatch(step)
            assert match is not None, step
            assert 1 <= int(match[3]) <= MAX_VALUE
        assert TASK.score(problem, working).value == 1.0
        assert len(TASK.format_prompt(problem)) + len(encode(working)) + 1 <= CONTEXT


@pytest.mark.parametrize("difficulty", [MIN_NUMBERS - 1, MAX_NUMBERS + 1])
def test_difficulty_out_of_range_is_refused(difficulty: int):
    with pytest.raises(ValueError, match="how many numbers"):
        TASK.sample(rng(0), 1, difficulty)


# --- scoring -------------------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["55+36-19-7", "55-19+36-7", "(55+36)-(19+7)", "36+55-7-19"])
def test_any_correct_answer_scores_full_marks(answer: str):
    assert score(answer) == Reward(value=1.0, format=0.0, parsed=answer, flags=())


def test_fractions_on_the_way_are_fine():
    problem = CountdownProblem(numbers=(6, 4, 2), target=3, solution="6/(4-2)")
    assert TASK.score(problem, "#6/4*2").value == 1.0  # 6/4 is 3/2 on the way: exact maths


@pytest.mark.parametrize(
    "answer",
    ["55+36-19", "55+36-19-7+7-7", "55+36-19-7+0", "65"],
    ids=["leaves-one-out", "uses-one-twice", "adds-a-number", "just-the-target"],
)
def test_answers_must_use_every_number_exactly_once(answer: str):
    wrong = Reward(value=0.0, format=0.0, parsed=answer, flags=("wrong_numbers",))
    assert score(answer) == wrong


def test_a_repeated_number_must_be_used_as_often_as_it_is_given():
    problem = CountdownProblem(numbers=(25, 3, 7, 3), target=157, solution="(25-3)*7+3")
    assert TASK.score(problem, "#(25-3)*7").flags == ("wrong_numbers",)  # only one of the 3s


@pytest.mark.parametrize("answer", ["55+36-19*7", "55-36-19-7", "(55+36)/(19+7)"])
def test_the_right_numbers_with_the_wrong_value_score_zero(answer: str):
    wrong = Reward(value=0.0, format=0.0, parsed=answer, flags=("wrong_answer",))
    assert score(answer) == wrong


def test_dividing_by_zero_is_a_wrong_answer_not_a_crash():
    problem = CountdownProblem(numbers=(2, 2, 6), target=24, solution="(2+2)*6")
    assert TASK.score(problem, "#6/(2-2)").flags == ("wrong_answer",)


@pytest.mark.parametrize(
    "answer",
    ["", "55+36-19-(7", "55 + 36 - 19 - 7", "__import__('os')", "(" * 100_000],
    ids=["empty", "unclosed-bracket", "spaces", "code", "huge"],
)
def test_malformed_answers_get_a_format_penalty(answer: str):
    malformed = Reward(value=0.0, format=-1.0, parsed=answer, flags=("malformed_answer",))
    assert score(answer) == malformed


def test_no_answer_mark_means_no_answer():
    assert TASK.score(PROBLEM, "55+36=91;91-19=72;").flags == ("no_answer",)


# --- the Task interface --------------------------------------------------------------------


def test_countdown_task_is_a_task():
    # Checked by pyright before the test runs, like ArithmeticTask's in test_arithmetic.py.
    task: Task[CountdownProblem] = CountdownTask()
    assert task.name == "countdown"
