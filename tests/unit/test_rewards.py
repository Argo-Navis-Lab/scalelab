"""Verifier and reward tests: reading the answer safely, and scoring arithmetic completions."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from scalelab.tasks.arithmetic import ArithmeticProblem, ArithmeticTask, Op
from scalelab.tasks.rewards import Flag, Reward, read_answer

TASK = ArithmeticTask()
PROBLEM = ArithmeticProblem(a=347, op="+", b=852)  # the answer is 1199
WORKING = "7+2+0=9,9;4+5+0=9,99;3+8+0=11,199;"  # its worked solution, up to the answer


def score(completion: str) -> Reward:
    return TASK.score(PROBLEM, completion)


# --- reading the answer --------------------------------------------------------------------


@pytest.mark.parametrize(("completion", "answer"), [("1+1=2;#2", "2"), ("#", ""), ("#x y", "x y")])
def test_the_answer_is_the_text_after_the_one_mark(completion: str, answer: str):
    assert read_answer(completion) == answer


@pytest.mark.parametrize(
    ("completion", "flag"),
    [
        ("", "no_answer"),
        (WORKING, "no_answer"),
        ("#1199#1199", "several_answers"),
        ("#1;#1199", "several_answers"),
    ],
    ids=["empty", "stopped-before-the-answer", "same-answer-twice", "two-answers"],
)
def test_without_exactly_one_mark_there_is_no_answer(completion: str, flag: Flag):
    unreadable = Reward(value=0.0, format=-1.0, parsed=None, flags=(flag,))
    assert read_answer(completion) == unreadable
    assert score(completion) == unreadable


# --- golden correct and incorrect answers --------------------------------------------------


def test_the_worked_solution_scores_full_marks():
    assert score(WORKING + "#1199") == Reward(value=1.0, format=0.0, parsed="1199", flags=())


def test_only_the_answer_is_scored():
    assert score("#1199").value == 1.0  # RL may find shorter working, and that's allowed


def test_negative_answers():
    problem = ArithmeticProblem(a=85, op="-", b=347)
    assert TASK.score(problem, "#-262").value == 1.0
    assert TASK.score(problem, "#262").flags == ("wrong_answer",)


@pytest.mark.parametrize("answer", ["1198", "-1199", "0", "11990"])
def test_a_wrong_number_scores_zero_without_a_format_penalty(answer: str):
    wrong = Reward(value=0.0, format=0.0, parsed=answer, flags=("wrong_answer",))
    assert score(WORKING + "#" + answer) == wrong


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "01199",
        "+1199",
        "-0",
        " 1199",
        "1199 ",
        "1199\n",
        "1_199",
        "1,199",
        "11.99",
        "١١٩٩",
        "1199;",
        "1199<eos>",
    ],
    ids=[
        "empty",
        "leading-zero",
        "plus-sign",
        "minus-zero",
        "space-before",
        "space-after",
        "newline",
        "underscore",
        "comma",
        "decimal-point",
        "arabic-digits",
        "trailing-semicolon",
        "eos-as-text",
    ],
)
def test_anything_but_a_plain_whole_number_is_malformed(answer: str):
    malformed = Reward(value=0.0, format=-1.0, parsed=answer, flags=("malformed_answer",))
    assert score("#" + answer) == malformed


# --- safety --------------------------------------------------------------------------------


def test_malicious_text_is_never_run(tmp_path: Path):
    marker = tmp_path / "hacked"
    reward = score(f"#__import__('pathlib').Path({str(marker)!r}).touch()")
    assert reward.flags == ("malformed_answer",)
    assert not marker.exists()  # the "answer" was only ever compared as text


def test_a_huge_answer_is_compared_without_converting_it():
    # int() refuses numbers over 4300 digits with an error; comparing text doesn't care.
    assert score("#" + "9" * 1_000_000).flags == ("wrong_answer",)


# --- every problem -------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["+", "-", "*"])
def test_every_worked_solution_is_correct_and_a_changed_last_digit_is_not(op: Op):
    task = ArithmeticTask(op)
    for problem in task.sample(np.random.default_rng(0), 100, 4):
        working = task.worked_solution(problem)
        assert task.score(problem, working).value == 1.0
        changed = working[:-1] + str((int(working[-1]) + 1) % 10)
        assert task.score(problem, changed).flags == ("wrong_answer",)


# --- the Reward record ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"value": 2.0, "format": 0.0, "parsed": None, "flags": ()},
        {"value": 1.0, "format": 0.5, "parsed": None, "flags": ()},
        {"value": 0.0, "format": 0.0, "parsed": None, "flags": ("lucky_guess",)},
    ],
    ids=["value-above-1", "positive-format", "unknown-flag"],
)
def test_impossible_rewards_are_refused(fields: dict[str, object]):
    with pytest.raises(ValidationError):
        Reward.model_validate(fields)
