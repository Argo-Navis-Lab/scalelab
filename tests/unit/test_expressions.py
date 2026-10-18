"""Expression tests: reading, working out and writing back expressions such as "(25-3)*7"."""

from fractions import Fraction

import numpy as np
import pytest

from scalelab.tasks.expressions import (
    MAX_LENGTH,
    Expr,
    ExpressionError,
    evaluate,
    numbers,
    parse,
    show,
)


def random_tree(rng: np.random.Generator, size: int) -> Expr:
    """A random expression tree with `size` numbers, of any shape, with any operators."""
    if size == 1:
        return int(rng.integers(100))
    left_size = int(rng.integers(1, size))
    op = "+-*/"[int(rng.integers(4))]
    return (op, random_tree(rng, left_size), random_tree(rng, size - left_size))


# --- reading -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "tree"),
    [
        ("7", 7),
        ("1+2", ("+", 1, 2)),
        ("1+2*3", ("+", 1, ("*", 2, 3))),
        ("(1+2)*3", ("*", ("+", 1, 2), 3)),
        ("8-3-2", ("-", ("-", 8, 3), 2)),
        ("8/(3-8/3)", ("/", 8, ("-", 3, ("/", 8, 3)))),
        ("((7))", 7),
    ],
    ids=[
        "number",
        "add",
        "times-first",
        "brackets-first",
        "left-to-right",
        "nested",
        "spare-brackets",
    ],
)
def test_parse(text: str, tree: Expr):
    assert parse(text) == tree


# --- working out ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [("8-3-2", 3), ("1+2*3", 7), ("(1+2)*3", 9), ("7/2", Fraction(7, 2)), ("8/(3-8/3)", 24)],
)
def test_evaluate_is_exact(text: str, value: Fraction):
    assert evaluate(parse(text)) == value


def test_floats_would_get_it_wrong():
    assert 8 / (3 - 8 / 3) != 24  # 23.99999999999999 with ordinary floats
    assert evaluate(parse("8/(3-8/3)")) == 24


def test_dividing_by_zero_has_no_value():
    assert evaluate(parse("5/(3-3)")) is None
    assert evaluate(parse("1+5/0")) is None  # the missing value carries through the rest


def test_numbers_are_listed_left_to_right():
    assert numbers(parse("(25-3)*7+3")) == [25, 3, 7, 3]


# --- writing back --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["7", "1+2*3", "(1+2)*3", "8-3-2", "8-(3-2)", "1+(2+3)", "8/(3-8/3)", "(25-3)*7+3"]
)
def test_show_writes_text_back_unchanged(text: str):
    assert show(parse(text)) == text


def test_show_drops_brackets_that_change_nothing():
    assert show(parse("((1+2))+3")) == "1+2+3"


def test_every_tree_survives_show_then_parse():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        tree = random_tree(rng, int(rng.integers(1, 7)))
        assert parse(show(tree)) == tree


# --- mistakes are caught -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "1+", "+1", "-1", "(1+2", "1+2)", "()", "1 + 2", "07", "1.5", "2^3", "2**3", "1(2)", "x"],
    ids=[
        "empty",
        "ends-early",
        "starts-with-plus",
        "negative-number",
        "unclosed-bracket",
        "extra-bracket",
        "empty-brackets",
        "spaces",
        "leading-zero",
        "decimal-point",
        "caret",
        "double-star",
        "number-before-bracket",
        "letter",
    ],
)
def test_text_outside_the_grammar_is_refused(text: str):
    with pytest.raises(ExpressionError):
        parse(text)


def test_other_digit_alphabets_are_refused():
    with pytest.raises(ExpressionError, match="unexpected '١'"):
        parse("١+٢")  # Arabic digits: Python's int() would accept these


def test_code_is_refused_at_its_first_character():
    with pytest.raises(ExpressionError, match="unexpected '_' at position 0"):
        parse("__import__('os').getcwd()")


@pytest.mark.parametrize(
    "text",
    ["(" * 100_000 + "1" + ")" * 100_000, "9" * 1_000_000],
    ids=["deep-brackets", "huge-number"],
)
def test_huge_text_is_refused_before_reading(text: str):
    with pytest.raises(ExpressionError, match="longer than"):
        parse(text)


def test_the_length_limit_is_exactly_max_length():
    longest = "1+" * 49 + "11"
    assert len(longest) == MAX_LENGTH
    assert evaluate(parse(longest)) == 60  # forty-nine 1s and an 11
    with pytest.raises(ExpressionError, match="longer than"):
        parse(longest + "1")


def test_deep_brackets_within_the_limit_are_fine():
    assert parse("(" * 49 + "7" + ")" * 49) == 7
