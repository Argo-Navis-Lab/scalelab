"""Countdown: combine the given numbers with + - * / to reach a target, using each exactly once."""

import numpy as np
from pydantic import model_validator

from scalelab.core.config import StrictModel
from scalelab.tasks.expressions import Expr, ExpressionError, evaluate, numbers, parse, show
from scalelab.tasks.rewards import Reward, read_answer
from scalelab.tasks.tokenizer import BOS, encode

MIN_NUMBERS = 3  # the difficulty range: how many numbers a problem gives
MAX_NUMBERS = 6
LARGEST_NUMBER = 99  # the given numbers are drawn from 1 to 99
MAX_VALUE = 999  # every result along the way of a generated solution stays from 1 to 999


class CountdownProblem(StrictModel):
    """Numbers to combine and a target to reach, with one known solution, such as
    19, 36, 55 and 7 to make 65 with "55+36-19-7"."""

    numbers: tuple[int, ...]
    target: int
    solution: str

    @model_validator(mode="after")
    def check_solution(self) -> "CountdownProblem":
        tree = parse(self.solution)  # a malformed solution raises ExpressionError
        if sorted(numbers(tree)) != sorted(self.numbers):
            raise ValueError("the solution must use every number exactly once")
        if _work_out(tree, []) != self.target:
            raise ValueError("the solution must reach the target")
        return self

    @property
    def question(self) -> str:
        """The problem as the model reads it, such as "19,36,55,7:65=" (the prompt without
        <bos>): the numbers, ":", then the target."""
        return ",".join(str(number) for number in self.numbers) + f":{self.target}="


class CountdownTask:
    """Countdown problems. The difficulty level is how many numbers there are to combine."""

    name = "countdown"

    def sample(self, rng: np.random.Generator, n: int, difficulty: int) -> list[CountdownProblem]:
        if not MIN_NUMBERS <= difficulty <= MAX_NUMBERS:
            raise ValueError(
                f"difficulty is how many numbers, from {MIN_NUMBERS} to {MAX_NUMBERS}; "
                f"got {difficulty}"
            )
        return [_one_problem(rng, difficulty) for _ in range(n)]

    def format_prompt(self, problem: CountdownProblem) -> list[int]:
        return [BOS, *encode(problem.question)]

    def worked_solution(self, problem: CountdownProblem) -> str:
        """One operation per step, in the order they're worked out, then "#" and the
        expression. For 19,36,55,7:65= it is "55+36=91;91-19=72;72-7=65;#55+36-19-7"."""
        steps: list[str] = []
        _work_out(parse(problem.solution), steps)
        return ";".join([*steps, f"#{problem.solution}"])

    def score(self, problem: CountdownProblem, completion: str) -> Reward:
        """Full marks for any expression after the one "#" that uses every given number
        exactly once and works out to exactly the target. It's read by our own parser and
        worked out with exact fractions: never run as code."""
        answer = read_answer(completion)
        if isinstance(answer, Reward):
            return answer  # no single answer to read
        try:
            tree = parse(answer)
        except ExpressionError:
            return Reward(value=0.0, format=-1.0, parsed=answer, flags=("malformed_answer",))
        # Sorted lists, not sets: a set would let the model use one number twice.
        if sorted(numbers(tree)) != sorted(problem.numbers):
            return Reward(value=0.0, format=0.0, parsed=answer, flags=("wrong_numbers",))
        if evaluate(tree) != problem.target:  # dividing by zero gives None, never the target
            return Reward(value=0.0, format=0.0, parsed=answer, flags=("wrong_answer",))
        return Reward(value=1.0, format=0.0, parsed=answer, flags=())


def _one_problem(rng: np.random.Generator, count: int) -> CountdownProblem:
    """Make a problem that surely has a solution, by building one: draw the numbers, then keep
    combining two of them until a single number, the target, is left."""
    drawn = [int(number) for number in rng.integers(1, LARGEST_NUMBER + 1, size=count)]
    pool: list[tuple[int, Expr]] = [(number, number) for number in drawn]  # (value, how it's made)
    while len(pool) > 1:
        a, left = pool.pop(int(rng.integers(len(pool))))
        b, right = pool.pop(int(rng.integers(len(pool))))
        if a < b:  # put the larger value first, so a - b and a / b can be whole and positive
            (a, left), (b, right) = (b, right), (a, left)
        options = _options(a, b)
        op = list(options)[int(rng.integers(len(options)))]
        pool.append((options[op], (op, left, right)))
    (target, tree) = pool[0]
    return CountdownProblem(numbers=tuple(drawn), target=target, solution=show(tree))


def _options(a: int, b: int) -> dict[str, int]:
    """Every operation allowed on a and b (with a >= b), and its result. A result must be a
    whole number from 1 to MAX_VALUE. There is always at least one option: a - b when a > b,
    and a / b = 1 when a == b."""
    results = {"+": a + b, "-": a - b, "*": a * b}
    if a % b == 0:
        results["/"] = a // b
    return {op: value for op, value in results.items() if 1 <= value <= MAX_VALUE}


def _work_out(tree: Expr, steps: list[str]) -> int:
    """Work out `tree` one operation at a time, appending each step (such as "91-19=72") to
    `steps`, and return the value. Every step must give a whole number."""
    if isinstance(tree, int):
        return tree
    op, left, right = tree
    a, b = _work_out(left, steps), _work_out(right, steps)
    value = evaluate((op, a, b))
    if value is None or value.denominator != 1:
        raise ValueError(f"{a}{op}{b} isn't a whole number")
    steps.append(f"{a}{op}{b}={value}")
    return int(value)
