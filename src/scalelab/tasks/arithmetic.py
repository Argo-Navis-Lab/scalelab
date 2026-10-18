"""Multi-digit arithmetic: add, subtract or multiply two whole numbers."""

import re
from typing import Literal

import numpy as np
from pydantic import Field

from scalelab.core.config import StrictModel
from scalelab.tasks.rewards import Reward, read_answer
from scalelab.tasks.tokenizer import BOS, encode

Op = Literal["+", "-", "*"]
OPS: tuple[Op, ...] = ("+", "-", "*")

MIN_DIGITS = 2  # the difficulty range: how many digits each of the two numbers has
MAX_DIGITS = 12

# A whole number written the way the worked solutions write it: plain digits 0-9, a minus
# sign only in front of a non-zero number, and no leading zeros, "+", spaces or "_".
_INTEGER = re.compile(r"0|-?[1-9][0-9]*")


class ArithmeticProblem(StrictModel):
    """Two whole numbers and an operation, such as 347 + 852."""

    a: int = Field(ge=0)
    op: Op
    b: int = Field(ge=0)

    @property
    def question(self) -> str:
        """The problem as the model reads it, such as "347+852=" (the prompt without <bos>)."""
        return f"{self.a}{self.op}{self.b}="

    @property
    def answer(self) -> int:
        """The correct result. Subtraction can give a negative number."""
        if self.op == "+":
            return self.a + self.b
        if self.op == "-":
            return self.a - self.b
        return self.a * self.b


class ArithmeticTask:
    """Arithmetic problems. The difficulty level is how many digits each number has."""

    name = "arithmetic"

    def __init__(self, ops: str = "+-*") -> None:
        """`ops` picks the operations to mix, such as "+-" for adding and subtracting only."""
        if not ops or set(ops) - set(OPS) or len(set(ops)) != len(ops):
            raise ValueError(f"ops must be different characters from {''.join(OPS)!r}, got {ops!r}")
        # Always in OPS order, so "+*" and "*+" give exactly the same problems.
        self.ops: tuple[Op, ...] = tuple(op for op in OPS if op in ops)

    def sample(self, rng: np.random.Generator, n: int, difficulty: int) -> list[ArithmeticProblem]:
        if not MIN_DIGITS <= difficulty <= MAX_DIGITS:
            raise ValueError(
                f"difficulty is the number of digits, from {MIN_DIGITS} to {MAX_DIGITS}; "
                f"got {difficulty}"
            )
        low, high = 10 ** (difficulty - 1), 10**difficulty  # e.g. 100 and 1000 for 3 digits
        problems: list[ArithmeticProblem] = []
        for _ in range(n):
            # One problem at a time, so asking for more problems only adds to the end:
            # the first 10 of 20 problems are the same 10 that asking for 10 would give.
            a, b = rng.integers(low, high, size=2)  # `high` itself is never drawn
            op = self.ops[int(rng.integers(len(self.ops)))]
            problems.append(ArithmeticProblem(a=int(a), op=op, b=int(b)))
        return problems

    def format_prompt(self, problem: ArithmeticProblem) -> list[int]:
        return [BOS, *encode(problem.question)]

    def worked_solution(self, problem: ArithmeticProblem) -> str:
        """Column by column from the right, the way it's done on paper. For 347+852= it is
        "7+2+0=9,9;4+5+0=9,99;3+8+0=11,199;#1199".

        Each step is a true equation: the column's digits, then the carry from the column
        before, "=", the value, ",", then the result so far. The value's last digit joins the
        front of the result, and the rest of the value is the next column's carry.
        """
        a, b, op = problem.a, problem.b, problem.op
        steps: list[str] = []
        negative = op == "-" and a < b
        if negative:
            steps.append("-")  # the answer is negative: work out b - a, then add the minus sign
            a, b = b, a
        xs, ys = _digits(a), _digits(b)
        columns = len(xs) + len(ys) - 1 if op == "*" else max(len(xs), len(ys))
        carry, result = 0, ""
        for k in range(columns):
            if op == "*":
                # Column k collects every pair of digits whose positions add up to k.
                pairs = [(xs[i], ys[k - i]) for i in range(len(xs)) if 0 <= k - i < len(ys)]
                terms = "+".join(f"{x}*{y}" for x, y in pairs)
                value = sum(x * y for x, y in pairs) + carry
            else:
                x, y = _digit(xs, k), _digit(ys, k)
                terms = f"{x}{op}{y}"
                value = (x + y if op == "+" else x - y) + carry
            result = str(value % 10) + result
            steps.append(f"{terms}{carry:+d}={value},{result}")
            carry = value // 10  # Python rounds down, so a value of -1 gives digit 9, carry -1
        answer = ((str(carry) if carry else "") + result).lstrip("0") or "0"
        steps.append(f"#{'-' if negative else ''}{answer}")
        return ";".join(steps)

    def score(self, problem: ArithmeticProblem, completion: str) -> Reward:
        """Full marks only for exactly the right whole number after the one "#". The answer is
        compared as text, never converted with int(): int() accepts " 1_199", "+1199" and
        other digit alphabets, and crashes on numbers longer than 4300 digits."""
        answer = read_answer(completion)
        if isinstance(answer, Reward):
            return answer  # no single answer to read
        if not _INTEGER.fullmatch(answer):
            return Reward(value=0.0, format=-1.0, parsed=answer, flags=("malformed_answer",))
        if answer != str(problem.answer):
            return Reward(value=0.0, format=0.0, parsed=answer, flags=("wrong_answer",))
        return Reward(value=1.0, format=0.0, parsed=answer, flags=())


def _digits(n: int) -> list[int]:
    """The digits of n, last digit first: 347 gives [7, 4, 3]."""
    return [int(digit) for digit in reversed(str(n))]


def _digit(digits: list[int], k: int) -> int:
    """Digit k counting from the right, or 0 past the front of a shorter number."""
    return digits[k] if k < len(digits) else 0
