"""Multi-digit arithmetic: add, subtract or multiply two whole numbers."""

from typing import Literal

import numpy as np
from pydantic import Field

from scalelab.core.config import StrictModel
from scalelab.tasks.tokenizer import BOS, encode

Op = Literal["+", "-", "*"]
OPS: tuple[Op, ...] = ("+", "-", "*")

MIN_DIGITS = 2  # the difficulty range: how many digits each of the two numbers has
MAX_DIGITS = 12


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
