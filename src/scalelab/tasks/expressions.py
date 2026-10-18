"""Arithmetic expressions such as "(25-3)*7": read safely, worked out exactly, written back.

The grammar, from the loosest-binding part to the tightest:

    expression = term, { ("+" | "-"), term }
    term       = factor, { ("*" | "/"), factor }
    factor     = number | "(", expression, ")"
    number     = "0" | a digit 1-9 followed by any digits 0-9

Anything else is an ExpressionError. Text is only ever read, never run as code.
"""

import re
from fractions import Fraction

# An expression is a whole number, or a tuple (operator, left side, right side).
type Expr = int | tuple[str, Expr, Expr]

# Longer text is refused before reading it, which keeps out absurdly deep brackets (Python
# would run out of stack) and absurdly long numbers (int() refuses over 4300 digits).
MAX_LENGTH = 100

_TOKEN = re.compile(r"0|[1-9][0-9]*|[-+*/()]")
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}  # * and / bind tighter than + and -


class ExpressionError(ValueError):
    """The text isn't an expression the grammar allows."""


def parse(text: str) -> Expr:
    """Read text such as "(25-3)*7" into a tree: ("*", ("-", 25, 3), 7)."""
    if len(text) > MAX_LENGTH:
        raise ExpressionError(f"longer than {MAX_LENGTH} characters")
    parser = _Parser(_tokenize(text))
    expr = parser.expression()
    if parser.pos < len(parser.tokens):
        raise ExpressionError(f"unexpected {parser.tokens[parser.pos]!r} after the expression")
    return expr


def evaluate(expr: Expr) -> Fraction | None:
    """The exact value, as a fraction so that 8/3 stays exactly 8/3. None if it divides by 0."""
    if isinstance(expr, int):
        return Fraction(expr)
    op, left, right = expr
    a, b = evaluate(left), evaluate(right)
    if a is None or b is None:
        return None
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return None if b == 0 else a / b
    raise ValueError(f"unknown operator {op!r}")


def numbers(expr: Expr) -> list[int]:
    """The numbers in an expression, from left to right."""
    if isinstance(expr, int):
        return [expr]
    _op, left, right = expr
    return numbers(left) + numbers(right)


def show(expr: Expr) -> str:
    """Write a tree back as text, with brackets exactly where parse() needs them to read the
    same tree back: parse(show(expr)) == expr."""
    if isinstance(expr, int):
        return str(expr)
    op, left, right = expr
    left_text, right_text = show(left), show(right)
    # A looser operation needs brackets inside a tighter one: (1+2)*3. On the right, an equally
    # tight one does too, because parse() reads left to right: 1-(2-3) and 1+(2+3).
    if not isinstance(left, int) and _PRECEDENCE[left[0]] < _PRECEDENCE[op]:
        left_text = f"({left_text})"
    if not isinstance(right, int) and _PRECEDENCE[right[0]] <= _PRECEDENCE[op]:
        right_text = f"({right_text})"
    return left_text + op + right_text


def _tokenize(text: str) -> list[str]:
    """Split text into numbers, operators and brackets: "(25-3)*7" -> ["(", "25", "-", ...]."""
    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise ExpressionError(f"unexpected {text[pos]!r} at position {pos}")
        tokens.append(match.group())
        pos = match.end()
    return tokens


class _Parser:
    """Reads tokens from left to right with one method per grammar rule. Each method calls the
    rule below it, which is why this is called a recursive-descent parser."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0  # the next token to read

    def next_is(self, *options: str) -> bool:
        return self.pos < len(self.tokens) and self.tokens[self.pos] in options

    def take(self) -> str:
        if self.pos == len(self.tokens):
            raise ExpressionError("the expression ends too early")
        self.pos += 1
        return self.tokens[self.pos - 1]

    def expression(self) -> Expr:
        expr = self.term()
        while self.next_is("+", "-"):
            op = self.take()
            expr = (op, expr, self.term())  # left to right: 8-3-2 is (8-3)-2
        return expr

    def term(self) -> Expr:
        expr = self.factor()
        while self.next_is("*", "/"):
            op = self.take()
            expr = (op, expr, self.factor())
        return expr

    def factor(self) -> Expr:
        token = self.take()
        if token == "(":
            expr = self.expression()
            if not self.next_is(")"):
                raise ExpressionError("a '(' is never closed")
            self.take()
            return expr
        if token.isdigit():  # the tokenizer only lets plain digits 0-9 through
            return int(token)
        raise ExpressionError(f"expected a number or '(', got {token!r}")
