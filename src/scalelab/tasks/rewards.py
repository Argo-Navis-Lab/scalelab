"""Rewards: how a completion scored, and reading its answer out safely."""

from typing import Literal

from pydantic import Field

from scalelab.core.config import StrictModel

ANSWER_MARK = "#"  # a completion ends with "#" and then its answer

# Why a completion lost reward.
Flag = Literal["no_answer", "several_answers", "malformed_answer", "wrong_answer"]


class Reward(StrictModel):
    """How one completion scored, and why.

    `value` is what RL maximizes by default: 1 for a correct answer, 0 otherwise. `format` is
    kept apart: -1 when the answer couldn't be read at all, 0 otherwise. An experiment can add
    a small multiple of it to `value`, or leave it out, and measure the difference.
    """

    value: float = Field(ge=0.0, le=1.0)
    format: float = Field(ge=-1.0, le=0.0)
    parsed: str | None  # the answer as read after "#", or None when there was none to read
    flags: tuple[Flag, ...]  # what went wrong; empty for a correct answer


def read_answer(completion: str) -> str | Reward:
    """The text after the "#" in a completion. Without exactly one "#", there is no answer to
    read, and this returns the zero Reward that says why: a model may not offer several answers
    and let the verifier pick the right one."""
    marks = completion.count(ANSWER_MARK)
    if marks != 1:
        flag: Flag = "no_answer" if marks == 0 else "several_answers"
        return Reward(value=0.0, format=-1.0, parsed=None, flags=(flag,))
    return completion.partition(ANSWER_MARK)[2]
