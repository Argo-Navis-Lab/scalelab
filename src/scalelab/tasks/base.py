"""What every task offers: problems drawn at a chosen difficulty, and prompts for the model."""

from typing import Protocol

import numpy as np


class Task[P](Protocol):
    """A family of problems, such as arithmetic. P is the kind of problem the task makes.

    Any class with these members counts as a Task; it doesn't need to inherit from this one.
    """

    name: str

    def sample(self, rng: np.random.Generator, n: int, difficulty: int) -> list[P]:
        """Draw n problems at one difficulty level, a whole number whose meaning each task
        defines. The same generator state always gives the same problems."""
        ...

    def format_prompt(self, problem: P) -> list[int]:
        """The token ids the model reads before it starts to answer."""
        ...

    def worked_solution(self, problem: P) -> str:
        """What a perfect model writes after the prompt: its working, then "#" and the answer.
        Warm-start training teaches the model to write exactly this, followed by <eos>."""
        ...
