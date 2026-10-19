"""Warm-start data: worked solutions as training batches, and the loss that teaches them."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from scalelab.core.config import StrictModel
from scalelab.tasks.base import Task
from scalelab.tasks.evalsets import EVAL_SETS, read_eval_set
from scalelab.tasks.tokenizer import EOS, PAD, encode

# A target the loss skips. -100 is also what F.cross_entropy skips when told nothing else.
IGNORE = -100


@dataclass(frozen=True)
class Example:
    """One training sequence: the prompt, then the answer the model learns to write."""

    prompt: list[int]
    answer: list[int]  # the worked solution's tokens, then <eos>


@dataclass(frozen=True)
class Batch:
    """Training sequences side by side, padded at the end to one length.

    At each position t of row i, the model reads inputs[i, t] (and everything before it) and
    should predict targets[i, t], the token that comes next. Targets inside the prompt and in
    the padding are IGNORE: the model reads the prompt but isn't trained to write it.
    """

    inputs: Tensor  # (batch, seq) token ids, PAD after each sequence's end
    targets: Tensor  # (batch, seq) the next token ids, or IGNORE

    @property
    def tokens(self) -> int:
        """Positions the model reads, padding excluded: what the token budget counts."""
        return int((self.inputs != PAD).sum())

    @property
    def trained_tokens(self) -> int:
        """Positions with a target, the ones the loss averages over."""
        return int((self.targets != IGNORE).sum())


def worked_example[P](task: Task[P], problem: P) -> Example:
    """The problem's prompt, then its worked solution and <eos>."""
    return Example(task.format_prompt(problem), encode(task.worked_solution(problem)) + [EOS])


def make_batch(examples: Sequence[Example]) -> Batch:
    """Line the examples up as one batch. A sequence of n tokens gives n - 1 positions: every
    token but the last is read, and every token but the first is a target."""
    width = max(len(e.prompt) + len(e.answer) for e in examples) - 1
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for example in examples:
        read = (example.prompt + example.answer)[:-1]
        wanted = [IGNORE] * (len(example.prompt) - 1) + example.answer
        padding = width - len(read)
        inputs.append(read + [PAD] * padding)
        targets.append(wanted + [IGNORE] * padding)
    return Batch(torch.tensor(inputs), torch.tensor(targets))


def warm_start_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy averaged over the trained targets: on average, how surprised the model
    is by each token of the answers. -log(probability it gave the right token), so 0 means
    certain and right, and ln(64) = 4.16 means an even guess among all 64 tokens."""
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE)


def sample_examples[P](
    task: Task[P],
    rng: np.random.Generator,
    n: int,
    difficulties: Sequence[int],
    context: int,
    excluded: frozenset[tuple[int, ...]] = frozenset(),
) -> list[Example]:
    """n training examples, each at a difficulty drawn uniformly from `difficulties`.

    A problem whose prompt is in `excluded` (the eval sets' prompts) is skipped and another
    is drawn, so the model is never trained on a question it will be tested on.
    """
    examples: list[Example] = []
    while len(examples) < n:
        difficulty = difficulties[int(rng.integers(len(difficulties)))]
        (problem,) = task.sample(rng, 1, difficulty)
        example = worked_example(task, problem)
        if tuple(example.prompt) in excluded:
            continue
        length = len(example.prompt) + len(example.answer)
        if length > context:
            raise ValueError(
                f"a worked {task.name} solution at difficulty {difficulty} takes {length} "
                f"tokens with its prompt, but the context holds {context}"
            )
        examples.append(example)
    return examples


def eval_prompts[P: StrictModel](
    task: Task[P], problem_type: type[P]
) -> frozenset[tuple[int, ...]]:
    """Every prompt in the task's frozen eval sets, to keep out of training."""
    prompts: set[tuple[int, ...]] = set()
    for name in EVAL_SETS:
        if name.startswith(f"{task.name}-"):
            problems = read_eval_set(name, problem_type)
            prompts.update(tuple(task.format_prompt(problem)) for problem in problems)
    return frozenset(prompts)
