"""Greedy generation and pass@1 evaluation: the first time a model gets graded."""

import math
from collections import defaultdict
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from scalelab.core.config import StrictModel
from scalelab.models.transformer import Transformer
from scalelab.rl.sampling import NEVER_WRITTEN
from scalelab.rl.scoring import completion_text
from scalelab.tasks.base import Task
from scalelab.tasks.tokenizer import EOS


class EvalResult(StrictModel):
    """One model's greedy pass@1 on one eval set: how many of n problems it solved."""

    name: str  # the eval set, such as "arithmetic-3"
    n: int
    passed: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n

    @property
    def stderr(self) -> float:
        return standard_error(self.pass_rate, self.n)


def standard_error(p: float, n: int) -> float:
    """How much a pass rate p measured on n problems wobbles: sqrt(p * (1 - p) / n)."""
    return math.sqrt(p * (1 - p) / n)


@torch.no_grad()
def generate_greedy(
    model: Transformer, prompts: Sequence[Sequence[int]], batch_size: int = 128
) -> list[list[int]]:
    """What the model writes after each prompt, taking its most likely token every time,
    until it writes <eos> (kept at the end) or prompt and answer fill the context.

    Prompts of the same length are generated together, so no padding is needed. The
    results come back in the order of the prompts.
    """
    same_length: dict[int, list[int]] = defaultdict(list)  # length -> indexes of those prompts
    for index, prompt in enumerate(prompts):
        same_length[len(prompt)].append(index)
    results: list[list[int]] = [[] for _ in prompts]
    for indexes in same_length.values():
        for first in range(0, len(indexes), batch_size):
            batch = indexes[first : first + batch_size]
            completions = _generate_batch(model, [prompts[i] for i in batch])
            for index, completion in zip(batch, completions, strict=True):
                results[index] = completion  # back to the prompt's own place
    return results


def evaluate[P](model: Transformer, task: Task[P], problems: Sequence[P], name: str) -> EvalResult:
    """Greedy pass@1: how many problems the model's greedy answer gets full marks on."""
    completions = generate_greedy(model, [task.format_prompt(p) for p in problems])
    passed = 0
    for problem, tokens in zip(problems, completions, strict=True):
        if task.score(problem, completion_text(tokens)).value == 1.0:
            passed += 1
    return EvalResult(name=name, n=len(problems), passed=passed)


def evaluate_by[P](
    model: Transformer,
    task: Task[P],
    problems: Sequence[P],
    name: str,
    key: Callable[[P], str],
) -> list[EvalResult]:
    """Greedy pass@1 of a whole eval set, then of each part of it. `key` labels every problem,
    and the problems with the same label make one part, named "<name>/<label>". Split by
    operation, arithmetic-5 gives "arithmetic-5", then "arithmetic-5/*", "arithmetic-5/+" and
    "arithmetic-5/-" (parts in the sorted order of their labels).

    Each problem is answered once, in its part; the whole set's counts are the parts' counts
    added up.
    """
    parts: dict[str, list[P]] = defaultdict(list)  # label -> the problems it labels
    for problem in problems:
        parts[key(problem)].append(problem)
    results = [evaluate(model, task, parts[label], f"{name}/{label}") for label in sorted(parts)]
    n = sum(result.n for result in results)
    passed = sum(result.passed for result in results)
    return [EvalResult(name=name, n=n, passed=passed), *results]


def _generate_batch(model: Transformer, prompts: list[Sequence[int]]) -> list[list[int]]:
    """Greedy generation for prompts that all have the same length."""
    device = model.embed.weight.device
    cache = model.new_cache(len(prompts))
    logits = model.decode(torch.tensor(prompts, device=device), cache)[:, -1]  # read the prompts
    written: list[Tensor] = []
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    while True:
        logits[:, NEVER_WRITTEN] = float("-inf")  # make those tokens impossible to pick
        next_tokens = logits.argmax(dim=-1)  # each sequence's most likely next token
        written.append(next_tokens)
        finished |= next_tokens == EOS
        if finished.all() or cache.length + 1 == cache.max_len:  # all done, or context full
            break
        logits = model.decode(next_tokens[:, None], cache)[:, -1]
    rows: list[list[int]] = torch.stack(written, dim=1).tolist()
    return [row[: row.index(EOS) + 1] if EOS in row else row for row in rows]
