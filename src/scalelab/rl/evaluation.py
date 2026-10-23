"""Grading a model: greedy pass@1, and sampled pass@k from several answers to each problem."""

import math
from collections import defaultdict
from collections.abc import Callable, Sequence

import torch
from pydantic import Field
from torch import Tensor

from scalelab.core.config import SamplingParams, StrictModel
from scalelab.models.transformer import Transformer
from scalelab.rl.rollout import generate
from scalelab.rl.sampling import NEVER_WRITTEN
from scalelab.rl.scoring import completion_text, score_rollouts
from scalelab.tasks.base import Task
from scalelab.tasks.tokenizer import EOS


class EvalResult(StrictModel):
    """One model's greedy pass@1 on one eval set: which of its problems it solved. The
    outcomes, in the order of the problems, are the stored truth (m2s38); the counts come
    from them."""

    name: str  # the eval set, such as "arithmetic-3"
    solved: tuple[bool, ...] = Field(min_length=1)  # full marks or not, one per problem

    @property
    def n(self) -> int:
        return len(self.solved)

    @property
    def passed(self) -> int:
        return sum(self.solved)

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
    """Greedy pass@1: which problems the model's greedy answer gets full marks on."""
    completions = generate_greedy(model, [task.format_prompt(p) for p in problems])
    solved = [
        task.score(problem, completion_text(tokens)).value == 1.0
        for problem, tokens in zip(problems, completions, strict=True)
    ]
    return EvalResult(name=name, solved=tuple(solved))


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

    Each problem is answered once, in its part. The whole set's outcomes are the parts',
    each put back in its problem's own place: any two gradings of a set line up problem by
    problem (m2s38).
    """
    parts: dict[str, list[int]] = defaultdict(list)  # label -> where its problems are
    for index, problem in enumerate(problems):
        parts[key(problem)].append(index)
    solved = [False] * len(problems)
    results: list[EvalResult] = []
    for label in sorted(parts):
        indexes = parts[label]
        result = evaluate(model, task, [problems[i] for i in indexes], f"{name}/{label}")
        for index, outcome in zip(indexes, result.solved, strict=True):
            solved[index] = outcome  # back to the problem's place in the whole set
        results.append(result)
    return [EvalResult(name=name, solved=tuple(solved)), *results]


# --- sampled pass@k -------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased estimate of pass@k from n answers to one problem, c of them right: the
    chance that k answers picked at random from the n include a right one,
    1 - C(n - c, k) / C(n, k) (Chen et al., 2021). With k = 1 it is c / n."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"pass@k needs 0 <= c <= n and 1 <= k <= n, got n={n}, c={c}, k={k}")
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


class SampledResult(StrictModel):
    """Sampled answers to one eval set: correct[i] of the `samples` answers drawn for problem
    i were right. The counts are the stored truth; pass@k comes from them."""

    name: str  # the eval set, such as "arithmetic-3"
    samples: int = Field(gt=0)  # answers drawn for each problem
    correct: tuple[int, ...] = Field(min_length=1)  # one count per problem

    @property
    def n(self) -> int:
        return len(self.correct)

    def pass_at(self, k: int) -> float:
        """pass@k averaged over the problems, each problem's estimated by pass_at_k."""
        return sum(pass_at_k(self.samples, c, k) for c in self.correct) / self.n

    def stderr(self, k: int) -> float:
        """How much pass_at(k) wobbles: the spread of the problems' own estimates, divided by
        sqrt(n). With one sample per problem and k = 1, it is standard_error's binomial one."""
        values = [pass_at_k(self.samples, c, k) for c in self.correct]
        mean = sum(values) / self.n
        variance = sum((value - mean) ** 2 for value in values) / self.n
        return math.sqrt(variance / self.n)

    def by_part[P](self, problems: Sequence[P], key: Callable[[P], str]) -> list["SampledResult"]:
        """This result, then one per part of its problems, as evaluate_by splits greedy grades:
        `key` labels each problem, a part is named "<name>/<label>", and parts come in the
        sorted order of their labels. `problems` are the ones graded, in the same order."""
        if len(problems) != self.n:
            raise ValueError(f"{self.n} problems were graded, but {len(problems)} were given")
        parts: dict[str, list[int]] = defaultdict(list)  # label -> those problems' counts
        for problem, count in zip(problems, self.correct, strict=True):
            parts[key(problem)].append(count)
        results: list[SampledResult] = [self]
        for label in sorted(parts):
            counts = tuple(parts[label])
            results.append(
                SampledResult(name=f"{self.name}/{label}", samples=self.samples, correct=counts)
            )
        return results


def evaluate_sampled[P](
    model: Transformer,
    task: Task[P],
    problems: Sequence[P],
    name: str,
    generator: torch.Generator,
    samples: int = 4,
    temperature: float = 1.0,
) -> SampledResult:
    """Draw `samples` answers to each problem at `temperature`, and count each problem's
    right ones. Every draw comes from `generator` (on the model's device), so grading never
    moves the random state that training uses."""
    prompts = [task.format_prompt(problem) for problem in problems]
    batch = generate(model, prompts, samples, SamplingParams(temperature=temperature), generator)
    rewards = score_rollouts(task, problems, batch)
    correct = [0] * len(problems)
    for group, reward in zip(batch.group_ids.tolist(), rewards, strict=True):
        if reward.value == 1.0:
            correct[group] += 1
    return SampledResult(name=name, samples=samples, correct=tuple(correct))


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
