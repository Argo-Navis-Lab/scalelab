"""Generation and evaluation tests: greedy answers, batching by length, pass rates."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scalelab.core.config import ModelConfig
from scalelab.models.transformer import Transformer
from scalelab.rl.evaluation import (
    NEVER_WRITTEN,
    EvalResult,
    evaluate,
    evaluate_by,
    generate_greedy,
    standard_error,
)
from scalelab.tasks.arithmetic import ArithmeticProblem, ArithmeticTask
from scalelab.tasks.countdown import CountdownProblem, CountdownTask
from scalelab.tasks.evalsets import read_eval_set
from scalelab.tasks.tokenizer import EOS, PAD, decode, encode

TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1)
TASK = ArithmeticTask()


def make() -> Transformer:
    """The same starting weights every time, without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(TINY)


@pytest.fixture(scope="module")
def memorized() -> tuple[Transformer, list[ArithmeticProblem]]:
    """A tiny model trained to know four worked solutions by heart. Their prompts have three
    different lengths (3, 2, 4 and 2 digits), mixed up on purpose."""
    rng = np.random.default_rng(0)
    problems = [TASK.sample(rng, 1, difficulty)[0] for difficulty in (3, 2, 4, 2)]
    rows = [TASK.format_prompt(p) + encode(TASK.worked_solution(p)) + [EOS] for p in problems]
    width = max(len(row) for row in rows)
    tokens = torch.tensor([row + [PAD] * (width - len(row)) for row in rows])
    model = make()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = F.cross_entropy(model(tokens[:, :-1]).flatten(0, 1), tokens[:, 1:].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, problems


# --- a model that knows its answers ---------------------------------------------------------


def test_greedy_generation_writes_the_memorized_worked_solutions(
    memorized: tuple[Transformer, list[ArithmeticProblem]],
):
    model, problems = memorized
    completions = generate_greedy(model, [TASK.format_prompt(p) for p in problems])
    # Each answer back in its own prompt's place, although they were generated in 3 groups.
    assert completions == [encode(TASK.worked_solution(p)) + [EOS] for p in problems]


def test_the_memorized_model_passes_all_four(
    memorized: tuple[Transformer, list[ArithmeticProblem]],
):
    model, problems = memorized
    result = evaluate(model, TASK, problems, "memorized")
    assert result == EvalResult(name="memorized", n=4, passed=4)
    assert result.pass_rate == 1.0
    assert result.stderr == 0.0


def test_the_batch_size_does_not_change_the_answers(
    memorized: tuple[Transformer, list[ArithmeticProblem]],
):
    model, problems = memorized
    prompts = [TASK.format_prompt(p) for p in problems]
    assert generate_greedy(model, prompts, batch_size=1) == generate_greedy(model, prompts)


# --- grades by part -------------------------------------------------------------------------


def test_grades_by_operation_add_up_to_the_whole_set(
    memorized: tuple[Transformer, list[ArithmeticProblem]],
):
    model, problems = memorized  # one "-", two "+" and one "*"
    results = evaluate_by(model, TASK, problems, "memorized", key=lambda p: p.op)
    assert results == [
        EvalResult(name="memorized", n=4, passed=4),  # the whole set first...
        EvalResult(name="memorized/*", n=1, passed=1),  # ...then each part, labels sorted
        EvalResult(name="memorized/+", n=2, passed=2),
        EvalResult(name="memorized/-", n=1, passed=1),
    ]
    assert results[0] == evaluate(model, TASK, problems, "memorized")


def test_each_part_is_graded_on_its_own_problems(
    memorized: tuple[Transformer, list[ArithmeticProblem]],
):
    model, problems = memorized
    unseen = read_eval_set("arithmetic-3", ArithmeticProblem)[:6]  # never trained on
    results = evaluate_by(
        model,
        TASK,
        unseen + problems,
        "mixed",
        key=lambda p: "known" if p in problems else "unseen",
    )
    assert results == [
        EvalResult(name="mixed", n=10, passed=4),
        EvalResult(name="mixed/known", n=4, passed=4),
        EvalResult(name="mixed/unseen", n=6, passed=0),
    ]


@pytest.mark.parametrize(("band", "multiplications"), [(2, 319), (3, 329), (4, 316), (5, 337)])
def test_a_third_of_each_arithmetic_band_is_multiplication(band: int, multiplications: int):
    # Why grades by operation matter: after m1's exit warm starts every model scored about 66%
    # on arithmetic-5, and 663 of its 1000 problems are + or -.
    problems = read_eval_set(f"arithmetic-{band}", ArithmeticProblem)
    assert sum(problem.op == "*" for problem in problems) == multiplications


# --- an untrained model: nonsense, but well-behaved nonsense --------------------------------


def test_an_untrained_model_only_writes_real_tokens_and_stops_in_time():
    problems = read_eval_set("countdown-3", CountdownProblem)[:20]
    prompts = [CountdownTask().format_prompt(p) for p in problems]
    assert len({len(prompt) for prompt in prompts}) > 1  # prompts of several lengths
    for prompt, completion in zip(prompts, generate_greedy(make(), prompts), strict=True):
        assert not set(completion) & set(NEVER_WRITTEN)
        assert EOS not in completion[:-1]  # <eos> can only be the last token
        assert len(prompt) + len(completion) <= TINY.context
        decode(completion)  # every token is a real one, so this never raises


def test_an_untrained_model_solves_nothing():
    problems = read_eval_set("arithmetic-2", ArithmeticProblem)[:100]
    assert evaluate(make(), TASK, problems, "arithmetic-2").passed == 0


# --- pass rates and their uncertainty -------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "n", "expected"),
    [(0.5, 1000, math.sqrt(0.25 / 1000)), (0.0, 1000, 0.0), (1.0, 4, 0.0), (0.25, 4, 0.2165)],
)
def test_standard_error(p: float, n: int, expected: float):
    assert standard_error(p, n) == pytest.approx(expected, abs=1e-4)


def test_an_eval_result_derives_its_rate_and_error_from_its_counts():
    result = EvalResult(name="arithmetic-3", n=1000, passed=250)
    assert result.pass_rate == 0.25
    assert result.stderr == pytest.approx(0.0137, abs=1e-4)  # sqrt(0.25 * 0.75 / 1000)
