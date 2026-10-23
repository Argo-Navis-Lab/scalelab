"""Generation and evaluation tests: greedy answers, batching by length, pass rates, and
sampled pass@k with its unbiased estimator."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scalelab.core.config import ModelConfig
from scalelab.models.transformer import Transformer
from scalelab.rl.evaluation import (
    EvalResult,
    SampledResult,
    evaluate,
    evaluate_by,
    evaluate_sampled,
    generate_greedy,
    pass_at_k,
    standard_error,
)
from scalelab.rl.sampling import NEVER_WRITTEN
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
    assert result == EvalResult(name="memorized", solved=(True,) * 4)
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
        EvalResult(name="memorized", solved=(True,) * 4),  # the whole set first...
        EvalResult(name="memorized/*", solved=(True,)),  # ...then each part, labels sorted
        EvalResult(name="memorized/+", solved=(True, True)),
        EvalResult(name="memorized/-", solved=(True,)),
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
        EvalResult(name="mixed", solved=(False,) * 6 + (True,) * 4),  # the set's own order
        EvalResult(name="mixed/known", solved=(True,) * 4),
        EvalResult(name="mixed/unseen", solved=(False,) * 6),
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


def test_an_eval_result_derives_its_counts_rate_and_error_from_its_outcomes():
    result = EvalResult(name="arithmetic-3", solved=(True,) * 250 + (False,) * 750)
    assert (result.n, result.passed) == (1000, 250)
    assert result.pass_rate == 0.25
    assert result.stderr == pytest.approx(0.0137, abs=1e-4)  # sqrt(0.25 * 0.75 / 1000)


# --- sampled pass@k (added in m2s20) --------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (4, 0, 1, 0.0),
        (4, 1, 1, 0.25),  # with k = 1: simply c / n
        (4, 3, 1, 0.75),
        (4, 4, 1, 1.0),
        (4, 1, 2, 0.5),  # 1 - C(3, 2) / C(4, 2) = 1 - 3/6
        (4, 2, 2, 5 / 6),  # 1 - C(2, 2) / C(4, 2) = 1 - 1/6
        (4, 1, 4, 1.0),  # all four picked: the right one is among them
        (4, 0, 4, 0.0),
        (10, 3, 5, 11 / 12),  # 1 - C(7, 5) / C(10, 5) = 1 - 21/252
    ],
)
def test_pass_at_k_by_hand(n: int, c: int, k: int, expected: float):
    assert pass_at_k(n, c, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [(4, 5, 1), (4, -1, 1), (4, 1, 0), (4, 1, 5)],
    ids=["more-right-than-drawn", "negative", "k-zero", "k-above-n"],
)
def test_impossible_counts_are_refused(n: int, c: int, k: int):
    with pytest.raises(ValueError, match="pass@k needs"):
        pass_at_k(n, c, k)


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_the_estimator_is_unbiased(p: float, k: int):
    # When each answer is right with chance p, the number right of 4 is binomial. Averaged
    # over every possible count, the estimate is exactly the true pass@k, 1 - (1 - p)^k.
    expected = sum(
        math.comb(4, c) * p**c * (1 - p) ** (4 - c) * pass_at_k(4, c, k) for c in range(5)
    )
    assert expected == pytest.approx(1 - (1 - p) ** k)


def test_a_sampled_result_derives_pass_at_k_from_its_counts():
    result = SampledResult(name="arithmetic-3", samples=4, correct=(4, 0, 2, 1))
    assert result.n == 4
    assert result.pass_at(1) == pytest.approx(7 / 16)  # 7 right answers of 16
    assert result.pass_at(4) == pytest.approx(3 / 4)  # 3 problems with at least one right


def test_with_one_sample_it_is_greedy_style_counting():
    sampled = SampledResult(name="x", samples=1, correct=(1,) * 250 + (0,) * 750)
    greedy = EvalResult(name="x", solved=(True,) * 250 + (False,) * 750)
    assert sampled.pass_at(1) == pytest.approx(greedy.pass_rate)
    assert sampled.stderr(1) == pytest.approx(greedy.stderr)


def test_near_zero_temperature_sampling_is_greedy(
    memorized: tuple[Transformer, list[ArithmeticProblem]],
):
    model, problems = memorized  # its top tokens lead by 6.29 logits at least: x 20 at T 0.05
    generator = torch.Generator().manual_seed(0)
    result = evaluate_sampled(model, TASK, problems, "memorized", generator, temperature=0.05)
    assert result == SampledResult(name="memorized", samples=4, correct=(4, 4, 4, 4))


def test_sampled_grading_draws_only_from_its_own_generator(
    memorized: tuple[Transformer, list[ArithmeticProblem]],
):
    model, problems = memorized
    before = torch.get_rng_state()
    first = evaluate_sampled(model, TASK, problems, "m", torch.Generator().manual_seed(1))
    again = evaluate_sampled(model, TASK, problems, "m", torch.Generator().manual_seed(1))
    assert first == again  # the same generator state draws the same answers
    assert torch.equal(torch.get_rng_state(), before)  # training's random state is untouched
    assert (first.n, first.samples) == (4, 4)
    assert all(0 <= c <= 4 for c in first.correct)


# --- sampled grades by part (added in m2s21) ------------------------------------------------


def test_sampled_grades_split_by_part_like_greedy_ones():
    problems = read_eval_set("arithmetic-2", ArithmeticProblem)[:10]  # 3 "*", 5 "+", 2 "-"
    counts = tuple(i % 5 for i in range(10))  # made-up counts: 0, 1, 2, 3, 4, 0, 1, ...
    whole = SampledResult(name="arithmetic-2", samples=4, correct=counts)
    parts = whole.by_part(problems, key=lambda p: p.op)
    assert [p.name for p in parts] == [
        "arithmetic-2",
        "arithmetic-2/*",
        "arithmetic-2/+",
        "arithmetic-2/-",
    ]
    assert parts[0] == whole
    for part, op in zip(parts[1:], "*+-", strict=True):  # each problem keeps its own count
        assert part.correct == tuple(c for p, c in zip(problems, counts, strict=True) if p.op == op)
    assert sum(sum(part.correct) for part in parts[1:]) == sum(counts)


def test_splitting_needs_the_problems_that_were_graded():
    whole = SampledResult(name="x", samples=4, correct=(1, 2, 3))
    problems = read_eval_set("arithmetic-2", ArithmeticProblem)[:2]
    with pytest.raises(ValueError, match="3 problems were graded, but 2 were given"):
        whole.by_part(problems, key=lambda p: p.op)
