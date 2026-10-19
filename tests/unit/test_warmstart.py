"""Warm-start tests: examples, batches whose targets skip the prompt, the loss, and sampling
that never trains on an eval question."""

import math

import numpy as np
import pytest
import torch

from scalelab.core.config import ModelConfig
from scalelab.models.transformer import Transformer
from scalelab.rl.evaluation import generate_greedy
from scalelab.rl.warmstart import (
    IGNORE,
    Example,
    eval_prompts,
    make_batch,
    sample_examples,
    warm_start_loss,
    worked_example,
)
from scalelab.tasks.arithmetic import ArithmeticProblem, ArithmeticTask
from scalelab.tasks.countdown import CountdownProblem, CountdownTask
from scalelab.tasks.tokenizer import EOS, PAD, VOCAB_SIZE, decode, encode

TASK = ArithmeticTask()
M1 = ModelConfig(name="m1", n_layers=4, d_model=128, n_heads=2)
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1)


def example(question: str, solution: str) -> Example:
    return Example(encode(f"<bos>{question}"), encode(solution) + [EOS])


def two_examples() -> list[Example]:
    return [example("1+2=", "#3"), example("10+20=", "#30")]


def make(config: ModelConfig) -> Transformer:
    """The same starting weights every time, without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(config)


def random_logits(shape: torch.Size) -> torch.Tensor:
    return torch.randn(*shape, VOCAB_SIZE, generator=torch.Generator().manual_seed(0))


# --- examples -------------------------------------------------------------------------------


def test_an_arithmetic_example_by_hand():
    problem = ArithmeticProblem(a=347, op="+", b=852)
    assert worked_example(TASK, problem) == Example(
        prompt=encode("<bos>347+852="),
        answer=encode("7+2+0=9,9;4+5+0=9,99;3+8+0=11,199;#1199") + [EOS],
    )


def test_a_countdown_example_by_hand():
    problem = CountdownProblem(numbers=(19, 36, 55, 7), target=65, solution="55+36-19-7")
    assert worked_example(CountdownTask(), problem) == Example(
        prompt=encode("<bos>19,36,55,7:65="),
        answer=encode("55+36=91;91-19=72;72-7=65;#55+36-19-7") + [EOS],
    )


# --- batches --------------------------------------------------------------------------------


def test_the_model_reads_all_but_the_last_token_and_learns_only_the_answer():
    # <bos> 1 + 2 = # 3 <eos>: 8 tokens, so 7 positions. The model reads <bos> to # and 3,
    # and from "=" on it learns the next token: "#", then "3", then <eos>.
    batch = make_batch([example("1+2=", "#3")])
    assert batch.inputs.tolist() == [encode("<bos>1+2=#3")]
    assert batch.targets.tolist() == [[IGNORE] * 4 + encode("#3") + [EOS]]


def test_shorter_sequences_are_padded_and_the_padding_is_never_trained():
    batch = make_batch(two_examples())
    assert batch.inputs.tolist() == [
        encode("<bos>1+2=#3") + [PAD] * 3,
        encode("<bos>10+20=#30"),
    ]
    assert batch.targets.tolist() == [
        [IGNORE] * 4 + encode("#3") + [EOS] + [IGNORE] * 3,
        [IGNORE] * 6 + encode("#30") + [EOS],
    ]


def test_tokens_count_the_positions_read_and_the_positions_trained():
    batch = make_batch(two_examples())
    assert (batch.tokens, batch.trained_tokens) == (7 + 10, 3 + 4)


def test_padding_changes_nothing_before_it():
    # Attention is causal: a position never sees what comes after it, so the padding that
    # follows a short sequence can't change that sequence's logits.
    model = make(TINY)
    with torch.no_grad():
        alone = model(make_batch(two_examples()[:1]).inputs)[0]
        padded = model(make_batch(two_examples()).inputs)[0, :7]
    torch.testing.assert_close(padded, alone)


# --- the loss -------------------------------------------------------------------------------


def test_the_loss_is_the_average_surprise_at_the_trained_targets():
    batch = make_batch(two_examples())
    logits = random_logits(batch.inputs.shape)
    log_probs = logits.log_softmax(dim=-1)
    surprises = [
        -log_probs[i, t, target]
        for i, row in enumerate(batch.targets.tolist())
        for t, target in enumerate(row)
        if target != IGNORE
    ]
    assert len(surprises) == batch.trained_tokens == 7
    expected = torch.stack(surprises).mean()
    torch.testing.assert_close(warm_start_loss(logits, batch.targets), expected)


def test_only_the_answer_positions_get_a_gradient():
    # In the prompt and the padding, whatever the model predicts leaves the loss unchanged,
    # so training never pushes it there.
    batch = make_batch(two_examples())
    logits = random_logits(batch.inputs.shape).requires_grad_()
    warm_start_loss(logits, batch.targets).backward()
    assert logits.grad is not None
    assert (logits.grad.abs().sum(dim=-1) > 0).tolist() == [
        [False] * 4 + [True] * 3 + [False] * 3,  # <bos>1+2=#3: from "=" on, 3 to learn
        [False] * 6 + [True] * 4,  # <bos>10+20=#30: from "=" on, 4 to learn
    ]


def test_an_untrained_model_guesses_evenly_among_all_64_tokens():
    # GPT-2's small starting weights make every logit close to 0, so every token gets about
    # 1/64 of the probability: a loss of ln(64) = 4.16.
    batch = make_batch(sample_examples(TASK, np.random.default_rng(0), 8, [2, 3], 256))
    with torch.no_grad():
        loss = warm_start_loss(make(M1)(batch.inputs), batch.targets)
    assert abs(loss.item() - math.log(64)) < 0.05


def test_a_tiny_model_learns_to_write_the_worked_solutions():
    examples = sample_examples(TASK, np.random.default_rng(0), 4, [2, 3], 256)
    batch = make_batch(examples)
    model = make(TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = warm_start_loss(model(batch.inputs), batch.targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    completions = generate_greedy(model, [e.prompt for e in examples])
    assert completions == [e.answer for e in examples]


# --- sampling -------------------------------------------------------------------------------


def test_sampling_is_pinned():
    # How warm-start data is drawn: a difficulty, then one problem. Changing that changes
    # every warm start's data for a given seed.
    examples = sample_examples(TASK, np.random.default_rng(0), 3, [2, 3, 4], 256)
    assert [decode(e.prompt) for e in examples] == [
        "<bos>6732+5600=",
        "<bos>13+16=",
        "<bos>83*68=",
    ]


def test_difficulties_come_from_the_whole_band():
    examples = sample_examples(TASK, np.random.default_rng(0), 300, [2, 4], 256)
    # An arithmetic prompt with d-digit numbers has 2d + 3 tokens: <bos>, a, op, b and "=".
    assert {(len(e.prompt) - 3) // 2 for e in examples} == {2, 4}


def test_eval_prompts_cover_every_band():
    assert len(eval_prompts(TASK, ArithmeticProblem)) == 4 * 1000
    assert len(eval_prompts(CountdownTask(), CountdownProblem)) == 4 * 1000


def test_an_eval_question_is_never_trained_on():
    excluded = eval_prompts(TASK, ArithmeticProblem)

    def eval_questions(examples: list[Example]) -> int:
        return sum(tuple(e.prompt) in excluded for e in examples)

    # About 1 in 24 two-digit problems is an eval question (1000 of 24,300)...
    assert eval_questions(sample_examples(TASK, np.random.default_rng(0), 3000, [2], 256)) > 50
    # ...and none of them is once they're excluded.
    examples = sample_examples(TASK, np.random.default_rng(0), 3000, [2], 256, excluded)
    assert eval_questions(examples) == 0


def test_a_worked_solution_longer_than_the_context_is_refused():
    # The shortest 6-digit multiplication takes 293 tokens: the model could never write it all.
    with pytest.raises(ValueError, match="the context holds 256"):
        sample_examples(ArithmeticTask("*"), np.random.default_rng(0), 1, [6], 256)
