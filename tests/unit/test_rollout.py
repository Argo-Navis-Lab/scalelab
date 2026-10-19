"""Rollout tests: rows in groups behind their prompts, answers that end where they should, the
sampler's log-probs, and draws decided by the generator alone."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scalelab.core.config import ModelConfig, SamplingParams
from scalelab.models.transformer import Transformer
from scalelab.rl.evaluation import generate_greedy
from scalelab.rl.rollout import RolloutBatch, generate
from scalelab.rl.sampling import NEVER_WRITTEN
from scalelab.tasks.arithmetic import ArithmeticTask
from scalelab.tasks.tokenizer import EOS, PAD, encode

TASK = ArithmeticTask()
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1)
SHORT = TINY.model_copy(update={"context": 32})  # quick to fill up
PROMPTS = [encode("<bos>12+34="), encode("<bos>123*456="), encode("<bos>56-78=")]  # 7, 9, 7


def make(config: ModelConfig) -> Transformer:
    """The same starting weights every time, without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(config)


def roll(seed: int = 0, batch_size: int = 512) -> RolloutBatch:
    """Three answers to each of PROMPTS from an untrained model, at temperature 1."""
    generator = torch.Generator().manual_seed(seed)
    return generate(make(SHORT), PROMPTS, 3, SamplingParams(), generator, 5, batch_size)


@pytest.fixture(scope="module")
def memorized() -> tuple[Transformer, list[list[int]]]:
    """A tiny model trained to know four worked solutions by heart (as in test_evaluation),
    and the prompts of those four problems."""
    rng = np.random.default_rng(0)
    problems = [TASK.sample(rng, 1, difficulty)[0] for difficulty in (3, 2, 4, 2)]
    rows = [TASK.format_prompt(p) + encode(TASK.worked_solution(p)) + [EOS] for p in problems]
    width = max(len(row) for row in rows)
    tokens = torch.tensor([row + [PAD] * (width - len(row)) for row in rows])
    model = make(TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = F.cross_entropy(model(tokens[:, :-1]).flatten(0, 1), tokens[:, 1:].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, [TASK.format_prompt(p) for p in problems]


# --- the layout -----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [512, 2])  # 2: decoded a couple of rows at a time
def test_rows_come_in_groups_each_behind_its_prompt(batch_size: int):
    batch = roll(batch_size=batch_size)
    assert batch.group_ids.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert batch.prompt_lengths.tolist() == [7, 7, 7, 9, 9, 9, 7, 7, 7]
    for row, group in enumerate(batch.group_ids.tolist()):
        assert batch.tokens[row, : len(PROMPTS[group])].tolist() == PROMPTS[group]
    assert batch.policy_version == 5


def test_each_answer_ends_at_its_eos_or_where_the_context_runs_out():
    # 60 rows, about a third of which fill the context: both endings come up on any machine,
    # however its rounding steers the draws (9 rows could all happen to end one way).
    batch = generate(make(SHORT), PROMPTS, 20, SamplingParams(), torch.Generator().manual_seed(0))
    assert set(batch.finish_reasons) == {"eos", "length"}


def test_answers_only_use_tokens_a_model_may_write():
    for answer in roll().answers():
        assert not set(answer) & set(NEVER_WRITTEN)


def test_the_sampler_log_probs_cover_exactly_the_answers():
    batch = roll()
    assert torch.equal(batch.sampler_logprobs < 0, batch.response_mask)  # log p < 0 when drawn
    assert torch.all(batch.sampler_logprobs[~batch.response_mask] == 0)  # 0 everywhere else


# --- the randomness -------------------------------------------------------------------------


def test_the_generator_decides_the_rollouts():
    first, again, other = roll(seed=0), roll(seed=0), roll(seed=1)
    assert torch.equal(first.tokens, again.tokens)
    assert torch.equal(first.sampler_logprobs, again.sampler_logprobs)
    assert not torch.equal(first.tokens, other.tokens)


def test_the_answers_to_one_prompt_differ():
    answers = roll().answers()
    assert len({tuple(answer) for answer in answers[:3]}) > 1  # the 3 answers to PROMPTS[0]


def test_at_a_low_temperature_rollouts_are_greedy(
    memorized: tuple[Transformer, list[list[int]]],
):
    # Only prompts the model knows by heart: there the likeliest token leads the next by at
    # least 6.29 logits, 126 at T = 0.05. On other prompts the top two can be within 0.5,
    # close enough for a machine's rounding to change which one a draw picks.
    model, prompts = memorized
    params = SamplingParams(temperature=0.05)  # 20 times the gaps between logits
    batch = generate(model, prompts, 2, params, torch.Generator().manual_seed(0))
    greedy = generate_greedy(model, prompts)
    assert batch.answers() == [answer for answer in greedy for _ in range(2)]
