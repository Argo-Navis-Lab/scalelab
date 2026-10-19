"""Trainer log-prob tests: one forward pass agrees with what the sampler recorded (at the same
temperature only), nothing leaks outside the answers, and gradients flow."""

import pytest
import torch

from scalelab.core.config import ModelConfig, SamplingParams
from scalelab.models.transformer import Transformer
from scalelab.rl.rollout import RolloutBatch, generate
from scalelab.rl.trainer import answer_logprobs
from scalelab.tasks.tokenizer import encode

SHORT = ModelConfig(name="short", n_layers=2, d_model=64, n_heads=1, context=32)
PROMPTS = [encode("<bos>12+34="), encode("<bos>123*456="), encode("<bos>56-78=")]  # 7, 9, 7


def make() -> Transformer:
    """The same starting weights every time, without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(SHORT)


def roll(model: Transformer, temperature: float) -> RolloutBatch:
    """Three answers to each of PROMPTS (9 rows; some end at <eos>, some fill the context)."""
    params = SamplingParams(temperature=temperature)
    return generate(model, PROMPTS, 3, params, torch.Generator().manual_seed(0))


def one_row(batch: RolloutBatch, row: int) -> RolloutBatch:
    """Row `row` of a batch on its own, without the padding the other rows needed."""
    end = int(batch.lengths[row])
    return RolloutBatch(
        tokens=batch.tokens[row : row + 1, :end],
        prompt_lengths=batch.prompt_lengths[row : row + 1],
        lengths=batch.lengths[row : row + 1],
        finish_reasons=batch.finish_reasons[row : row + 1],
        sampler_logprobs=batch.sampler_logprobs[row : row + 1, :end],
        group_ids=batch.group_ids[row : row + 1],
        policy_version=batch.policy_version,
    )


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_the_trainer_agrees_with_the_sampler(temperature: float):
    # The sampling test of the blueprint's Section 11, end to end: the trainer's forward pass
    # and the sampler's cached decoding give the same log-probs, up to float rounding.
    model = make()
    batch = roll(model, temperature)
    with torch.no_grad():
        trainer = answer_logprobs(model, batch, temperature)
    torch.testing.assert_close(trainer, batch.sampler_logprobs)


def test_a_temperature_left_out_shows_up_at_once():
    # Section 7.4's "temperature mismatch" bug: sampled at 0.7, recomputed at 1.
    model = make()
    batch = roll(model, 0.7)
    with torch.no_grad():
        gap = (answer_logprobs(model, batch, 1.0) - batch.sampler_logprobs).abs().max()
    assert gap > 0.1  # against about 5e-7 when the temperatures agree


def test_only_the_answers_have_log_probs():
    model = make()
    batch = roll(model, 1.0)
    with torch.no_grad():
        logp = answer_logprobs(model, batch, 1.0)
    assert torch.all(logp.isfinite())  # PAD's -inf never leaks out
    assert torch.all(logp[batch.response_mask] < 0)
    assert torch.all(logp[~batch.response_mask] == 0)


def test_other_rows_and_padding_change_nothing():
    model = make()
    batch = roll(model, 1.0)
    with torch.no_grad():
        together = answer_logprobs(model, batch, 1.0)
        for row in range(len(batch.finish_reasons)):
            alone = answer_logprobs(model, one_row(batch, row), 1.0)
            torch.testing.assert_close(alone[0], together[row, : alone.shape[1]])


def test_gradients_reach_the_weights():
    model = make()
    answer_logprobs(model, roll(model, 1.0), 1.0).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.all(parameter.grad.isfinite()), name
    assert model.head.weight.grad is not None and model.head.weight.grad.abs().sum() > 0
