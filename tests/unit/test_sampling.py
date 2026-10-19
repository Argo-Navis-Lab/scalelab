"""Sampling tests: the policy's log-probabilities, what temperature does to them, and draws
that follow them."""

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from scalelab.core.config import SamplingParams
from scalelab.rl.sampling import NEVER_WRITTEN, log_probs, sample
from scalelab.tasks.tokenizer import PAD, VOCAB_SIZE, encode

THREE = encode("012")  # the token ids of "0", "1" and "2"
ALLOWED = [token for token in range(VOCAB_SIZE) if token not in NEVER_WRITTEN]


def random_logits(rows: int) -> Tensor:
    return torch.randn(rows, VOCAB_SIZE, generator=torch.Generator().manual_seed(0))


def three_way(p: list[float]) -> Tensor:
    """Logits (1, VOCAB_SIZE) that give "0", "1" and "2" the probabilities p, and every other
    token next to none (e^-50 of the weight)."""
    logits = torch.full((1, VOCAB_SIZE), -50.0)
    logits[0, THREE] = torch.tensor(p).log()
    return logits


def frequencies(logits: Tensor, temperature: float, draws: int = 20_000) -> list[float]:
    """How often "0", "1" and "2" come up in `draws` draws from the same logits."""
    params = SamplingParams(temperature=temperature)
    tokens, _ = sample(logits.expand(draws, -1), params, torch.Generator().manual_seed(0))
    return [(tokens == token).float().mean().item() for token in THREE]


# --- the settings ---------------------------------------------------------------------------


def test_the_default_temperature_is_1():
    assert SamplingParams().temperature == 1.0


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_a_temperature_must_be_positive(temperature: float):
    with pytest.raises(ValidationError):
        SamplingParams(temperature=temperature)


# --- the policy's log-probabilities ---------------------------------------------------------


def test_log_probs_are_a_distribution_over_the_tokens_a_model_may_write():
    logp = log_probs(random_logits(4), temperature=1.0)
    torch.testing.assert_close(logp.exp().sum(dim=-1), torch.ones(4))
    assert torch.all(logp[:, NEVER_WRITTEN] == float("-inf"))  # probability exactly 0
    assert torch.all(logp[:, ALLOWED].isfinite())


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_temperature_divides_the_gaps_between_logits(temperature: float):
    # What temperature means: log p_i - log p_j = (logit_i - logit_j) / T, for every pair.
    logits = random_logits(1)[0]
    logp = log_probs(logits, temperature)
    gaps = logp[ALLOWED] - logp[ALLOWED[0]]
    torch.testing.assert_close(gaps, (logits[ALLOWED] - logits[ALLOWED[0]]) / temperature)


def test_gradients_reach_only_the_tokens_a_model_may_write():
    logits = random_logits(1).requires_grad_()
    log_probs(logits, temperature=0.7)[0, THREE[0]].backward()
    assert logits.grad is not None
    assert torch.all(logits.grad.isfinite())
    assert torch.all(logits.grad[0, NEVER_WRITTEN] == 0)


# --- drawing tokens -------------------------------------------------------------------------


def test_draws_follow_the_probabilities():
    observed = frequencies(three_way([0.5, 0.3, 0.2]), temperature=1.0)
    assert observed == pytest.approx([0.5, 0.3, 0.2], abs=0.015)


def test_temperature_2_draws_from_the_square_roots_of_the_probabilities():
    # p ** (1 / T), made to add up to 1 again: flatter than p.
    roots = [0.5**0.5, 0.3**0.5, 0.2**0.5]
    expected = [root / sum(roots) for root in roots]  # 0.415, 0.322, 0.263
    observed = frequencies(three_way([0.5, 0.3, 0.2]), temperature=2.0)
    assert observed == pytest.approx(expected, abs=0.015)


def test_never_written_tokens_are_never_drawn():
    logits = torch.zeros(1000, VOCAB_SIZE)
    logits[:, PAD] = 100.0  # the model "wants" padding above all
    tokens, _ = sample(logits, SamplingParams(), torch.Generator().manual_seed(0))
    assert not set(tokens.tolist()) & set(NEVER_WRITTEN)


def test_a_draws_log_probs_are_the_policys():
    # The sampler's record must be exactly what log_probs gives at the same temperature: the
    # trainer recomputes it that way, and the two are compared.
    logits = random_logits(8)
    params = SamplingParams(temperature=0.7)
    tokens, logp = sample(logits, params, torch.Generator().manual_seed(0))
    expected = log_probs(logits, 0.7).gather(-1, tokens[:, None]).squeeze(-1)
    assert torch.equal(logp, expected)


def test_the_generator_decides_the_draws():
    logits = random_logits(64)
    params = SamplingParams()

    def draw(seed: int) -> Tensor:
        return sample(logits, params, torch.Generator().manual_seed(seed))[0]

    assert torch.equal(draw(0), draw(0))
    assert not torch.equal(draw(0), draw(1))
