"""Sampling: drawing each next token at random from the model's probabilities, and the
log-probabilities that RL's loss is built from."""

import torch
import torch.nn.functional as F
from torch import Tensor

from scalelab.core.config import SamplingParams
from scalelab.tasks.tokenizer import BOS, PAD, VOCAB, VOCAB_SIZE

# Tokens the model may never write: padding, a second <bos>, and the embedding rows that no
# token uses yet (m1s1 left room for more symbols). An untrained model could pick any.
NEVER_WRITTEN = [PAD, BOS, *range(len(VOCAB), VOCAB_SIZE)]


def log_probs(logits: Tensor, temperature: float) -> Tensor:
    """The policy: the log-probability of every possible next token, from the model's logits
    (..., VOCAB_SIZE) divided by the temperature, with NEVER_WRITTEN at probability 0.

    The sampler draws from these and the trainer recomputes them, so both must come from
    this one function: a temperature applied by one and not the other is a classic RL bug.
    """
    never = torch.tensor(NEVER_WRITTEN, device=logits.device)
    scaled = (logits / temperature).index_fill(-1, never, float("-inf"))
    return F.log_softmax(scaled, dim=-1)


@torch.no_grad()
def sample(
    logits: Tensor, params: SamplingParams, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Draw one next token for each row of `logits` (batch, VOCAB_SIZE). Returns the tokens
    (batch,) and their log-probabilities (batch,): the sampler's own record of the draw.

    All the randomness comes from `generator` (on the logits' device), so the same generator
    state always draws the same tokens.
    """
    logp = log_probs(logits, params.temperature)
    tokens = torch.multinomial(logp.exp(), num_samples=1, generator=generator)  # (batch, 1)
    return tokens.squeeze(-1), logp.gather(-1, tokens).squeeze(-1)
