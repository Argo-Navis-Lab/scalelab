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


def nucleus(logp: Tensor, top_p: float) -> Tensor:
    """Top-p: keep the fewest likeliest tokens whose probabilities add up to at least top_p,
    give every other token probability 0, and scale the kept ones up to add up to 1 again.
    `logp` are log-probabilities (..., VOCAB_SIZE); top_p = 1 returns them unchanged."""
    if top_p >= 1:
        return logp
    ranked, order = logp.sort(dim=-1, descending=True, stable=True)  # likeliest first
    probs = ranked.exp()
    above = probs.cumsum(dim=-1) - probs  # the probability of the tokens ranked above each
    drop_ranked = above >= top_p  # those above already hold top_p: this one isn't needed
    drop = torch.zeros_like(drop_ranked).scatter(-1, order, drop_ranked)  # back to token order
    return logp.masked_fill(drop, float("-inf")).log_softmax(dim=-1)


@torch.no_grad()
def sample(
    logits: Tensor, params: SamplingParams, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Draw one next token for each row of `logits` (batch, VOCAB_SIZE). Returns the tokens
    (batch,) and their log-probabilities (batch,) under the distribution they were drawn
    from: the sampler's own record. With top_p = 1 that is the policy itself.

    All the randomness comes from `generator` (on the logits' device), so the same generator
    state always draws the same tokens.
    """
    logp = nucleus(log_probs(logits, params.temperature), params.top_p)
    tokens = torch.multinomial(logp.exp(), num_samples=1, generator=generator)  # (batch, 1)
    return tokens.squeeze(-1), logp.gather(-1, tokens).squeeze(-1)
