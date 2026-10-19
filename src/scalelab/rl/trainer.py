"""The trainer's side of RL: what the model being trained makes of the sampled answers."""

import torch
from torch import Tensor

from scalelab.models.transformer import Transformer
from scalelab.rl.rollout import RolloutBatch
from scalelab.rl.sampling import log_probs


def answer_logprobs(model: Transformer, batch: RolloutBatch, temperature: float) -> Tensor:
    """(rows, width) each answer token's log-probability under `model` at `temperature`, in
    the same places as batch.tokens, and 0 outside the answers. On the model's device.

    One forward pass reads every row whole: the logits at position t score the token at
    t + 1. Gradients flow through the result, so the loss can be built on it; under
    torch.no_grad() it gives the log-probs a training step starts from. With top_p = 1 it
    must match batch.sampler_logprobs: the same policy, worked out by the trainer.
    """
    device = model.embed.weight.device
    tokens = batch.tokens.to(device)
    logp = log_probs(model(tokens[:, :-1]), temperature)  # (rows, width - 1, vocab)
    next_logp = logp.gather(-1, tokens[:, 1:, None]).squeeze(-1)  # each next token's
    aligned = torch.cat([torch.zeros_like(next_logp[:, :1]), next_logp], dim=1)  # (rows, width)
    # where, not a multiplication: PAD's log-prob is -inf, and -inf * 0 would be NaN.
    return torch.where(batch.response_mask.to(device), aligned, 0.0)
