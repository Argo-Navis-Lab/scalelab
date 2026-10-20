"""The trainer's side of RL: what the model being trained makes of the sampled answers, and
the updates that train it on them."""

from collections.abc import Sequence

import torch
from torch import Tensor

from scalelab.core.config import AlgoConfig, StrictModel
from scalelab.models.transformer import Transformer
from scalelab.rl.losses import clip_engaged, kl_k3, normalizer_size, rl_loss
from scalelab.rl.rollout import RolloutBatch
from scalelab.rl.sampling import log_probs
from scalelab.telemetry.hooks import BackwardEvent, ForwardEvent, Hook, OptimizerStepEvent


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


class UpdateMetrics(StrictModel):
    """What one update (one pass over the batch, one optimizer step) did."""

    loss: float  # the whole batch's loss, before the step
    grad_norm: float  # the gradient's size before clipping
    clip_fraction: float  # the share of the loss's tokens whose gradient clipping took away
    kl: float  # the mean k3 KL to the reference per token; 0 without one


def rl_update(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    advantages: Tensor,
    mask: Tensor,
    algo: AlgoConfig,
    *,
    temperature: float,
    epochs: int = 1,
    micro_batch: int = 64,
    grad_clip: float = 1.0,
    reference: Transformer | None = None,
    hooks: Sequence[Hook] = (),
    step: int = 0,
) -> list[UpdateMetrics]:
    """Train `model` on a scored batch: `epochs` updates, each a pass over all its rows in
    micro-batches of `micro_batch` rows (their gradients add up), then one optimizer step
    with the gradient clipped to `grad_clip`. Returns each update's metrics.

    The old log-probs (where the ratio starts) are computed once, before any update, and
    the reference's too when algo.kl_coef > 0. `mask` (rows, width) marks the tokens the
    loss counts (see filters.loss_mask). With nothing to count, nothing changes and the
    result is [].

    Each of `hooks` is called after every micro-batch's loss (on_forward), once an update's
    gradients are complete (on_backward) and after each optimizer step (on_optimizer_step),
    told `step`: the RL step this is.
    """
    if algo.kl_coef > 0 and reference is None:
        raise ValueError("a KL penalty (kl_coef > 0) needs the reference model")
    device = model.embed.weight.device
    advantages, mask = advantages.to(device), mask.to(device)
    total = normalizer_size(mask, algo.normalizer)
    if total == 0:
        return []
    with torch.no_grad():
        old_logp = answer_logprobs(model, batch, temperature)
        ref_logp = None
        if algo.kl_coef > 0 and reference is not None:
            ref_logp = answer_logprobs(reference, batch, temperature)

    rows, tokens = len(batch.finish_reasons), int(mask.sum())
    metrics: list[UpdateMetrics] = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_sum, clipped, kl_sum = 0.0, 0, 0.0
        for start in range(0, rows, micro_batch):
            end = start + micro_batch
            part = batch.rows(start, end)
            width = part.tokens.shape[1]  # this micro-batch's own, often less than the batch's
            part_old, part_mask = old_logp[start:end, :width], mask[start:end, :width]
            part_adv = advantages[start:end]
            part_ref = None if ref_logp is None else ref_logp[start:end, :width]
            logp = answer_logprobs(model, part, temperature)
            loss = rl_loss(
                logp,
                part_old,
                part_adv,
                part_mask,
                clip_low=algo.clip_low,
                clip_high=algo.clip_high,
                normalizer=algo.normalizer,
                constant=algo.normalizer_constant,
                kl_coef=algo.kl_coef,
                ref_logp=part_ref,
            )
            forward = ForwardEvent(
                step=step,
                epoch=epoch,
                batch=part,
                logp=logp.detach(),  # hooks see the numbers, not the graph behind them
                old_logp=part_old,
                ref_logp=part_ref,
                advantages=part_adv,
                mask=part_mask,
                loss=loss.detach(),
            )
            for hook in hooks:
                hook.on_forward(forward)
            share = normalizer_size(part_mask, algo.normalizer) / total
            (loss * share).backward()  # the micro-batches' gradients add up to the batch's
            with torch.no_grad():
                ratio = (logp - part_old).exp()
                engaged = clip_engaged(ratio, part_adv, algo.clip_low, algo.clip_high)
                loss_sum += loss.item() * share
                clipped += int((engaged & part_mask).sum())
                if part_ref is not None:
                    kl_sum += float(kl_k3(logp, part_ref)[part_mask].sum())
        backward = BackwardEvent(step=step, epoch=epoch, model=model)
        for hook in hooks:
            hook.on_backward(backward)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        stepped = OptimizerStepEvent(
            step=step, epoch=epoch, model=model, optimizer=optimizer, grad_norm=grad_norm.item()
        )
        for hook in hooks:
            hook.on_optimizer_step(stepped)
        metrics.append(
            UpdateMetrics(
                loss=loss_sum,
                grad_norm=grad_norm.item(),
                clip_fraction=clipped / tokens,
                kl=kl_sum / tokens,
            )
        )
    return metrics
