"""RL's loss and its pieces, all built from per-token log-probabilities."""

from typing import Literal

import torch
from torch import Tensor

Normalizer = Literal["seq_mean", "token_mean", "constant"]


def kl_k3(logp: Tensor, ref_logp: Tensor) -> Tensor:
    """Each token's "k3" estimate of how far the policy has moved from the reference:
    r - log r - 1, with r = pi_ref / pi_theta = exp(ref_logp - logp).

    Never negative, and exactly 0 where the two log-probs agree (so also at the zeros
    outside the answers). On tokens drawn from the policy, its average is the true
    KL(pi_theta || pi_ref). Gradients flow through logp, the policy being trained; ref_logp
    comes from the frozen reference.
    """
    log_ratio = ref_logp - logp
    # expm1(x) - x is r - log r - 1 written for float32: near r = 1 the plain
    # exp(x) - x - 1 loses its small answer (about x² / 2) to rounding, and can even dip
    # below 0. The clamp removes whatever rounding still leaves below 0.
    return (torch.expm1(log_ratio) - log_ratio).clamp(min=0)


def clipped_objective(
    ratio: Tensor, advantages: Tensor, clip_low: float, clip_high: float
) -> Tensor:
    """Each token's PPO-style objective, the number training pushes up:
    min(ratio * A, clip(ratio, 1 - clip_low, 1 + clip_high) * A).

    `ratio` (rows, width) is pi_theta / pi_old for every token, exp(logp - old_logp);
    `advantages` (rows,) holds each row's A, shared by all its tokens. Inside the clip range
    this is ratio * A. Outside it, once moving further would only raise the objective, it
    stays put at the edge, and that token's gradient is 0. Positions outside the answers
    (ratio 1) come out as A: the loss masks them.
    """
    adv = advantages[:, None]  # (rows, 1): each row's A, for all its tokens
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    return torch.minimum(ratio * adv, clipped * adv)


def clip_engaged(ratio: Tensor, advantages: Tensor, clip_low: float, clip_high: float) -> Tensor:
    """(rows, width) True where clipping took a token's gradient away: ratio above
    1 + clip_high when A > 0, or below 1 - clip_low when A < 0."""
    adv = advantages[:, None]
    return ((adv > 0) & (ratio > 1 + clip_high)) | ((adv < 0) & (ratio < 1 - clip_low))


def normalize(
    per_token: Tensor, mask: Tensor, normalizer: Normalizer, constant: int | None = None
) -> Tensor:
    """Turn per-token values (rows, width) into one number, counting only where `mask`
    (the answers' tokens) is True:

    seq_mean    each answer's average, then the average over answers: every answer counts
                the same, so a short answer's tokens weigh more than a long one's
    token_mean  the average over all answer tokens: every token counts the same
    constant    the sum over all answer tokens divided by rows * `constant` (such as the
                longest answer allowed): no division by any answer's own length
    """
    total = torch.where(mask, per_token, 0.0)
    if normalizer == "seq_mean":
        return (total.sum(dim=1) / mask.sum(dim=1)).mean()
    if normalizer == "token_mean":
        return total.sum() / mask.sum()
    if constant is None:
        raise ValueError("the constant normalizer needs `constant`")
    return total.sum() / (mask.shape[0] * constant)


def rl_loss(
    logp: Tensor,
    old_logp: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    clip_low: float,
    clip_high: float,
    normalizer: Normalizer,
    constant: int | None = None,
    kl_coef: float = 0.0,
    ref_logp: Tensor | None = None,
    weights: Tensor | None = None,
) -> Tensor:
    """The loss to minimize, one number:
    -normalize(w * clipped objective) + kl_coef * normalize(k3 KL to the reference).

    logp (rows, width) are the policy's log-probs, with gradients; old_logp the log-probs
    the batch started from; ref_logp the frozen reference's (needed when kl_coef > 0);
    advantages (rows,); mask (rows, width) marks the answers' tokens. `weights` are the
    per-token w, 1 when None (importance-sampling corrections set them, from m5).
    """
    ratio = (logp - old_logp).exp()
    objective = clipped_objective(ratio, advantages, clip_low, clip_high)
    if weights is not None:
        objective = weights * objective
    per_token = -objective
    if kl_coef > 0:
        if ref_logp is None:
            raise ValueError("a KL penalty (kl_coef > 0) needs the reference's log-probs")
        per_token = per_token + kl_coef * kl_k3(logp, ref_logp)
    return normalize(per_token, mask, normalizer, constant)
