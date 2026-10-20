"""The pieces of RL's loss, all built from per-token log-probabilities."""

import torch
from torch import Tensor


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
