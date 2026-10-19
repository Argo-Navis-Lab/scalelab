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
