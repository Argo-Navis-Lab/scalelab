"""Loss tests: the k3 KL estimator, against hand-worked values, float32 rounding and the exact
KL it estimates."""

import math

import pytest
import torch

from scalelab.rl.losses import kl_k3

P = torch.tensor([0.5, 0.3, 0.2])  # the policy's probabilities for three tokens
Q = torch.tensor([0.2, 0.3, 0.5])  # the reference's


def test_k3_is_0_where_the_policies_agree():
    logp = torch.log(torch.tensor([0.5, 0.25, 1.0]))  # log 1 = 0, as outside the answers
    assert kl_k3(logp, logp).tolist() == [0.0, 0.0, 0.0]


def test_k3_by_hand():
    # r = 0.25 / 0.5 = 0.5: 0.5 - ln 0.5 - 1 = 0.193. The other way round, r = 2: 0.307.
    half, quarter = torch.tensor([math.log(0.5)]), torch.tensor([math.log(0.25)])
    assert kl_k3(half, quarter).item() == pytest.approx(0.5 + math.log(2) - 1)
    assert kl_k3(quarter, half).item() == pytest.approx(2 - math.log(2) - 1)


def test_k3_is_never_negative():
    generator = torch.Generator().manual_seed(0)
    logp = -torch.rand(100_000, generator=generator) * 10  # log-probs from 0 down to -10
    far = -torch.rand(100_000, generator=generator) * 10  # a reference far away
    near = logp + (torch.rand(100_000, generator=generator) - 0.5) * 1e-2  # and one close by
    assert torch.all(kl_k3(logp, far) >= 0)
    assert torch.all(kl_k3(logp, near) >= 0)


def test_close_to_r_1_k3_is_still_accurate():
    # For a log ratio x near 0, r - log r - 1 = x² / 2 + x³ / 6 + ...: 5.0e-7 at x = ±0.001.
    # Written as exp(x) - x - 1 in float32 it comes out 4.8e-7, 5% off.
    x = torch.tensor([1e-3, -1e-3])
    expected = (x**2 / 2 + x**3 / 6).tolist()
    assert kl_k3(torch.zeros(2), x).tolist() == pytest.approx(expected, rel=1e-2)


def test_on_the_policys_own_draws_k3_averages_to_the_true_kl():
    exact = (P * (P / Q).log()).sum()  # KL(P || Q) = 0.5 ln 2.5 + 0.2 ln 0.4 = 0.2749
    generator = torch.Generator().manual_seed(0)
    tokens = torch.multinomial(P, 100_000, replacement=True, generator=generator)  # from P
    estimate = kl_k3(P.log()[tokens], Q.log()[tokens]).mean()
    assert estimate.item() == pytest.approx(exact.item(), abs=0.003)


def test_the_gradient_pulls_the_policy_toward_the_reference():
    ref_logp = torch.full((3,), math.log(0.3))
    logp = (ref_logp + torch.tensor([0.5, 0.0, -0.5])).requires_grad_()  # above, equal, below
    kl_k3(logp, ref_logp).sum().backward()
    assert logp.grad is not None
    above, equal, below = logp.grad.tolist()
    # Gradient descent subtracts the gradient: a log-prob above the reference's goes down,
    # one below goes up, and one already equal stays put.
    assert above > 0 and equal == 0 and below < 0
