"""Loss tests: the k3 KL estimator (against hand-worked values, float32 rounding and the exact
KL it estimates), the clipped objective (where clipping engages, and where it doesn't), the
three normalizers by hand, and the whole loss."""

import math
from typing import Any

import pytest
import torch
from torch import Tensor

from scalelab.rl.losses import (
    Normalizer,
    clip_engaged,
    clipped_objective,
    kl_k3,
    normalize,
    rl_loss,
)

P = torch.tensor([0.5, 0.3, 0.2])  # the policy's probabilities for three tokens
Q = torch.tensor([0.2, 0.3, 0.5])  # the reference's
CLIP_LOW, CLIP_HIGH = 0.2, 0.28  # the DAPO-style range: [0.8, 1.28]
# Two answers side by side: one of 3 tokens, one of 1, then padding.
MASK = torch.tensor([[True, True, True, False], [True, False, False, False]])


def test_k3_is_0_where_the_policies_agree():
    logp = torch.log(torch.tensor([0.5, 0.25, 1.0]))  # log 1 = 0, as outside the answers
    assert kl_k3(logp, logp).tolist() == [0.0, 0.0, 0.0]


def test_k3_by_hand():
    # r = 0.25 / 0.5 = 0.5: 0.5 - ln 0.5 - 1 = 0.193. The other way round, r = 2: 0.307.
    half, quarter = torch.tensor([math.log(0.5)]), torch.tensor([math.log(0.25)])
    assert kl_k3(half, quarter).item() == pytest.approx(0.5 + math.log(2) - 1, abs=1e-6)
    assert kl_k3(quarter, half).item() == pytest.approx(2 - math.log(2) - 1, abs=1e-6)


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


# --- the clipped objective ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ratio", "advantage", "objective", "clipped"),
    [
        (1.0, 1.0, 1.0, False),  # nothing has changed yet: the objective is A
        (1.1, 1.0, 1.1, False),  # inside the range: ratio * A
        (1.25, 1.0, 1.25, False),  # still inside: the upper edge is 1.28
        (1.5, 1.0, 1.28, True),  # a good answer pushed past the edge: held there
        (0.5, 1.0, 0.5, False),  # a good answer made rarer: nothing stops pushing it back up
        (0.9, -1.0, -0.9, False),  # inside the range
        (0.5, -1.0, -0.8, True),  # a bad answer already made much rarer: held at 0.8
        (1.5, -1.0, -1.5, False),  # a bad answer made likelier: nothing stops pulling it back
    ],
)
def test_clipping_engages_only_outside_the_range_and_only_in_the_advantages_direction(
    ratio: float, advantage: float, objective: float, clipped: bool
):
    logp = torch.tensor([[math.log(ratio)]], requires_grad=True)  # old log-prob 0: ratio = e^logp
    advantages = torch.tensor([advantage])
    value = clipped_objective(logp.exp(), advantages, CLIP_LOW, CLIP_HIGH)
    value.sum().backward()
    assert value.item() == pytest.approx(objective, abs=1e-6)
    assert clip_engaged(logp.exp(), advantages, CLIP_LOW, CLIP_HIGH).item() == clipped
    assert logp.grad is not None
    assert (logp.grad.item() == 0) == clipped  # a clipped token gets no gradient


def test_before_any_update_the_gradient_is_each_rows_advantage():
    # Ratio 1 everywhere, as in a step's first update: the clipped objective's gradient is
    # then the plain policy gradient, A for every token of the row.
    logp = torch.zeros(3, 4, requires_grad=True)  # equal to the old log-probs
    advantages = torch.tensor([0.5, -1.0, 0.0])
    clipped_objective(logp.exp(), advantages, CLIP_LOW, CLIP_HIGH).sum().backward()
    assert logp.grad is not None
    assert torch.equal(logp.grad, advantages[:, None].expand(3, 4))


# --- normalizers and the whole loss ---------------------------------------------------------


@pytest.mark.parametrize(
    ("normalizer", "expected"),
    [
        ("seq_mean", 3.0),  # (6 / 3 + 4 / 1) / 2: every answer counts the same
        ("token_mean", 2.5),  # 10 / 4: every token counts the same
        ("constant", 1.25),  # 10 / (2 rows * 4)
    ],
)
def test_each_normalizer_by_hand(normalizer: Normalizer, expected: float):
    per_token = torch.tensor([[1.0, 2, 3, 99], [4, 99, 99, 99]])  # 99s: not in an answer
    assert normalize(per_token, MASK, normalizer, constant=4).item() == expected


def test_the_constant_normalizer_needs_its_constant():
    with pytest.raises(ValueError, match="needs `constant`"):
        normalize(torch.ones(2, 4), MASK, "constant")


def loss_of(logp: Tensor, normalizer: Normalizer = "token_mean", **knobs: Any) -> Tensor:
    """rl_loss for MASK's two answers: a right one (A = 1, 3 tokens) and a wrong one (A = -1,
    1 token), with old log-probs 0, the clip range [0.8, 1.2] and the constant 4."""
    advantages = torch.tensor([1.0, -1.0])
    return rl_loss(
        logp,
        torch.zeros(2, 4),
        advantages,
        MASK,
        clip_low=0.2,
        clip_high=0.2,
        normalizer=normalizer,
        constant=4,
        **knobs,
    )


@pytest.mark.parametrize(
    ("normalizer", "loss", "push"),
    [
        ("seq_mean", 0.0, [1 / 6, -1 / 2]),  # each answer's push is shared by its tokens
        ("token_mean", -0.5, [1 / 4, -1 / 4]),  # every token pushed alike
        ("constant", -0.25, [1 / 8, -1 / 8]),  # alike, and smaller: / (2 rows * 4)
    ],
)
def test_how_each_normalizer_shares_out_the_push(
    normalizer: Normalizer, loss: float, push: list[float]
):
    logp = torch.zeros(2, 4, requires_grad=True)  # equal to the old log-probs: ratio 1
    value = loss_of(logp, normalizer)
    value.backward()
    assert value.item() == pytest.approx(loss)
    # Gradient descent moves each log-prob by minus its gradient: that is the push.
    assert logp.grad is not None
    right, wrong = push
    torch.testing.assert_close(-logp.grad, torch.tensor([[right] * 3 + [0], [wrong, 0, 0, 0]]))


def test_the_kl_term_adds_kl_coef_times_the_normalized_k3():
    # The policy finds two of the right answer's tokens twice as likely as the reference does,
    # r = 0.5 for each: k3 = 0.5 + ln 2 - 1 = 0.193. Averaged over the 4 answer tokens.
    ref_logp = torch.log(torch.tensor([[0.5, 1.0, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]]))
    without = loss_of(torch.zeros(2, 4)).item()
    with_kl = loss_of(torch.zeros(2, 4), kl_coef=0.1, ref_logp=ref_logp).item()
    # float32 losses near -0.5, subtracted: rounding of ~1e-8 each, which another CPU's exp
    # and log can double. pytest.approx's default (one part in a million) is too tight.
    expected = 0.1 * 2 * (0.5 + math.log(2) - 1) / 4
    assert with_kl - without == pytest.approx(expected, abs=1e-6)


def test_a_kl_penalty_needs_the_references_log_probs():
    with pytest.raises(ValueError, match="needs the reference"):
        loss_of(torch.zeros(2, 4), kl_coef=0.04)


def test_a_weight_of_0_takes_a_token_out_of_the_policy_term():
    weights = torch.tensor([[1.0, 1, 1, 1], [0, 1, 1, 1]])  # the wrong answer's token weighs 0
    assert loss_of(torch.zeros(2, 4), weights=weights).item() == -3 / 4  # 3 tokens of -1, / 4


def test_nothing_outside_the_answers_reaches_the_loss():
    wild = torch.where(MASK, 0.0, 123.0)  # e^123 is inf in float32: ratios of inf out there
    knobs: dict[str, Any] = {"kl_coef": 0.04, "ref_logp": torch.zeros(2, 4)}
    assert loss_of(wild, **knobs).item() == loss_of(torch.zeros(2, 4), **knobs).item()
