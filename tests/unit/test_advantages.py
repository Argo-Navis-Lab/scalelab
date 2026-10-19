"""Advantage tests: hand-worked groups for each estimator, the blueprint's RLOO identity, and
groups with nothing to learn from."""

import pytest
import torch
from torch import Tensor

from scalelab.rl.advantages import Estimator, advantages, group_size

ESTIMATORS: list[Estimator] = ["grpo", "dr_grpo", "rloo"]


def groups_of(size: int, count: int) -> Tensor:
    """group_ids laid out as rollouts lay them out: 0, 0, ..., 1, 1, ..., count - 1."""
    return torch.arange(count).repeat_interleave(size)


# --- worked by hand -------------------------------------------------------------------------

# Two groups of 4 answers: one right of four (mean 0.25), then two right of four (mean 0.5).
REWARDS = torch.tensor([1.0, 0, 0, 0, 1, 1, 0, 0])


def test_dr_grpo_subtracts_the_group_mean():
    expected = [0.75, -0.25, -0.25, -0.25, 0.5, 0.5, -0.5, -0.5]
    assert advantages(REWARDS, groups_of(4, 2), "dr_grpo").tolist() == expected


def test_grpo_also_divides_by_the_groups_standard_deviation():
    # Sample standard deviations: sqrt((0.75² + 3 · 0.25²) / 3) = 0.5, and sqrt(4 · 0.5² / 3).
    second = 0.5 / (4 * 0.5**2 / 3) ** 0.5  # 0.866
    expected = [1.5, -0.5, -0.5, -0.5, second, second, -second, -second]
    grpo = advantages(REWARDS, groups_of(4, 2), "grpo").tolist()
    assert grpo == pytest.approx(expected, abs=1e-5)  # eps = 1e-6 moves the 6th digit


def test_rloo_compares_each_answer_with_the_others():
    # Row 0: the other three got 0, 0, 0. Row 1: the others got 1, 0, 0, a mean of 1/3.
    expected = [1, -1 / 3, -1 / 3, -1 / 3, 2 / 3, 2 / 3, -2 / 3, -2 / 3]
    assert advantages(REWARDS, groups_of(4, 2), "rloo").tolist() == pytest.approx(expected)


# --- properties -----------------------------------------------------------------------------


@pytest.mark.parametrize("size", [2, 4, 8])
def test_rloo_is_dr_grpo_scaled_by_g_over_g_minus_1(size: int):
    # The blueprint's test: RLOO = G / (G - 1) · (R - mean).
    rewards = torch.rand(5 * size, generator=torch.Generator().manual_seed(0))
    rloo = advantages(rewards, groups_of(size, 5), "rloo")
    dr_grpo = advantages(rewards, groups_of(size, 5), "dr_grpo")
    torch.testing.assert_close(rloo, size / (size - 1) * dr_grpo)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_a_group_whose_rewards_are_all_equal_teaches_nothing(estimator: Estimator):
    rewards = torch.tensor([1.0, 1, 1, 1, 0, 0, 0, 0])  # all right, then all wrong
    assert advantages(rewards, groups_of(4, 2), estimator).tolist() == [0.0] * 8


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_each_groups_advantages_add_up_to_0(estimator: Estimator):
    rewards = torch.rand(24, generator=torch.Generator().manual_seed(0))
    sums = advantages(rewards, groups_of(8, 3), estimator).view(3, 8).sum(dim=1)
    torch.testing.assert_close(sums, torch.zeros(3), atol=1e-5, rtol=0)


def test_only_grpo_ignores_the_scale_of_the_rewards():
    scaled = REWARDS * 10
    torch.testing.assert_close(
        advantages(scaled, groups_of(4, 2), "grpo"), advantages(REWARDS, groups_of(4, 2), "grpo")
    )
    torch.testing.assert_close(
        advantages(scaled, groups_of(4, 2), "dr_grpo"),
        10 * advantages(REWARDS, groups_of(4, 2), "dr_grpo"),
    )


# --- the layout -----------------------------------------------------------------------------


def test_the_group_size_comes_from_the_layout():
    assert group_size(groups_of(8, 3)) == 8


@pytest.mark.parametrize(
    "group_ids",
    [[0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 1, 2]],
    ids=["interleaved", "unequal", "one-answer-each"],
)
def test_layouts_that_cannot_be_grouped_are_refused(group_ids: list[int]):
    with pytest.raises(ValueError):
        group_size(torch.tensor(group_ids))
