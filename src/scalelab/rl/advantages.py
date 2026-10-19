"""Advantages: how much better or worse each answer did than the other answers to its prompt."""

from typing import Literal

import torch
from torch import Tensor

Estimator = Literal["grpo", "dr_grpo", "rloo"]


def group_size(group_ids: Tensor) -> int:
    """How many answers each prompt got. Rows must come the way rollouts lay them out: equal
    groups, one after another (0, 0, 1, 1, 2, 2, ...), at least 2 answers each."""
    groups = int(group_ids.max()) + 1
    size = len(group_ids) // groups
    expected = torch.arange(groups).repeat_interleave(size)
    if not torch.equal(group_ids.cpu(), expected):
        raise ValueError("rows must come in equal groups, one after another")
    if size < 2:
        raise ValueError("advantages compare answers to the same prompt: each needs at least 2")
    return size


def advantages(
    rewards: Tensor, group_ids: Tensor, estimator: Estimator, eps: float = 1e-6
) -> Tensor:
    """(rows,) each row's advantage: its reward compared with the other rewards in its group.

    grpo     (R - mean) / (std + eps): the group's spread sets the scale
    dr_grpo  R - mean: the same, without dividing by the spread
    rloo     R - the mean of the *other* rewards ("leave one out")

    std is the sample standard deviation, as common GRPO code computes it. A group whose
    rewards are all equal gets advantage 0 everywhere: nothing in it to learn from.
    """
    grouped = rewards.view(-1, group_size(group_ids))  # one row per prompt
    mean = grouped.mean(dim=1, keepdim=True)
    if estimator == "dr_grpo":
        result = grouped - mean
    elif estimator == "grpo":
        result = (grouped - mean) / (grouped.std(dim=1, keepdim=True) + eps)
    else:
        others = (grouped.sum(dim=1, keepdim=True) - grouped) / (grouped.shape[1] - 1)
        result = grouped - others
    return result.flatten()
