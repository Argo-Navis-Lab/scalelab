"""Filters between scoring and training: which rollouts are worth a training pass."""

import torch
from torch import Tensor

from scalelab.core.config import Overlong
from scalelab.rl.advantages import group_size
from scalelab.rl.rollout import RolloutBatch


def varied_groups(rewards: Tensor, group_ids: Tensor) -> Tensor:
    """(groups,) True for each group whose rewards are not all the same: the only groups
    whose advantages, under any estimator, are not all 0."""
    grouped = rewards.view(-1, group_size(group_ids))  # one row per prompt
    return grouped.max(dim=1).values > grouped.min(dim=1).values


def keep_groups(batch: RolloutBatch, keep: Tensor) -> RolloutBatch:
    """The batch with only the groups where `keep` (groups,) is True, numbered 0, 1, ... again
    in their old order, and cut to the width its longest remaining row needs."""
    index = keep[batch.group_ids].nonzero().squeeze(1)  # the rows to keep
    lengths = batch.lengths[index]
    width = int(lengths.max()) if len(index) else 0
    renumbered = keep.long().cumsum(dim=0) - 1  # old group -> its new number
    return RolloutBatch(
        tokens=batch.tokens[index, :width],
        prompt_lengths=batch.prompt_lengths[index],
        lengths=lengths,
        finish_reasons=[batch.finish_reasons[row] for row in index.tolist()],
        sampler_logprobs=batch.sampler_logprobs[index, :width],
        group_ids=renumbered[batch.group_ids[index]],
        policy_version=batch.policy_version,
    )


def drop_flat_groups(batch: RolloutBatch, rewards: Tensor) -> tuple[RolloutBatch, Tensor]:
    """Dynamic sampling (DAPO): drop every group whose answers all scored the same. Their
    advantages are all 0, so training on them would cost a pass and teach nothing.

    Returns the smaller batch and its rewards. It can be empty when no group varied: the
    loop then skips the update.
    """
    varied = varied_groups(rewards, batch.group_ids)
    return keep_groups(batch, varied), rewards[varied[batch.group_ids]]


def loss_mask(batch: RolloutBatch, overlong: Overlong) -> Tensor:
    """(rows, width) the tokens the loss counts: every answer's, except that "truncate_mask"
    (DAPO's overlong filtering) leaves out the answers that ran out of context before writing
    <eos>. An unfinished answer's reward says little about the tokens it wrote."""
    if overlong != "truncate_mask":
        return batch.response_mask
    finished = torch.tensor([reason == "eos" for reason in batch.finish_reasons])
    return batch.response_mask & finished[:, None]


def overlong_penalty(batch: RolloutBatch, context: int, buffer: int) -> Tensor:
    """(rows,) DAPO's soft overlong penalty, added to each row's reward: 0 while an answer
    stays at least `buffer` tokens short of the longest it could be (the context minus its
    prompt), then falling in a straight line to -1 at that limit."""
    room = context - batch.prompt_lengths  # the longest each answer could be
    written = batch.lengths - batch.prompt_lengths
    into_zone = (written - (room - buffer)).clamp(min=0)
    return -into_zone.float() / buffer
