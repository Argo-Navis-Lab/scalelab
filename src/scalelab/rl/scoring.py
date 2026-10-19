"""Scoring rollouts: each sampled answer checked by the task's verifier, as RL's reward."""

from collections.abc import Sequence

import torch
from torch import Tensor

from scalelab.rl.rollout import RolloutBatch
from scalelab.tasks.base import Task
from scalelab.tasks.rewards import Reward
from scalelab.tasks.tokenizer import EOS, decode


def completion_text(tokens: Sequence[int]) -> str:
    """What a verifier reads: the text a model wrote, without the <eos> that ended it. Grading
    and RL both read answers through this, so a reward means the same thing in both."""
    return decode(tokens[:-1] if tokens[-1:] == [EOS] else tokens)


def score_rollouts[P](task: Task[P], problems: Sequence[P], batch: RolloutBatch) -> list[Reward]:
    """Each row's Reward: its answer scored against the problem its group answers
    (problems[group]). Refuses a batch whose prompts aren't those problems' prompts, so an
    answer can never be scored against the wrong question."""
    rewards: list[Reward] = []
    groups, prompt_lengths = batch.group_ids.tolist(), batch.prompt_lengths.tolist()
    for row, answer in enumerate(batch.answers()):
        group = groups[row]
        prompt = batch.tokens[row, : prompt_lengths[row]].tolist()
        if group >= len(problems) or prompt != task.format_prompt(problems[group]):
            raise ValueError(f"row {row} doesn't answer problem {group}: the prompts differ")
        rewards.append(task.score(problems[group], completion_text(answer)))
    return rewards


def reward_values(rewards: Sequence[Reward], format_weight: float = 0.0) -> Tensor:
    """(rows,) the number RL maximizes for each row: the reward's value, plus format_weight
    times its format penalty. The weight is 0 unless an experiment sets it: the penalty is
    kept apart so its effect can be measured."""
    return torch.tensor([reward.value + format_weight * reward.format for reward in rewards])
