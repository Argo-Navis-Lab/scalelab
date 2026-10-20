"""Filter tests. Dynamic sampling: only groups whose rewards vary are kept, renumbered and
narrowed, and the advantages of what is kept don't change. Overlong handling: unfinished
answers left out of the loss, and the soft penalty by hand."""

import pytest
import torch

from scalelab.core.config import Estimator, Overlong
from scalelab.rl.advantages import advantages
from scalelab.rl.filters import drop_flat_groups, loss_mask, overlong_penalty, varied_groups
from scalelab.rl.rollout import RolloutBatch
from scalelab.tasks.tokenizer import EOS, PAD, encode

# Three groups of two answers: both right (written out in full, the batch's longest rows), one
# right and one wrong, both wrong. Only the middle group has something to teach.
GROUPS = [
    ("<bos>12+34=", ["4+2+0=6,6;1+3+0=4,46;#46<eos>", "4+2+0=6,6;1+3+0=4,46;#46<eos>"]),
    ("<bos>123*456=", ["#56088<eos>", "#5608<eos>"]),
    ("<bos>56-78=", ["#1<eos>", "#2<eos>"]),
]
REWARDS = torch.tensor([1.0, 1, 1, 0, 0, 0])
# Four answers of 21, 22, 23 and 25 tokens to a 7-token prompt, in a context of 32: room for
# 25. The last ran out of context before writing <eos>.
LONG = [("<bos>12+34=", ["1" * 20 + "<eos>", "1" * 21 + "<eos>", "1" * 22 + "<eos>", "1" * 25])]


def batch_of(groups: list[tuple[str, list[str]]]) -> RolloutBatch:
    """A RolloutBatch built by hand. Each row's sampler log-probs are marked -0.1 * (row + 1),
    so the rows can be told apart after filtering."""
    rows = [
        (group, encode(prompt), encode(text))
        for group, (prompt, texts) in enumerate(groups)
        for text in texts
    ]
    lengths = [len(prompt) + len(answer) for _, prompt, answer in rows]
    tokens = torch.full((len(rows), max(lengths)), PAD)
    sampler_logprobs = torch.zeros(len(rows), max(lengths))
    for row, (_, prompt, answer) in enumerate(rows):
        tokens[row, : lengths[row]] = torch.tensor(prompt + answer)
        sampler_logprobs[row, len(prompt) : lengths[row]] = -0.1 * (row + 1)
    return RolloutBatch(
        tokens=tokens,
        prompt_lengths=torch.tensor([len(prompt) for _, prompt, _ in rows]),
        lengths=torch.tensor(lengths),
        finish_reasons=["eos" if answer[-1] == EOS else "length" for _, _, answer in rows],
        sampler_logprobs=sampler_logprobs,
        group_ids=torch.tensor([group for group, _, _ in rows]),
        policy_version=7,
    )


def test_only_groups_whose_rewards_vary_count_as_varied():
    batch = batch_of(GROUPS)
    assert varied_groups(REWARDS, batch.group_ids).tolist() == [False, True, False]


def test_dropping_keeps_the_varied_groups_rows_renumbered():
    batch = batch_of(GROUPS)
    kept, rewards = drop_flat_groups(batch, REWARDS)
    assert kept.group_ids.tolist() == [0, 0]  # the old group 1, now the only one
    assert rewards.tolist() == [1.0, 0.0]
    assert kept.prompt_lengths.tolist() == [9, 9]
    assert kept.lengths.tolist() == [16, 15]
    assert kept.finish_reasons == ["eos", "eos"]
    assert kept.policy_version == 7
    width = kept.tokens.shape[1]
    assert torch.equal(kept.tokens, batch.tokens[2:4, :width])  # rows 2 and 3, untouched
    assert torch.equal(kept.sampler_logprobs, batch.sampler_logprobs[2:4, :width])


def test_the_batch_narrows_to_its_longest_remaining_row():
    batch = batch_of(GROUPS)
    kept, _ = drop_flat_groups(batch, REWARDS)
    assert (batch.tokens.shape[1], kept.tokens.shape[1]) == (32, 16)  # less padding to train on


@pytest.mark.parametrize("estimator", ["grpo", "dr_grpo", "rloo"])
def test_the_kept_rows_advantages_do_not_change(estimator: Estimator):
    batch = batch_of(GROUPS)
    before = advantages(REWARDS, batch.group_ids, estimator)
    kept, rewards = drop_flat_groups(batch, REWARDS)
    after = advantages(rewards, kept.group_ids, estimator)
    assert torch.equal(after, before[2:4])
    assert before[[0, 1, 4, 5]].tolist() == [0.0] * 4  # what was dropped had nothing to teach


def test_when_no_group_varies_nothing_is_left():
    kept, rewards = drop_flat_groups(batch_of(GROUPS), torch.ones(6))
    assert kept.tokens.shape == (0, 0)
    assert kept.finish_reasons == []
    assert rewards.tolist() == []


# --- overlong handling ----------------------------------------------------------------------


def test_the_soft_penalty_by_hand():
    # With a zone of 4 tokens: none at 21 (4 short of the room of 25), then -1/4 a token.
    penalty = overlong_penalty(batch_of(LONG), context=32, buffer=4)
    assert penalty.tolist() == [0.0, -0.25, -0.5, -1.0]


def test_truncate_mask_leaves_out_the_answers_that_ran_out_of_context():
    batch = batch_of(LONG)
    mask = loss_mask(batch, "truncate_mask")
    assert torch.equal(mask[:3], batch.response_mask[:3])  # the three that wrote <eos>
    assert not mask[3].any()  # the one that didn't: none of its tokens


@pytest.mark.parametrize("overlong", ["none", "soft_penalty"])
def test_the_other_settings_keep_every_answer_in_the_loss(overlong: Overlong):
    batch = batch_of(LONG)
    assert torch.equal(loss_mask(batch, overlong), batch.response_mask)
