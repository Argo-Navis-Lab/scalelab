"""Scoring tests: each rollout row checked against its own group's problem, the format penalty
kept apart, and answers read exactly as grading reads them."""

import pytest
import torch

from scalelab.rl.rollout import RolloutBatch
from scalelab.rl.scoring import completion_text, reward_values, score_rollouts
from scalelab.tasks.arithmetic import ArithmeticProblem, ArithmeticTask
from scalelab.tasks.tokenizer import EOS, PAD, encode

TASK = ArithmeticTask()
ADD = ArithmeticProblem(a=12, op="+", b=34)  # 46
MUL = ArithmeticProblem(a=23, op="*", b=45)  # 1035
ANSWERS = [  # what the model "wrote" for each problem; "<eos>" stands for the token
    (ADD, ["#46<eos>", "#47<eos>", "46<eos>", "#4#6<eos>", "#46"]),
    (MUL, ["#1035<eos>", "#01035<eos>", "#-1035<eos>"]),
]


def batch_of(answers: list[tuple[ArithmeticProblem, list[str]]]) -> RolloutBatch:
    """A RolloutBatch built by hand: one group per problem, one row per answer text."""
    rows = [
        (group, TASK.format_prompt(problem), encode(text))
        for group, (problem, texts) in enumerate(answers)
        for text in texts
    ]
    lengths = [len(prompt) + len(answer) for _, prompt, answer in rows]
    tokens = torch.full((len(rows), max(lengths)), PAD)
    for row, (_, prompt, answer) in enumerate(rows):
        tokens[row, : lengths[row]] = torch.tensor(prompt + answer)
    return RolloutBatch(
        tokens=tokens,
        prompt_lengths=torch.tensor([len(prompt) for _, prompt, _ in rows]),
        lengths=torch.tensor(lengths),
        finish_reasons=["eos" if answer[-1] == EOS else "length" for _, _, answer in rows],
        sampler_logprobs=torch.zeros(tokens.shape),
        group_ids=torch.tensor([group for group, _, _ in rows]),
        policy_version=0,
    )


def test_each_row_is_scored_against_its_own_groups_problem():
    rewards = score_rollouts(TASK, [ADD, MUL], batch_of(ANSWERS))
    assert [reward.value for reward in rewards] == [1, 0, 0, 0, 1, 1, 0, 0]
    assert [reward.flags for reward in rewards] == [
        (),
        ("wrong_answer",),
        ("no_answer",),
        ("several_answers",),
        (),  # ran out of context before <eos>, but what it wrote reads right
        (),
        ("malformed_answer",),
        ("wrong_answer",),
    ]


def test_the_format_penalty_counts_only_when_weighted():
    rewards = score_rollouts(TASK, [ADD, MUL], batch_of(ANSWERS))
    assert reward_values(rewards).tolist() == [1, 0, 0, 0, 1, 1, 0, 0]
    weighted = reward_values(rewards, format_weight=0.5).tolist()
    assert weighted == [1, 0, -0.5, -0.5, 1, 1, -0.5, 0]  # unreadable answers lose 0.5 more


def test_problems_that_dont_match_the_prompts_are_refused():
    with pytest.raises(ValueError, match="row 0 doesn't answer problem 0"):
        score_rollouts(TASK, [MUL, ADD], batch_of(ANSWERS))  # the wrong order
    with pytest.raises(ValueError, match="row 5 doesn't answer problem 1"):
        score_rollouts(TASK, [ADD], batch_of(ANSWERS))  # one problem missing


@pytest.mark.parametrize(
    ("tokens", "text"),
    [(encode("#46<eos>"), "#46"), (encode("#46"), "#46"), (encode("<eos>"), ""), ([], "")],
)
def test_the_completion_text_drops_only_a_final_eos(tokens: list[int], text: str):
    assert completion_text(tokens) == text
