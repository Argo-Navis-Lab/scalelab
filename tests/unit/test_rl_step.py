"""RL step model tests: the longest of several rows, the expected maximum against the mean, a
step's phases as the sums of their op graphs, the same FLOPs the loop counts, the update, and a
warm start's step. And the padding cut (m3s15): the rank-r shortest row, sorted micro-batches."""

import itertools
from collections import Counter

import pytest
import torch
from pydantic import ValidationError

from scalelab.core.config import ModelConfig
from scalelab.costmodel.ops import count_flops
from scalelab.costmodel.rlstep import (
    RLStepSpec,
    StepPrediction,
    WarmStartStepSpec,
    expected_max,
    longest,
    optimizer_ops,
    predict_rl_step,
    predict_warm_start_step,
    ranked,
    sorted_pieces,
)
from scalelab.costmodel.roofline import group, predict, time_ops
from scalelab.costmodel.specs import HARDWARE, ModelSpec, WorkloadSpec
from scalelab.rl.flops import rl_step_flops
from scalelab.rl.rollout import RolloutBatch

SPEC = ModelSpec.from_config(ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1))
T4 = HARDWARE["t4"]
# Prompts of 5 and 7 tokens, 4 answers each: 16 rows. Length 5's 12 rows decode in chunks of 6
# and 6, length 7's 4 rows in one; the micro-batches of 8 are 5 + 7 (13 wide) and 5 + 5 (8).
PROMPTS = (5, 7, 5, 5)
ANSWERS = {5: (3,), 7: (6,)}


def _step(
    answers: dict[int, tuple[int, ...]], reference: bool = True, epochs: int = 1
) -> RLStepSpec:
    return RLStepSpec(
        prompt_lengths=PROMPTS,
        answer_lengths=answers,
        group_size=4,
        generation_batch=6,
        micro_batch=8,
        epochs=epochs,
        reference=reference,
    )


def _seconds(*workloads: WorkloadSpec) -> float:
    return sum(predict(SPEC, workload, T4).seconds for workload in workloads)


def _batch(step: RLStepSpec) -> RolloutBatch:
    """The step's rows as generate() returns them, each answer its prompt length's one."""
    prompts = torch.tensor(step.rows)
    lengths = prompts + torch.tensor([step.answer_lengths[n][0] for n in step.rows])
    rows, width = len(step.rows), int(lengths.max())
    return RolloutBatch(
        tokens=torch.zeros(rows, width, dtype=torch.long),
        prompt_lengths=prompts,
        lengths=lengths,
        finish_reasons=["eos"] * rows,
        sampler_logprobs=torch.zeros(rows, width),
        group_ids=torch.arange(len(step.prompt_lengths)).repeat_interleave(step.group_size),
        policy_version=0,
    )


def test_the_longest_of_rows() -> None:
    assert longest([[3, 5, 5]]) == pytest.approx({3: 1 / 3, 5: 2 / 3})
    assert longest([[1, 2], [1, 2]]) == pytest.approx({1: 0.25, 2: 0.75})
    assert longest([[1, 2], [3]]) == {1: 0.0, 2: 0.0, 3: 1.0}  # the second row always wins


@pytest.mark.parametrize("rows", [1, 2, 8, 512])
def test_expected_max_of_one_or_two_tokens(rows: int) -> None:
    assert expected_max([1, 2], rows) == pytest.approx(2 - 0.5**rows)


def test_one_row_runs_as_long_as_the_mean_and_a_batch_as_its_tail() -> None:
    tail = [10] * 99 + [100]  # 1 answer in 100 rambles
    assert expected_max(tail, 1) == pytest.approx(10.9)
    assert expected_max(tail, 512) == pytest.approx(100 - 90 * 0.99**512)  # 99.48
    assert _batch_time({5: tuple(tail)}) > 5 * _batch_time({5: (11,)})  # 11: about the mean


def _batch_time(answers: dict[int, tuple[int, ...]]) -> float:
    step = RLStepSpec(
        prompt_lengths=(5,) * 64,
        answer_lengths=answers,
        group_size=8,
        generation_batch=512,
        micro_batch=64,
        epochs=1,
        reference=True,
    )
    return predict_rl_step(SPEC, step, T4).phases["generate"].seconds


def test_each_phase_is_its_op_graphs_summed() -> None:
    phases = predict_rl_step(SPEC, _step(ANSWERS), T4).phases

    def chunk(rows: int, prompt: int, answer: int) -> float:  # the prompt, then each new token
        decodes = [
            WorkloadSpec(phase="decode", batch=rows, tokens=1, past=past)
            for past in range(prompt, prompt + answer - 1)
        ]
        return _seconds(WorkloadSpec(phase="prefill", batch=rows, tokens=prompt), *decodes)

    generate = 2 * chunk(6, 5, 3) + chunk(4, 7, 6)
    assert phases["generate"].seconds == pytest.approx(generate, rel=1e-12)
    logprobs = 2 * _seconds(WorkloadSpec(phase="prefill", batch=16, tokens=12))
    assert phases["logprobs"].seconds == pytest.approx(logprobs, rel=1e-12)
    micro_batches = _seconds(
        WorkloadSpec(phase="train", batch=8, tokens=12),
        WorkloadSpec(phase="train", batch=8, tokens=7),
    )
    update = time_ops(optimizer_ops(SPEC), T4).seconds
    assert phases["train"].seconds == pytest.approx(micro_batches + update, rel=1e-12)


@pytest.mark.parametrize("reference", [True, False])
@pytest.mark.parametrize("epochs", [1, 2])
def test_the_flops_are_what_the_loop_counts(reference: bool, epochs: int) -> None:
    step = _step(ANSWERS, reference, epochs)
    batch = _batch(step)
    flops = rl_step_flops(
        SPEC, batch, batch, batch_size=6, micro_batch=8, epochs=epochs, reference=reference
    )
    phases = predict_rl_step(SPEC, step, T4).phases
    assert phases["generate"].flops == flops.sample
    assert phases["logprobs"].flops == flops.logprobs
    assert phases["train"].flops == flops.train


def test_a_spread_of_lengths_lies_between_its_ends() -> None:
    def seconds(answers: tuple[int, ...]) -> StepPrediction:
        return predict_rl_step(SPEC, _step({5: answers, 7: answers}), T4)

    short, spread, long = seconds((2,)), seconds((2, 3, 9)), seconds((9,))
    for phase in ("generate", "logprobs", "train"):
        assert short.phases[phase].seconds < spread.phases[phase].seconds
        assert spread.phases[phase].seconds < long.phases[phase].seconds


def test_a_prediction_adds_up() -> None:
    prediction = predict_rl_step(SPEC, _step(ANSWERS), T4)
    for phase in prediction.phases.values():
        assert sum(phase.by_class.values()) == pytest.approx(phase.seconds, rel=1e-12)
    assert prediction.seconds == sum(phase.seconds for phase in prediction.phases.values())
    assert prediction.slowest in prediction.phases


def test_the_update_streams_every_weight_as_a_copy_does() -> None:
    ops = optimizer_ops(SPEC)
    values = SPEC.param_counts().total * 4
    assert sum(op.bytes_read for op in ops) == 6 * values  # gradients twice, AdamW four
    assert sum(op.bytes_written for op in ops) == 4 * values  # gradients once, AdamW three
    assert {group(op) for op in ops} == {"copy"}
    assert sum(op.flops for op in ops) == 0


@pytest.mark.parametrize(
    ("answers", "message"),
    [
        ({5: (3,)}, "no answer lengths for prompts of 7 tokens"),
        ({5: (3,), 7: ()}, "no answer lengths for prompts of 7 tokens"),
        ({5: (3,), 7: (0,)}, "at least one token"),
    ],
)
def test_every_prompt_length_needs_answers(
    answers: dict[int, tuple[int, ...]], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _step(answers)


# --- added in m3s11: a warm start's step ----------------------------------------------------


def test_a_warm_start_step_is_one_train_pass_then_the_update() -> None:
    step = predict_warm_start_step(SPEC, WarmStartStepSpec(lengths=(9,), batch_size=4), T4)
    assert list(step.phases) == ["train"]
    workload = WorkloadSpec(phase="train", batch=4, tokens=8)  # 9 tokens: 8 read in each row
    update = time_ops(optimizer_ops(SPEC), T4).seconds
    assert step.seconds == pytest.approx(_seconds(workload) + update, rel=1e-12)
    assert step.flops == count_flops(SPEC, workload).total  # what the loop counts


def test_a_warm_start_batch_is_as_wide_as_its_longest_example() -> None:
    # 5 or 40 tokens, even odds: all 8 rows are 5 long with chance 1/256, else the batch is 39 wide.
    step = predict_warm_start_step(SPEC, WarmStartStepSpec(lengths=(5, 40), batch_size=8), T4)
    narrow = _seconds(WorkloadSpec(phase="train", batch=8, tokens=4))
    wide = _seconds(WorkloadSpec(phase="train", batch=8, tokens=39))
    update = time_ops(optimizer_ops(SPEC), T4).seconds
    assert step.seconds == pytest.approx(narrow / 256 + wide * 255 / 256 + update, rel=1e-12)


def test_a_warm_start_example_reads_a_token_and_predicts_the_next() -> None:
    with pytest.raises(ValidationError, match="at least 2 tokens"):
        WarmStartStepSpec(lengths=(1, 9), batch_size=4)


def test_the_rank_r_shortest_of_rows() -> None:
    rows = [(1, 4), (2, 3, 3), (5,), (2, 4)]
    ways = list(itertools.product(*rows))  # every way the rows can fall, each as likely
    ranks = [1, 2, 3, 4]
    for rank, chances in zip(ranks, ranked(rows, ranks), strict=True):
        counts = Counter(sorted(way)[rank - 1] for way in ways)
        assert chances == pytest.approx({k: counts[k] / len(ways) for k in chances}, abs=1e-12)
    assert ranked(rows, [4])[0] == pytest.approx(longest(rows), abs=1e-12)


def test_sorted_pieces_are_as_wide_as_the_last_row_ranked_in_them() -> None:
    rows = [(1, 4), (2, 3, 3), (5,), (2, 4), (3,)]
    pieces = sorted_pieces(rows, 2)
    assert [size for size, _ in pieces] == [2, 2, 1]  # 5 rows, 2 at a time
    for (_, widths), rank in zip(pieces, [2, 4, 5], strict=True):
        assert widths == ranked(rows, [rank])[0]


def test_the_padding_cut_sorts_the_rows_and_runs_log_probs_in_micro_batches() -> None:
    # The 12 rows of 8 tokens (prompt 5 + answer 3) go first, then the 4 of 13: log-probs run as
    # 8 rows 8 wide and 8 rows 13 wide, not all 16 rows 13 wide (twice: a reference too).
    # Training's micro-batches were 13 and 8 wide as the rows came, and are 8 and 13 sorted.
    step = _step(ANSWERS)
    before = predict_rl_step(SPEC, step, T4)
    cut = predict_rl_step(SPEC, step.model_copy(update={"sort_rows": True}), T4)
    pieces = _seconds(
        WorkloadSpec(phase="prefill", batch=8, tokens=7),
        WorkloadSpec(phase="prefill", batch=8, tokens=12),
    )
    assert cut.phases["logprobs"].seconds == pytest.approx(2 * pieces, rel=1e-12)
    assert cut.phases["generate"] == before.phases["generate"]  # decoding never pads
    assert cut.phases["train"].seconds == pytest.approx(before.phases["train"].seconds, rel=1e-12)


def test_a_warm_start_under_the_padding_cut_runs_sorted_micro_batches() -> None:
    # 5 or 40 tokens, even odds, 8 rows in sorted pieces of 4. The first is narrow when 4 or more
    # rows are 5 long (163 of 256 ways); the second, the longest row's, when all 8 are (1 of 256).
    spec = WarmStartStepSpec(lengths=(5, 40), batch_size=8, micro_batch=4)
    step = predict_warm_start_step(SPEC, spec, T4)
    narrow = _seconds(WorkloadSpec(phase="train", batch=4, tokens=4))
    wide = _seconds(WorkloadSpec(phase="train", batch=4, tokens=39))
    update = time_ops(optimizer_ops(SPEC), T4).seconds
    expected = narrow * (163 + 1) / 256 + wide * (93 + 255) / 256 + update
    assert step.seconds == pytest.approx(expected, rel=1e-12)
