"""How long a whole RL step takes, phase by phase: the work the loop does, as op graphs timed on
a device, with answers as long as a model has been seen to write them; and a warm start's step,
which is a train phase alone."""

import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from scalelab.core.config import StrictModel
from scalelab.costmodel.ops import VALUE_BYTES, Op, OpClass, op_graph
from scalelab.costmodel.roofline import Bound, Efficiency, Group, Prediction, time_ops
from scalelab.costmodel.specs import HardwareSpec, ModelSpec, Precision, WorkloadSpec

# The phases a step is timed in, named as telemetry's laps (m2s29). The data and score laps run
# Python on the CPU, a few ms; syncing weights is free while the sampler IS the policy.
Phase = Literal["generate", "logprobs", "train"]


def decode_chunks(prompt_lengths: Sequence[int], batch_size: int) -> list[list[int]]:
    """Which rows generate() decodes together: rows whose prompts have the same length (in
    the order those lengths first appear), at most batch_size at a time. The cost model
    reads the same plan, so it counts the work generate() really did."""
    same_length: dict[int, list[int]] = defaultdict(list)  # prompt length -> those rows
    for row, length in enumerate(prompt_lengths):
        same_length[length].append(row)
    return [
        rows[first : first + batch_size]
        for rows in same_length.values()
        for first in range(0, len(rows), batch_size)
    ]


def longest(rows: Sequence[Sequence[int]]) -> dict[int, float]:
    """The chance that the longest of several rows is k tokens, for every k it can be. Each
    row is one of its own lengths, all equally likely (one listed twice is twice as likely),
    independently of the other rows."""
    ordered = [sorted(lengths) for lengths in rows]
    chances: dict[int, float] = {}
    below = 0.0  # the chance that every row is shorter than k
    for k in sorted({length for lengths in ordered for length in lengths}):
        at_most = math.prod(bisect_right(lengths, k) / len(lengths) for lengths in ordered)
        chances[k] = at_most - below
        below = at_most
    return chances


def expected_max(lengths: Sequence[int], rows: int) -> float:
    """The mean length of the longest of `rows` answers drawn from `lengths`: how long a batch
    of that many rows keeps decoding. One row: the mean; many: close to the longest seen."""
    return sum(k * chance for k, chance in longest([lengths] * rows).items())


def ranked(rows: Sequence[Sequence[int]], ranks: Sequence[int]) -> list[dict[int, float]]:
    """For each rank r, the chance that the r-th shortest of several rows is k tokens, for every
    k it can be (m3s15). Rows as in longest(), which is the rank len(rows) case."""
    ordered = [np.sort(np.asarray(lengths)) for lengths in rows]
    ks = np.unique(np.concatenate(ordered))
    count = np.zeros((len(ks), len(rows) + 1))  # [i, j]: the chance j of the rows are <= ks[i]
    count[:, 0] = 1.0  # before any row
    for lengths in ordered:  # one row more: it is <= ks[i] with chance p, or it isn't
        p = (np.searchsorted(lengths, ks, side="right") / len(lengths))[:, np.newaxis]
        count[:, 1:] = count[:, 1:] * (1 - p) + count[:, :-1] * p
        count[:, :1] *= 1 - p
    chances: list[dict[int, float]] = []
    for rank in ranks:
        at_most = count[:, rank:].sum(axis=1)  # the r-th shortest is <= k: r or more rows are
        exactly = np.diff(at_most, prepend=0.0)
        chances.append(dict(zip(ks.tolist(), exactly.tolist(), strict=True)))
    return chances


def sorted_pieces(
    rows: Sequence[Sequence[int]], micro_batch: int
) -> list[tuple[int, dict[int, float]]]:
    """The padding cut (m3s15): rows sorted by length, then run micro_batch at a time, so each
    piece is padded only to its own longest row, the last one ranked in it. Each piece's rows,
    and the chance its width is k tokens."""
    starts = range(0, len(rows), micro_batch)
    ends = [min(start + micro_batch, len(rows)) for start in starts]
    sizes = [end - start for start, end in zip(starts, ends, strict=True)]
    return list(zip(sizes, ranked(rows, ends), strict=True))


class RLStepSpec(StrictModel):
    """One RL step's work, as the loop does it: its prompts' lengths in order, the lengths of
    the answers seen to prompts of each length (tokens written, <eos> included), and the
    loop's settings."""

    prompt_lengths: tuple[int, ...] = Field(min_length=1)
    answer_lengths: dict[int, tuple[int, ...]]  # prompt length -> the answers seen
    group_size: int = Field(gt=0)
    generation_batch: int = Field(gt=0)  # rows decoded at once (the loop's GENERATION_BATCH)
    micro_batch: int = Field(gt=0)
    epochs: int = Field(gt=0)
    reference: bool  # a KL penalty: the reference's log-probs are a second forward pass
    sort_rows: bool = False  # m3s15: the padding cut (see predict_rl_step)
    precision: Precision = "fp32"  # m3s18

    @model_validator(mode="after")
    def check_answers(self) -> "RLStepSpec":
        for length in self.prompt_lengths:
            if not self.answer_lengths.get(length):
                raise ValueError(f"no answer lengths for prompts of {length} tokens")
        if any(n < 1 for lengths in self.answer_lengths.values() for n in lengths):
            raise ValueError("an answer has at least one token, its <eos>")
        return self

    @property
    def rows(self) -> list[int]:
        """Each row's prompt length, in the loop's order: group_size rows a prompt."""
        return [length for length in self.prompt_lengths for _ in range(self.group_size)]


class PhaseTime(StrictModel):
    """One phase's expected time, split by what bounds it and by op class, and its FLOPs."""

    by_bound: dict[Bound, float]
    by_class: dict[OpClass, float]
    flops: float  # expected matmul FLOPs, as count_flops counts them

    @property
    def seconds(self) -> float:
        return sum(self.by_bound.values())

    @property
    def bottleneck(self) -> Bound:
        return max(self.by_bound, key=lambda bound: self.by_bound[bound])


class StepPrediction(StrictModel):
    """A whole step's expected time on one device, phase by phase (a warm start's: train)."""

    hardware: str
    phases: dict[Phase, PhaseTime]

    @property
    def seconds(self) -> float:
        return sum(phase.seconds for phase in self.phases.values())

    @property
    def flops(self) -> float:
        """Its expected matmul FLOPs: what the loop counts against the budget (m3s9)."""
        return sum(phase.flops for phase in self.phases.values())

    @property
    def slowest(self) -> Phase:
        """The phase that takes longest: where a speed-up pays most."""
        return max(self.phases, key=lambda phase: self.phases[phase].seconds)


class _Tally:
    """Timed op graphs added up, each weighted by how often it runs (or its chance to)."""

    def __init__(self) -> None:
        self.by_bound: dict[Bound, float] = {"compute": 0.0, "memory": 0.0, "launch": 0.0}
        self.by_class: dict[OpClass, float] = {}
        self.flops = 0.0

    def add(self, prediction: Prediction, weight: float = 1.0) -> None:
        for bound, seconds in prediction.by_bound.items():
            self.by_bound[bound] += weight * seconds
        for op_class, share in prediction.by_class.items():
            self.by_class[op_class] = self.by_class.get(op_class, 0.0) + weight * share.seconds
        self.flops += weight * sum(t.op.flops for t in prediction.times)

    def phase(self) -> PhaseTime:
        return PhaseTime(by_bound=self.by_bound, by_class=self.by_class, flops=self.flops)


def optimizer_ops(model: ModelSpec, precision: Precision = "fp32") -> list[Op]:
    """What an update does once, after every micro-batch's backward pass (m3s1 left it to the
    step): clip_grad_norm_ reads every gradient for the norm, then scales them all; AdamW
    reads each weight, its gradient and both moments, and writes the weight and moments.
    zero_grad only drops the gradients: no kernel. Filed under layer 0: they cover all. Under
    fp16 (m3s18) GradScaler unscales every gradient first, checking for inf and NaN."""
    values = model.param_counts().total * VALUE_BYTES  # one number for every weight
    sizes = {"optim.clip_norm": (1, 0), "optim.clip_scale": (1, 1), "optim.adamw": (4, 3)}
    if precision == "fp16":
        sizes = {"optim.unscale": (1, 1)} | sizes
    return [
        Op(
            name=name,
            layer=0,
            op_class="elementwise",
            flops=0,
            bytes_read=read * values,
            bytes_written=written * values,
        )
        for name, (read, written) in sizes.items()
    ]


def generation_time(
    model: ModelSpec,
    rows: Sequence[int],
    answer_lengths: Mapping[int, Sequence[int]],
    batch_size: int,
    hardware: HardwareSpec,
    efficiencies: Mapping[Group, Efficiency] | None = None,
    precision: Precision = "fp32",
) -> PhaseTime:
    """generate()'s expected time for rows with these prompt lengths, each answer one of the
    lengths seen after prompts of its length: rows decode in decode_chunks' plan, and a chunk
    keeps decoding until its longest answer is written (m4s3: moved out of predict_rl_step,
    so gradings are priced the same way)."""
    timed: dict[WorkloadSpec, Prediction] = {}
    p = precision

    def run(workload: WorkloadSpec) -> Prediction:  # each shape is timed once
        if workload not in timed:
            timed[workload] = time_ops(op_graph(model, workload), hardware, efficiencies)
        return timed[workload]

    generate = _Tally()
    for chunk in decode_chunks(rows, batch_size):
        n, prompt = len(chunk), rows[chunk[0]]
        generate.add(run(WorkloadSpec(phase="prefill", batch=n, tokens=prompt, precision=p)))
        chances = longest([answer_lengths[prompt]] * n)
        running = 1.0  # the chance the longest answer is longer than j + 1 tokens
        for j in range(max(chances) - 1):  # the last token is never fed back
            running -= chances.get(j + 1, 0.0)
            decode = WorkloadSpec(phase="decode", batch=n, past=prompt + j, tokens=1, precision=p)
            generate.add(run(decode), running)
    return generate.phase()


def predict_rl_step(
    model: ModelSpec,
    step: RLStepSpec,
    hardware: HardwareSpec,
    efficiencies: Mapping[Group, Efficiency] | None = None,
) -> StepPrediction:
    """A step's expected time, phase by phase. A chunk keeps decoding until its longest answer
    is written, so each decode position counts with the chance the chunk still runs there:
    the expected MAXIMUM length sets generation's time, not the mean. Log-probs and training
    run on rows padded to their longest prompt + answer, weighted the same way: log-probs in
    one pass over every row, training micro_batch rows at a time as they come. Under the
    padding cut (sort_rows, m3s15) the rows are sorted by length first, and both run in
    micro-batches. Under fp16 (m3s18) every op graph is autocast's."""
    timed: dict[WorkloadSpec, Prediction] = {}
    p = step.precision

    def run(workload: WorkloadSpec) -> Prediction:  # each shape is timed once
        if workload not in timed:
            timed[workload] = time_ops(op_graph(model, workload), hardware, efficiencies)
        return timed[workload]

    def passes(
        tally: _Tally,
        phase: Literal["prefill", "train"],
        pieces: Sequence[tuple[int, Mapping[int, float]]],
        weight: int,
    ) -> None:
        """Forward (or forward and backward) passes over pieces of rows, each its number of
        rows and the chances of its width (its longest prompt + answer), every row read but
        its last token."""
        for size, widths in pieces:
            for width, chance in widths.items():
                if chance > 0:  # widths it can't be need no timing
                    workload = WorkloadSpec(phase=phase, batch=size, tokens=width - 1, precision=p)
                    tally.add(run(workload), weight * chance)

    rows = step.rows
    generate = generation_time(
        model, rows, step.answer_lengths, step.generation_batch, hardware, efficiencies, p
    )

    totals = [[length + n for n in step.answer_lengths[length]] for length in rows]
    if step.sort_rows:  # m3s15: the padding cut
        logprob_pieces = train_pieces = sorted_pieces(totals, step.micro_batch)
    else:
        logprob_pieces = [(len(totals), longest(totals))]  # one pass, all rows
        parts = [totals[i : i + step.micro_batch] for i in range(0, len(totals), step.micro_batch)]
        train_pieces = [(len(part), longest(part)) for part in parts]

    logprobs = _Tally()
    passes(logprobs, "prefill", logprob_pieces, weight=2 if step.reference else 1)

    train = _Tally()
    passes(train, "train", train_pieces, weight=step.epochs)
    train.add(time_ops(optimizer_ops(model, p), hardware, efficiencies), step.epochs)

    phases: dict[Phase, PhaseTime] = {
        "generate": generate,
        "logprobs": logprobs.phase(),
        "train": train.phase(),
    }
    return StepPrediction(hardware=hardware.name, phases=phases)


class WarmStartStepSpec(StrictModel):
    """One warm-start step's work (m3s11): batch_size worked examples side by side, each as many
    tokens long (prompt, answer, <eos>) as one of `lengths`, all equally likely."""

    lengths: tuple[int, ...] = Field(min_length=1)
    batch_size: int = Field(gt=0)
    micro_batch: int | None = Field(default=None, gt=0)  # m3s15: the padding cut
    precision: Precision = "fp32"  # m3s18

    @model_validator(mode="after")
    def check_lengths(self) -> "WarmStartStepSpec":
        if min(self.lengths) < 2:
            raise ValueError("an example has at least 2 tokens: one read, the next predicted")
        return self


def predict_warm_start_step(
    model: ModelSpec,
    step: WarmStartStepSpec,
    hardware: HardwareSpec,
    efficiencies: Mapping[Group, Efficiency] | None = None,
) -> StepPrediction:
    """A warm-start step's expected time: one forward and backward pass over the batch padded to
    its longest example, every token read but the last (each width weighted by its chance, as
    RL's training is), then one update. Drawing the problems is CPU time the model leaves out.
    Under the padding cut (micro_batch, m3s15) the batch is sorted by length and run in pieces
    of micro_batch rows, each padded only to its own longest example, then the one update."""

    timed: dict[WorkloadSpec, Prediction] = {}

    def run(workload: WorkloadSpec) -> Prediction:  # m3s15: pieces of one size share widths
        if workload not in timed:
            timed[workload] = time_ops(op_graph(model, workload), hardware, efficiencies)
        return timed[workload]

    p = step.precision  # m3s18
    rows = [step.lengths] * step.batch_size
    if step.micro_batch is None:
        pieces = [(step.batch_size, longest(rows))]
    else:  # m3s15: the padding cut
        pieces = sorted_pieces(rows, step.micro_batch)
    train = _Tally()
    for size, widths in pieces:
        for width, chance in widths.items():
            if chance > 0:  # widths it can't be need no timing
                workload = WorkloadSpec(phase="train", batch=size, tokens=width - 1, precision=p)
                train.add(run(workload), chance)
    train.add(time_ops(optimizer_ops(model, p), hardware, efficiencies))
    return StepPrediction(hardware=hardware.name, phases={"train": train.phase()})
