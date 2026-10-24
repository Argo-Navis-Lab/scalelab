"""How long a whole RL step takes, phase by phase: the work the loop does, as op graphs timed on
a device, with answers as long as a model has been seen to write them."""

import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from scalelab.core.config import StrictModel
from scalelab.costmodel.ops import VALUE_BYTES, Op, OpClass, op_graph
from scalelab.costmodel.roofline import Bound, Efficiency, Group, Prediction, time_ops
from scalelab.costmodel.specs import HardwareSpec, ModelSpec, WorkloadSpec

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
    """A whole RL step's expected time on one device, phase by phase."""

    hardware: str
    phases: dict[Phase, PhaseTime]

    @property
    def seconds(self) -> float:
        return sum(phase.seconds for phase in self.phases.values())

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


def optimizer_ops(model: ModelSpec) -> list[Op]:
    """What an update does once, after every micro-batch's backward pass (m3s1 left it to the
    step): clip_grad_norm_ reads every gradient for the norm, then scales them all; AdamW
    reads each weight, its gradient and both moments, and writes the weight and moments.
    zero_grad only drops the gradients: no kernel. Filed under layer 0: they cover all."""
    values = model.param_counts().total * VALUE_BYTES  # one number for every weight
    sizes = {"optim.clip_norm": (1, 0), "optim.clip_scale": (1, 1), "optim.adamw": (4, 3)}
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


def predict_rl_step(
    model: ModelSpec,
    step: RLStepSpec,
    hardware: HardwareSpec,
    efficiencies: Mapping[Group, Efficiency] | None = None,
) -> StepPrediction:
    """A step's expected time, phase by phase. A chunk keeps decoding until its longest answer
    is written, so each decode position counts with the chance the chunk still runs there:
    the expected MAXIMUM length sets generation's time, not the mean. Log-probs and training
    run on rows padded to their longest prompt + answer, weighted the same way."""
    timed: dict[WorkloadSpec, Prediction] = {}

    def run(workload: WorkloadSpec) -> Prediction:  # each shape is timed once
        if workload not in timed:
            timed[workload] = time_ops(op_graph(model, workload), hardware, efficiencies)
        return timed[workload]

    def passes(
        tally: _Tally, phase: Literal["prefill", "train"], rows: list[int], weight: int
    ) -> None:
        """Forward (or forward and backward) passes over these rows, padded to the longest
        prompt + answer, every row read but its last token."""
        totals = [[length + n for n in step.answer_lengths[length]] for length in rows]
        for width, chance in longest(totals).items():
            if chance > 0:  # widths it can't be need no timing
                workload = WorkloadSpec(phase=phase, batch=len(rows), tokens=width - 1)
                tally.add(run(workload), weight * chance)

    rows = step.rows
    generate = _Tally()
    for chunk in decode_chunks(rows, step.generation_batch):
        n, prompt = len(chunk), rows[chunk[0]]
        generate.add(run(WorkloadSpec(phase="prefill", batch=n, tokens=prompt)))
        chances = longest([step.answer_lengths[prompt]] * n)
        running = 1.0  # the chance the longest answer is longer than j + 1 tokens
        for j in range(max(chances) - 1):  # the last token is never fed back
            running -= chances.get(j + 1, 0.0)
            generate.add(
                run(WorkloadSpec(phase="decode", batch=n, past=prompt + j, tokens=1)), running
            )

    logprobs = _Tally()
    passes(logprobs, "prefill", rows, weight=2 if step.reference else 1)  # one pass, all rows

    train = _Tally()
    for start in range(0, len(rows), step.micro_batch):
        passes(train, "train", rows[start : start + step.micro_batch], weight=step.epochs)
    train.add(time_ops(optimizer_ops(model), hardware, efficiencies), step.epochs)

    phases: dict[Phase, PhaseTime] = {
        "generate": generate.phase(),
        "logprobs": logprobs.phase(),
        "train": train.phase(),
    }
    return StepPrediction(hardware=hardware.name, phases=phases)
