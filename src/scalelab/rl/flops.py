"""What an RL step costs, in FLOPs: B describes the work it did and the cost model (A) counts
it. B never counts FLOPs itself."""

from scalelab.core.config import StrictModel
from scalelab.costmodel.ops import NO_FLOPS, FlopCounts, count_flops
from scalelab.costmodel.specs import ModelSpec, WorkloadSpec
from scalelab.rl.rollout import RolloutBatch, decode_chunks


class StepFlops(StrictModel):
    """One RL step's FLOPs, by phase."""

    sample: int  # generating the answers
    logprobs: int  # the old (and reference) log-probs: forward passes only
    train: int  # the updates: forward and backward

    @property
    def total(self) -> int:
        return self.sample + self.logprobs + self.train


def sampling_flops(model: ModelSpec, batch: RolloutBatch, batch_size: int) -> FlopCounts:
    """Generating `batch` as generate() did, `batch_size` rows at most at a time: each chunk
    reads its prompts in one pass, then adds one token at a time until its longest answer
    is written. Rows that finished earlier are still computed, so they are counted."""
    lengths, prompt_lengths = batch.lengths.tolist(), batch.prompt_lengths.tolist()
    total = NO_FLOPS
    for chunk in decode_chunks(prompt_lengths, batch_size):
        rows, prompt = len(chunk), prompt_lengths[chunk[0]]
        written = max(lengths[row] - prompt for row in chunk)  # the chunk's longest answer
        total += count_flops(model, WorkloadSpec(phase="prefill", batch=rows, tokens=prompt))
        for past in range(prompt, prompt + written - 1):  # the last token is never fed back
            total += count_flops(
                model, WorkloadSpec(phase="decode", batch=rows, tokens=1, past=past)
            )
    return total


def logprob_flops(model: ModelSpec, batch: RolloutBatch) -> FlopCounts:
    """One forward pass over the whole batch, every row read but its last token: how the
    trainer works out log-probs (rl_update's old and reference log-probs)."""
    rows, width = batch.tokens.shape
    return count_flops(model, WorkloadSpec(phase="prefill", batch=rows, tokens=width - 1))


def training_flops(
    model: ModelSpec, batch: RolloutBatch, micro_batch: int, epochs: int
) -> FlopCounts:
    """`epochs` passes over the batch in micro-batches, forward and backward, as rl_update
    trains: each micro-batch cut to its own width."""
    total = NO_FLOPS
    for _ in range(epochs):
        for start in range(0, len(batch.finish_reasons), micro_batch):
            rows, width = batch.rows(start, start + micro_batch).tokens.shape
            workload = WorkloadSpec(phase="train", batch=rows, tokens=width - 1)
            total += count_flops(model, workload)
    return total


def rl_step_flops(
    model: ModelSpec,
    generated: RolloutBatch,
    trained: RolloutBatch | None,
    *,
    batch_size: int,
    micro_batch: int,
    epochs: int,
    reference: bool,
) -> StepFlops:
    """The FLOPs of a whole RL step: generating `generated`, then (unless nothing was left
    to train on: trained None) the old log-probs, the reference's when there is one, and
    the updates on `trained`, the batch that remained after the filters."""
    sample = sampling_flops(model, generated, batch_size).total
    if trained is None:
        return StepFlops(sample=sample, logprobs=0, train=0)
    passes = 2 if reference else 1  # the old log-probs, and the reference's
    return StepFlops(
        sample=sample,
        logprobs=passes * logprob_flops(model, trained).total,
        train=training_flops(model, trained, micro_batch, epochs).total,
    )
