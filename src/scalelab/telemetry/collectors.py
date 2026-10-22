"""Collectors: hooks that measure training and record what they find in a metrics sink. Each
reduces what it is shown to a few numbers where the tensors are (the GPU, on a GPU), and
moves them to the CPU in one go."""

import math

import torch
from torch import Tensor

from scalelab.core.config import AlgoConfig
from scalelab.rl.filters import varied_groups
from scalelab.rl.losses import clip_engaged, kl_k3
from scalelab.telemetry.hooks import ForwardEvent, Hook, RolloutEvent, StepEndEvent
from scalelab.telemetry.sink import MetricsSink

# What RolloutCollector records, in order, each named "rollout/<name>".
ROLLOUT_METRICS = (
    "reward_mean",
    "reward_std",
    "length_mean",
    "length_p95",
    "length_max",
    "truncated",
)


class RolloutCollector(Hook):
    """Every step's answers: the rewards' mean and spread, how many tokens the answers took
    (mean, 95th percentile and longest, <eos> included), and the share that ran out of
    context before <eos>."""

    def __init__(self, sink: MetricsSink) -> None:
        self.sink = sink

    def on_rollout(self, event: RolloutEvent) -> None:
        batch, rewards = event.batch, event.rewards
        written = (batch.lengths - batch.prompt_lengths).float()  # each answer's tokens
        truncated = torch.tensor([reason == "length" for reason in batch.finish_reasons])
        values = torch.stack(
            [
                rewards.mean(),
                rewards.std(correction=0),  # the spread of these rewards themselves
                written.mean(),
                written.quantile(0.95),
                written.max(),
                truncated.float().mean(),
            ]
        ).tolist()
        for name, value in zip(ROLLOUT_METRICS, values, strict=True):
            self.sink.record(event.step, f"rollout/{name}", value)


class PolicyCollector(Hook):
    """Every step's policy. From the answers: its entropy, and the share of groups whose
    answers all scored the same (nothing to learn from them). From the updates, over every
    micro-batch of every epoch, so recorded when the step ends: the ratio's mean and largest
    value, the share of tokens whose gradient clipping took away, the k3 KL to the old policy
    and (with a KL penalty) to the reference, and the advantages' mean and spread over the
    trained answers. A step with no update records only the first two."""

    def __init__(self, sink: MetricsSink, algo: AlgoConfig) -> None:
        self.sink = sink
        self.clip_low, self.clip_high = algo.clip_low, algo.clip_high
        self.sums: list[Tensor] = []  # one per micro-batch, see on_forward
        self.largest: list[Tensor] = []  # each micro-batch's largest ratio
        self.answers = 0  # the answers the micro-batches held
        self.reference = False  # whether the updates had a reference to measure KL against

    def on_rollout(self, event: RolloutEvent) -> None:
        batch = event.batch
        # A sampled token's -log p averages to the entropy of what it was drawn from, so the
        # mean over every answer token estimates the policy's entropy per token for free.
        entropy = -batch.sampler_logprobs.sum() / batch.response_mask.sum()
        flat = 1 - varied_groups(event.rewards, batch.group_ids).float().mean()
        entropy, flat = torch.stack([entropy, flat]).tolist()
        self.sink.record(event.step, "policy/entropy", entropy)
        self.sink.record(event.step, "policy/flat_groups", flat)

    def on_forward(self, event: ForwardEvent) -> None:
        mask = event.mask
        ratio = (event.logp - event.old_logp).exp()
        clipped = clip_engaged(ratio, event.advantages, self.clip_low, self.clip_high) & mask
        kl_old = kl_k3(event.logp, event.old_logp)
        kl_ref = torch.zeros_like(kl_old)
        if event.ref_logp is not None:
            kl_ref = kl_k3(event.logp, event.ref_logp)
            self.reference = True
        # Each epoch sees the same answers again: that changes neither their advantages' mean
        # nor their spread.
        self.answers += len(event.advantages)
        self.sums.append(
            torch.stack(
                [
                    mask.sum(),
                    torch.where(mask, ratio, 0.0).sum(),  # only the tokens the loss counts
                    clipped.sum(),
                    torch.where(mask, kl_old, 0.0).sum(),
                    torch.where(mask, kl_ref, 0.0).sum(),
                    event.advantages.sum(),
                    event.advantages.square().sum(),
                ]
            )
        )
        self.largest.append(torch.where(mask, ratio, 0.0).max())

    def on_step_end(self, event: StepEndEvent) -> None:
        if self.sums:  # none when the filters left nothing to learn from: no update
            totals = torch.stack(self.sums).sum(dim=0)
            largest = torch.stack(self.largest).max()
            values = torch.cat([totals, largest[None]]).tolist()
            tokens, ratio, clipped, kl_old, kl_ref, advantage, square, ratio_max = values
            mean = advantage / self.answers
            measured = {
                "ratio_mean": ratio / tokens,
                "ratio_max": ratio_max,
                "clip_fraction": clipped / tokens,
                "kl_old": kl_old / tokens,
                "advantage_mean": mean,
                "advantage_std": math.sqrt(max(0.0, square / self.answers - mean**2)),
            }
            if self.reference:
                measured["kl_ref"] = kl_ref / tokens
            for name, value in measured.items():
                self.sink.record(event.step, f"policy/{name}", value)
        self.sums, self.largest, self.answers, self.reference = [], [], 0, False


def make_collectors(sink: MetricsSink, algo: AlgoConfig) -> list[Hook]:
    """Every collector, each recording into `sink`."""
    return [RolloutCollector(sink), PolicyCollector(sink, algo)]
