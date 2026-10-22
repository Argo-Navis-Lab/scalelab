"""Collectors: hooks that measure training and record what they find in a metrics sink. Each
reduces what it is shown to a few numbers where the tensors are (the GPU, on a GPU), and
moves them to the CPU in one go."""

import math

import torch
from torch import Tensor

from scalelab.core.config import AlgoConfig
from scalelab.models.transformer import Transformer
from scalelab.rl.filters import varied_groups
from scalelab.rl.losses import clip_engaged, kl_k3
from scalelab.telemetry.hooks import (
    BackwardEvent,
    ForwardEvent,
    Hook,
    OptimizerStepEvent,
    RolloutEvent,
    StepEndEvent,
)
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


# What OptimizationCollector records, in order, each named "optim/<name>".
OPTIMIZATION_METRICS = ("loss", "grad_norm", "grad_clipped")


class OptimizationCollector(Hook):
    """Every step's updates (one per epoch), each metric a mean over them: the loss over the
    whole batch before the update, the gradient's global norm before clipping, and the share
    of updates whose gradient clipping cut down. A step with no update records nothing."""

    def __init__(self, sink: MetricsSink) -> None:
        self.sink = sink
        self.updates: list[tuple[float, float, float]] = []  # loss, norm, 1.0 if clipped

    def on_optimizer_step(self, event: OptimizerStepEvent) -> None:
        clipped = float(event.grad_norm > event.grad_clip)
        self.updates.append((event.loss, event.grad_norm, clipped))

    def on_step_end(self, event: StepEndEvent) -> None:
        if self.updates:
            columns = zip(*self.updates, strict=True)  # the losses, the norms, the clipped
            for name, values in zip(OPTIMIZATION_METRICS, columns, strict=True):
                self.sink.record(event.step, f"optim/{name}", sum(values) / len(values))
        self.updates = []


class SystemCollector(Hook):
    """Every step's cost. Wall-clock seconds, each phase's (see rl_train) and the whole
    step's; answer tokens written per second of generation, and tokens the updates read
    (padding included, every epoch) per second of training; the step's FLOPs by phase, as the
    cost model counts them; and on a GPU the most memory its tensors held at once, in bytes."""

    def __init__(self, sink: MetricsSink) -> None:
        self.sink = sink
        self.written = 0  # the step's answer tokens, <eos> included
        self.read = 0  # the tokens the step's updates read

    def on_rollout(self, event: RolloutEvent) -> None:
        batch = event.batch
        self.written = int((batch.lengths - batch.prompt_lengths).sum())

    def on_forward(self, event: ForwardEvent) -> None:
        rows, width = event.batch.tokens.shape
        self.read += rows * (width - 1)  # the trainer reads every row but its last token

    def on_step_end(self, event: StepEndEvent) -> None:
        measured = {f"{phase}_s": seconds for phase, seconds in event.phases.items()}
        measured["step_s"] = event.seconds
        if self.written:
            measured["sample_tokens_per_s"] = self.written / event.phases["generate"]
        if self.read:  # the step updated the model
            measured["train_tokens_per_s"] = self.read / event.phases["train"]
        measured["sample_flops"] = event.flops.sample
        measured["logprobs_flops"] = event.flops.logprobs
        measured["train_flops"] = event.flops.train
        if event.peak_memory is not None:
            measured["peak_memory"] = event.peak_memory
        for name, value in measured.items():
            self.sink.record(event.step, f"system/{name}", value)
        self.written, self.read = 0, 0


# m2s30: the detailed collectors (a number per layer) measure every DETAIL_EVERY steps, and
# every step for the DETAIL_AFTER_ALERT steps after a detector raises an alert.
DETAIL_EVERY = 10
DETAIL_AFTER_ALERT = 50


class Cadence:
    """Which steps the detailed collectors measure: every `every` steps, and every step for
    the `after_alert` steps after a detector raises an alert. The detailed collectors share
    one Cadence, so they all measure the same steps. A run asks about its steps in order."""

    def __init__(self, every: int = DETAIL_EVERY, after_alert: int = DETAIL_AFTER_ALERT) -> None:
        self.every = every
        self.after_alert = after_alert
        self.until = 0  # the last step an alert asked for; 0 before any alert

    def due(self, step: int) -> bool:
        """Whether the detailed collectors measure `step`."""
        return step % self.every == 0 or step <= self.until

    def alert(self, step: int) -> None:
        """A detector fired at `step`: measure each of the next `after_alert` steps."""
        self.until = max(self.until, step + self.after_alert)


def layers(model: Transformer) -> list[list[Tensor]]:
    """The model's parameters by layer, numbered as the residual stream is: layer 0 is the
    embedding, layers 1 to n_layers are the blocks in order, and layer n_layers + 1 is the
    final norm with the output head."""
    return [
        list(model.embed.parameters()),
        *(list(block.parameters()) for block in model.blocks),
        [*model.norm.parameters(), *model.head.parameters()],
    ]


def layer_norms(tensors: list[list[Tensor]]) -> Tensor:
    """(layers,) each layer's norm over all its tensors' entries: the norm of their norms."""
    norm = torch.linalg.vector_norm
    return torch.stack([norm(torch.stack([norm(t) for t in layer])) for layer in tensors])


class LayerCollector(Hook):
    """On the steps `cadence` picks, each layer's (see layers) part in the updates: the norm
    of its gradient before clipping (the squares of the layers' norms add up to the square
    of the global norm), and the size of the update next to the size of the weights: the RMS
    of the change it made over the RMS of the weights before it. Each is a mean over the
    step's updates."""

    def __init__(self, sink: MetricsSink, cadence: Cadence) -> None:
        self.sink = sink
        self.cadence = cadence
        self.grad_norms: Tensor | None = None  # (layers,) the update in progress's
        self.before: list[list[Tensor]] = []  # a copy of the weights it is about to change
        self.updates: list[Tensor] = []  # one (2, layers) per update: grad norms, ratios

    def on_backward(self, event: BackwardEvent) -> None:
        if not self.cadence.due(event.step):
            return
        weights = layers(event.model)
        grads = [[p.grad for p in layer if p.grad is not None] for layer in weights]
        self.grad_norms = layer_norms(grads)
        self.before = [[p.detach().clone() for p in layer] for layer in weights]

    def on_optimizer_step(self, event: OptimizerStepEvent) -> None:
        if self.grad_norms is None:  # not a step to measure
            return
        changes = [
            [p.detach() - old for p, old in zip(layer, before, strict=True)]
            for layer, before in zip(layers(event.model), self.before, strict=True)
        ]
        # Both RMSs are over the same entries, so their ratio is the ratio of the norms.
        ratios = layer_norms(changes) / layer_norms(self.before)
        self.updates.append(torch.stack([self.grad_norms, ratios]))
        self.grad_norms, self.before = None, []

    def on_step_end(self, event: StepEndEvent) -> None:
        if self.updates:
            grad_norms, ratios = torch.stack(self.updates).mean(dim=0).tolist()
            for layer, (grad_norm, ratio) in enumerate(zip(grad_norms, ratios, strict=True)):
                self.sink.record(event.step, "optim/layer_grad_norm", grad_norm, layer)
                self.sink.record(event.step, "optim/update_ratio", ratio, layer)
        self.updates = []


# What AdamCollector records for each layer, in order, each named "adam/<name>".
ADAM_METRICS = ("v_min", "v_median", "v_max", "eps_dominated")


class AdamCollector(Hook):
    """On the steps `cadence` picks, each layer's (see layers) Adam second moments v after
    the step's last update. Each v is bias-corrected (the v whose square root Adam divides
    by), and only the entries that have had a gradient count (the embedding's rows for
    tokens the task never uses have none): the smallest, median and largest v, and the
    share of entries with sqrt(v) below 10 eps. In those eps, there only so that Adam never
    divides by zero, is more than 1/11 of the denominator sqrt(v) + eps and holds the update
    back. A layer none of whose entries has had a gradient yet records nothing, and neither
    does an optimizer without second moments."""

    def __init__(self, sink: MetricsSink, cadence: Cadence) -> None:
        self.sink = sink
        self.cadence = cadence
        self.measured: Tensor | None = None  # (layers, 4) after the latest update

    def on_optimizer_step(self, event: OptimizerStepEvent) -> None:
        if not self.cadence.due(event.step):
            return
        optimizer = event.optimizer
        group_of = {id(p): group for group in optimizer.param_groups for p in group["params"]}
        rows: list[Tensor] = []
        for layer in layers(event.model):
            values: list[Tensor] = []  # the layer's corrected v, zeros included
            small: list[Tensor] = []  # whether each entry's sqrt(v) is below 10 eps
            for p in layer:
                state, group = optimizer.state.get(p, {}), group_of[id(p)]  # get adds nothing
                if "exp_avg_sq" not in state:  # not Adam
                    return
                beta2, updates = group["betas"][1], float(state["step"])
                corrected = (state["exp_avg_sq"] / (1 - beta2**updates)).flatten()
                values.append(corrected)
                small.append((corrected > 0) & (corrected.sqrt() < 10 * group["eps"]))
            v = torch.cat(values)
            seen = v > 0  # the entries that have had a gradient
            rows.append(
                torch.stack(
                    [
                        torch.where(seen, v, math.inf).min(),
                        torch.where(seen, v, math.nan).nanmedian(),  # NaNs don't count
                        v.max(),
                        torch.cat(small).sum() / seen.sum(),
                    ]
                )
            )
        self.measured = torch.stack(rows)

    def on_step_end(self, event: StepEndEvent) -> None:
        if self.measured is not None:
            for layer, values in enumerate(self.measured.tolist()):
                if values[2] == 0:  # the largest v: no entry has had a gradient yet
                    continue
                for name, value in zip(ADAM_METRICS, values, strict=True):
                    self.sink.record(event.step, f"adam/{name}", value, layer)
        self.measured = None


def make_collectors(
    sink: MetricsSink, algo: AlgoConfig, cadence: Cadence | None = None
) -> list[Hook]:
    """Every collector, each recording into `sink`. The detailed ones measure on the steps
    `cadence` picks: by default every DETAIL_EVERY steps."""
    cadence = cadence or Cadence()
    return [
        RolloutCollector(sink),
        PolicyCollector(sink, algo),
        OptimizationCollector(sink),
        SystemCollector(sink),
        LayerCollector(sink, cadence),
        AdamCollector(sink, cadence),
    ]
