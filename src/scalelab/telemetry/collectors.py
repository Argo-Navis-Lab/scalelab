"""Collectors: hooks that measure training and record what they find in a metrics sink. Each
reduces what it is shown to a few numbers where the tensors are (the GPU, on a GPU), and
moves them to the CPU in one go."""

import math
from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import Tensor, nn

from scalelab.core.config import AlgoConfig
from scalelab.models.attention.base import Attention
from scalelab.models.transformer import Block, Transformer
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


# m2s32: the groups NumericsCollector counts in, in order. Each group's NaNs and infinities
# are recorded as "numerics/<group>_nan" and "numerics/<group>_inf".
NUMERICS_GROUPS = ("logp", "loss", "grads", "weights")


def nonfinite(tensors: Iterable[Tensor]) -> Tensor:
    """(2,) how many entries of `tensors` (all on one device) are NaN, and how many are
    infinite, of either sign. On that device."""
    counts = [torch.stack([t.isnan().sum(), t.isinf().sum()]) for t in tensors]
    return torch.stack(counts).sum(dim=0)


class NumericsCollector(Hook):
    """Every step's broken numbers, counted by tensor group: NaNs and infinities in the
    trainer's log-probs of the answer tokens and in its losses (every micro-batch of every
    epoch), in the gradients before clipping and in the weights after each update. Counted
    each time they are shown, so recorded when the step ends. Any count above 0 is a
    numerical fault, whatever the algorithm is doing. A step with no update records nothing."""

    def __init__(self, sink: MetricsSink) -> None:
        self.sink = sink
        self.counts: dict[str, list[Tensor]] = {group: [] for group in NUMERICS_GROUPS}

    def on_forward(self, event: ForwardEvent) -> None:
        # Outside the answers the log-probs are 0 (see answer_logprobs): only answers count.
        self.counts["logp"].append(nonfinite([event.logp]))
        self.counts["loss"].append(nonfinite([event.loss]))

    def on_backward(self, event: BackwardEvent) -> None:
        grads = [p.grad for p in event.model.parameters() if p.grad is not None]
        self.counts["grads"].append(nonfinite(grads))

    def on_optimizer_step(self, event: OptimizerStepEvent) -> None:
        self.counts["weights"].append(nonfinite(event.model.parameters()))

    def on_step_end(self, event: StepEndEvent) -> None:
        if self.counts["loss"]:  # the step updated the model
            totals = torch.stack([torch.stack(parts).sum(dim=0) for parts in self.counts.values()])
            for group, (nans, infs) in zip(NUMERICS_GROUPS, totals.tolist(), strict=True):
                self.sink.record(event.step, f"numerics/{group}_nan", nans)
                self.sink.record(event.step, f"numerics/{group}_inf", infs)
        self.counts = {group: [] for group in NUMERICS_GROUPS}


# m2s32: what MismatchCollector records, in order, each named "mismatch/<name>".
MISMATCH_METRICS = ("logp_mean", "logp_max", "ratio_mean", "ratio_max")


class MismatchCollector(Hook):
    """Every step's sampler-trainer mismatch, per answer token: how far the log-prob the
    sampler recorded as it drew the token is from the trainer's, worked out with the same
    weights before the step's first update. Recorded: the mean and the largest absolute
    difference, and the same for the ratio they imply (the trainer's probability over the
    sampler's) minus 1. Over every micro-batch of the first epoch (later epochs read the same
    tokens again), so recorded when the step ends. A step with no update records nothing.

    The same policy worked out twice should agree, but the sampler reads an answer one token
    at a time from its cache while the trainer reads it whole, so the rounding differs: a
    little, in fp32. A sampler that draws from something else, such as top_p < 1, shows as a
    gap far above that."""

    def __init__(self, sink: MetricsSink) -> None:
        self.sink = sink
        self.tokens = 0  # the answer tokens compared
        self.sums: list[Tensor] = []  # one per micro-batch: the two gaps added up
        self.largest: list[Tensor] = []  # one per micro-batch: the two gaps' largest

    def on_forward(self, event: ForwardEvent) -> None:
        if event.epoch > 0:
            return
        batch = event.batch
        # Both are 0 outside the answers, so both gaps are 0 there.
        diff = event.old_logp - batch.sampler_logprobs.to(event.old_logp.device)
        logp_gap = diff.abs()
        ratio_gap = diff.expm1().abs()  # |exp(diff) - 1|, still exact when diff is tiny
        self.tokens += int(batch.response_mask.sum())  # on the CPU: no wait for the GPU
        self.sums.append(torch.stack([logp_gap.sum(), ratio_gap.sum()]))
        self.largest.append(torch.stack([logp_gap.max(), ratio_gap.max()]))

    def on_step_end(self, event: StepEndEvent) -> None:
        if self.sums:
            totals = torch.stack(self.sums).sum(dim=0)
            largest = torch.stack(self.largest).amax(dim=0)
            logp_sum, ratio_sum, logp_max, ratio_max = torch.cat([totals, largest]).tolist()
            measured = {
                "logp_mean": logp_sum / self.tokens,
                "logp_max": logp_max,
                "ratio_mean": ratio_sum / self.tokens,
                "ratio_max": ratio_max,
            }
            for name, value in measured.items():
                self.sink.record(event.step, f"mismatch/{name}", value)
        self.tokens, self.sums, self.largest = 0, [], []


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


# m2s31: what ActivationCollector records about each part of the model, the first from a
# mean, the second from a largest value: the residual stream after layers 0 to n_layers, the
# attention of the blocks (layers 1 to n_layers), and the output (no layer).
STREAM_METRICS = ("activations/rms", "activations/abs_max")
ATTENTION_METRICS = ("attention/entropy", "attention/max_logit")
OUTPUT_METRICS = ("output/z_mean", "output/max_logit")


def summary(mean_of: Tensor, max_of: Tensor, real: Tensor) -> Tensor:
    """(3,): the sum of `mean_of`'s entries, how many there are, and the largest of `max_of`'s,
    counting only the positions `real` (rows, seq) marks. Both are (rows, seq, k). Sums,
    counts and maxima of several micro-batches combine into the whole step's.

    torch.where leaves the other positions out (even an inf or NaN there) without picking
    entries out, which would make the CPU wait for the GPU to count them."""
    inside = real[..., None]  # (rows, seq, 1): the same for each of the k
    return torch.stack(
        [
            torch.where(inside, mean_of, 0).sum(),
            inside.sum() * mean_of.shape[-1],
            torch.where(inside, max_of, -math.inf).amax(),
        ]
    )


class ActivationCollector(Hook):
    """On the steps `cadence` picks, what the forward passes of the step's first update work
    out, at every position whose next token is real (the prompt's and the answer's, not
    PAD's): the RMS and the largest absolute entry of the residual stream after the embedding
    (layer 0) and after each block (layers 1 to n_layers); each block's attention, by the
    mean entropy of its weights (in nats, over heads and positions) and the largest
    pre-softmax logit q.k / sqrt(head_dim); and the output, by the mean log-sum-exp z of the
    logits (what z-loss pushes towards 0) and the largest logit.

    The trainer's pass keeps none of these, so each micro-batch is read again, without
    gradients, through PyTorch forward hooks on the model's modules: each reduces its
    module's output to a few numbers as the pass goes, and all come off after it.
    """

    def __init__(self, sink: MetricsSink, cadence: Cadence) -> None:
        self.sink = sink
        self.cadence = cadence
        self.n_layers = 0
        self.parts: list[Tensor] = []  # one (2 n_layers + 2, 3) per micro-batch, see summary

    def on_forward(self, event: ForwardEvent) -> None:
        if event.epoch > 0 or not self.cadence.due(event.step):
            return
        model, batch = event.model, event.batch
        device = model.embed.weight.device
        tokens = batch.tokens[:, :-1].to(device)  # what the trainer read (see answer_logprobs)
        real = (torch.arange(tokens.shape[1]) < batch.lengths[:, None] - 1).to(device)
        stream: list[Tensor] = []  # after the embedding, then after each block
        attention: list[Tensor] = []
        output: list[Tensor] = []

        def on_stream(module: nn.Module, args: tuple[Any, ...], x: Tensor) -> None:
            stream.append(summary(x.square(), x.abs(), real))

        def on_attention(module: nn.Module, args: tuple[Any, ...], out: Tensor) -> None:
            scores = cast(Attention, module).scores(*args)  # (rows, heads, seq, seq)
            entropy = torch.special.entr(scores.softmax(dim=-1)).sum(dim=-1)  # sum of -p log p
            top = scores.amax(dim=-1)  # each query's largest
            attention.append(summary(entropy.transpose(1, 2), top.transpose(1, 2), real))

        def on_output(module: nn.Module, args: tuple[Any, ...], logits: Tensor) -> None:
            output.append(summary(logits.logsumexp(dim=-1, keepdim=True), logits, real))

        blocks = [cast(Block, block) for block in model.blocks]
        handles = [model.embed.register_forward_hook(on_stream)]
        for block in blocks:
            handles.append(block.attn.register_forward_hook(on_attention))
            handles.append(block.register_forward_hook(on_stream))  # its output is the stream
        handles.append(model.head.register_forward_hook(on_output))
        try:
            with torch.no_grad():
                model(tokens)
        finally:  # even if the pass fails, the model is left as it was
            for handle in handles:
                handle.remove()
        self.n_layers = len(blocks)
        self.parts.append(torch.stack([*stream, *attention, *output]))

    def on_step_end(self, event: StepEndEvent) -> None:
        if self.parts:
            parts = torch.stack(self.parts)  # (micro-batches, 2 n_layers + 2, 3)
            averages = parts[..., 0].sum(dim=0) / parts[..., 1].sum(dim=0)
            stream = slice(0, self.n_layers + 1)
            averages[stream] = averages[stream].sqrt()  # the stream's mean squares: its RMS
            means, maxima = torch.stack([averages, parts[..., 2].amax(dim=0)]).tolist()
            places = [
                *((STREAM_METRICS, layer) for layer in range(self.n_layers + 1)),
                *((ATTENTION_METRICS, layer) for layer in range(1, self.n_layers + 1)),
                (OUTPUT_METRICS, None),
            ]
            for (names, layer), mean, largest in zip(places, means, maxima, strict=True):
                self.sink.record(event.step, names[0], mean, layer)
                self.sink.record(event.step, names[1], largest, layer)
        self.parts = []


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
        NumericsCollector(sink),
        MismatchCollector(sink),
        LayerCollector(sink, cadence),
        AdamCollector(sink, cadence),
        ActivationCollector(sink, cadence),
    ]
