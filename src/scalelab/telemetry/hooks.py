"""The hook points the RL trainer exposes. Whatever measures training (a collector) is a Hook
that overrides the points it needs. The trainer calls every hook it is given at each point,
and never knows what they do."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from scalelab.models.transformer import Transformer
from scalelab.rl.flops import StepFlops
from scalelab.rl.rollout import RolloutBatch


@dataclass(frozen=True)
class RolloutEvent:
    """A step's answers, sampled and scored, before the filters."""

    step: int  # the step they are for; steps count from 1
    batch: RolloutBatch  # every answer generated
    rewards: Tensor  # (rows,) the rewards the advantages come from, penalties included


@dataclass(frozen=True)
class ForwardEvent:
    """One micro-batch's forward pass and loss, before its backward pass."""

    step: int
    epoch: int  # which of the step's updates: 0 for the first
    model: Transformer  # m2s31: the weights the pass used
    batch: RolloutBatch  # the micro-batch's rows
    logp: Tensor  # (rows, width) the trainer's log-probs of the answer tokens
    old_logp: Tensor  # the same before the step's first update: where the ratio starts
    ref_logp: Tensor | None  # the reference's, when there is a KL penalty
    advantages: Tensor  # (rows,)
    mask: Tensor  # (rows, width) the tokens the loss counts
    loss: Tensor  # the micro-batch's loss, a single number


@dataclass(frozen=True)
class BackwardEvent:
    """An update's gradients, complete (every micro-batch's added up) and not yet clipped."""

    step: int
    epoch: int
    model: Transformer  # each parameter's .grad holds its gradient


@dataclass(frozen=True)
class OptimizerStepEvent:
    """An update, done: the optimizer has changed the weights."""

    step: int
    epoch: int
    model: Transformer
    optimizer: torch.optim.Optimizer
    loss: float  # the update's loss over the whole batch, before the step
    grad_norm: float  # the gradient's size before clipping
    grad_clip: float  # the size clipping cuts a bigger gradient down to


@dataclass(frozen=True)
class StepEndEvent:
    """A step is over: its answers were scored and its updates, if any, are done."""

    step: int
    seconds: float  # the step's wall-clock time so far
    phases: Mapping[str, float]  # seconds by phase, see rl_train
    flops: StepFlops  # what the step cost, as the cost model counts it
    peak_memory: int | None  # the most bytes its tensors held at once on a GPU; None on a CPU


@dataclass(frozen=True)
class CheckpointEvent:
    """A checkpoint of this step is about to be saved. A resumed run starts after it, so
    whatever a hook keeps about the steps so far must be on disk before it is."""

    step: int


class Hook:
    """Something that watches training. Each method is one hook point and does nothing here:
    a collector overrides the ones it needs.

    Hooks look and never touch. They must not change the tensors, model or optimizer they
    are shown, and should reduce what they need to a few numbers during the call rather
    than keep the event's tensors.
    """

    def on_rollout(self, event: RolloutEvent) -> None:
        """A step's answers have been sampled and scored."""

    def on_forward(self, event: ForwardEvent) -> None:
        """A micro-batch's loss has been computed."""

    def on_backward(self, event: BackwardEvent) -> None:
        """An update's gradients are complete."""

    def on_optimizer_step(self, event: OptimizerStepEvent) -> None:
        """An update has changed the weights."""

    def on_step_end(self, event: StepEndEvent) -> None:
        """A step is over."""

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        """A checkpoint is about to be saved."""
