"""The communication model: how long a collective takes between devices in the alpha-beta form
(a fixed latency a message, plus the bytes over the link's bandwidth), and what data, tensor and
pipeline parallelism and RL's weight sync cost with it."""

from collections.abc import Sequence
from typing import Literal, get_args

import numpy as np
from pydantic import Field, PositiveFloat
from scipy.optimize import nnls

from scalelab.core.config import StrictModel

Collective = Literal["all_reduce", "all_gather", "reduce_scatter"]
COLLECTIVES: tuple[Collective, ...] = get_args(Collective)


class Link(StrictModel):
    """What connects the devices, as the formulas see it."""

    alpha: float = Field(ge=0)  # seconds a message: the fixed cost of one step of the ring
    beta: PositiveFloat  # bytes per second a device sends


def terms(collective: Collective, world: int) -> tuple[float, int]:
    """(the share of the bytes each device sends, the messages in a row) for `world` devices in
    a ring. All-gather and reduce-scatter pass p - 1 pieces of M / p round the ring; all-reduce
    is a reduce-scatter and then an all-gather, so twice both."""
    rounds = 2 if collective == "all_reduce" else 1
    return rounds * (world - 1) / world, rounds * (world - 1)


def collective_time(collective: Collective, nbytes: int, world: int, link: Link) -> float:
    """Seconds for one collective over a tensor of `nbytes`: all-reduce's buffer, all-gather's
    output, reduce-scatter's input. One device alone sends nothing: 0."""
    share, messages = terms(collective, world)
    return share * nbytes / link.beta + messages * link.alpha


def fit_link(collective: Collective, world: int, points: Sequence[tuple[int, float]]) -> Link:
    """The link that best explains measured (nbytes, seconds) of one collective at one world
    size: least squares on the relative error, so a 4 KB message counts as much as a 64 MB one.
    Time is linear in (1 / beta, alpha), so it is a linear fit, kept non-negative (nnls)."""
    share, messages = terms(collective, world)
    nbytes = np.array([n for n, _ in points], dtype=np.float64)
    seconds = np.array([s for _, s in points], dtype=np.float64)
    # predicted / measured = share·n/s · (1/beta) + messages/s · alpha, which should be 1
    design = np.stack([share * nbytes / seconds, messages / seconds], axis=1)
    (inverse_beta, alpha), _ = nnls(design, np.ones(len(points)))
    if inverse_beta <= 0:
        raise ValueError(f"{collective}'s times don't grow with the message size: no bandwidth")
    return Link(alpha=float(alpha), beta=float(1 / inverse_beta))


def data_parallel_time(grad_bytes: int, world: int, link: Link, overlap: float = 0.0) -> float:
    """Data parallel: one all-reduce of the gradients a step, `overlap` of it hidden behind the
    backward pass (0: none, 1: all)."""
    return (1 - overlap) * collective_time("all_reduce", grad_bytes, world, link)


def tensor_parallel_time(
    n_layers: int, activation_bytes: int, world: int, link: Link, *, backward: bool = True
) -> float:
    """Tensor parallel (Megatron style): 2 all-reduces of a layer's activations in the forward
    pass (after attention, after the MLP), 2 more in the backward pass."""
    per_layer = 4 if backward else 2
    return n_layers * per_layer * collective_time("all_reduce", activation_bytes, world, link)


def pipeline_bubble(stages: int, microbatches: int) -> float:
    """Pipeline parallel: the share of a step each stage sits idle, (p - 1) / (m + p - 1)."""
    return (stages - 1) / (microbatches + stages - 1)


def weight_sync_time(param_bytes: int, link: Link, *, shared: bool) -> float:
    """RL: sending the trainer's updated weights to the sampler, Φ·b / β, or nothing when the
    two share the weights in memory (every run so far: one process, one device)."""
    return 0.0 if shared else param_bytes / link.beta
