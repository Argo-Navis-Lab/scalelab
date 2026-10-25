"""Timing one operation on a device: warm it up, run it often enough to measure, and keep the
median of several samples."""

import statistics
import time
from collections.abc import Callable, Sequence
from functools import partial

import torch
from pydantic import Field, PositiveFloat
from torch import Tensor

from scalelab.core.config import StrictModel
from scalelab.costmodel.ops import Op, grad_op

WARMUP = 3  # untimed calls first: the first ones pick a kernel, allocate memory, warm caches
REPEATS = 7  # timed samples; the median ignores a slow one (a hiccup elsewhere on the machine)
# A sample repeats the op until it lasts this long, so reading the clock and waiting for the
# GPU (tens of microseconds) add a few percent at most.
MIN_SAMPLE_S = 1e-3


class Timing(StrictModel):
    """Repeated samples of one op's time, each the mean over `number` calls in a row."""

    number: int = Field(gt=0)  # calls per sample
    samples: tuple[PositiveFloat, ...] = Field(min_length=1)  # seconds a call, by sample

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def spread(self) -> float:
        """How much slower the slowest sample was than the fastest: 0.05 is 5%."""
        return max(self.samples) / min(self.samples) - 1


class Measurement(StrictModel):
    """A benchmark's result: the op as the cost model counts it, and how long it took."""

    op: Op
    timing: Timing

    @property
    def seconds(self) -> float:
        """One call's time: the median sample."""
        return self.timing.median


def measure(
    fn: Callable[[], object],
    device: torch.device,
    *,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    min_sample_s: float = MIN_SAMPLE_S,
    clock: Callable[[], float] = time.perf_counter,
) -> Timing:
    """Times fn(): `warmup` untimed calls, then twice as many calls a sample until a sample
    lasts `min_sample_s`, then `repeats` samples of that many calls."""

    def run(number: int) -> float:
        """Seconds for `number` calls in a row. On a GPU, Python only queues the work, so the
        clock is read once the device has finished: before the first call and after the last."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = clock()
        for _ in range(number):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return clock() - start

    for _ in range(warmup):
        fn()
    number = 1
    while run(number) < min_sample_s:
        number *= 2
    samples = tuple(run(number) / number for _ in range(repeats))
    return Timing(number=number, samples=samples)


def measure_forward_backward(
    forward: Callable[[], Tensor],
    inputs: Sequence[Tensor],
    op: Op,
    device: torch.device,
    generator: torch.Generator,
    *,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    min_sample_s: float = MIN_SAMPLE_S,
) -> list[Measurement]:
    """Times forward() recording no graph (as sampling runs it), then, if it has inputs to give
    gradients to, its backward pass alone: the forward runs once, and its gradients are taken
    again and again. The backward's op is the op graph's rule of thumb, grad_op(op)."""
    timed = partial(
        measure, device=device, warmup=warmup, repeats=repeats, min_sample_s=min_sample_s
    )
    with torch.no_grad():
        results = [Measurement(op=op, timing=timed(forward))]
    if inputs:
        out = forward()
        grad = torch.randn(out.shape, generator=generator, device=device, dtype=out.dtype)
        # retain_graph keeps what the forward saved, so the next call can use it again
        backward = partial(torch.autograd.grad, out, inputs, grad, retain_graph=True)
        results.append(Measurement(op=grad_op(op), timing=timed(backward)))
    return results
