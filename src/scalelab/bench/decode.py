"""The decode-step benchmark: whole generation steps, one new token for every sequence of a
batch, at several cache lengths. A step is dozens of small kernels and the gaps between them,
so the cost model predicts it as the sum of its ops."""

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import NamedTuple

import torch
from torch import Tensor

from scalelab.bench.attention import BATCHES, LENGTHS
from scalelab.bench.timing import MIN_SAMPLE_S, REPEATS, WARMUP, Timing, measure
from scalelab.core.config import ModelConfig, StrictModel
from scalelab.costmodel.ops import Op, op_graph
from scalelab.costmodel.specs import ModelSpec, Precision, WorkloadSpec
from scalelab.models.kvcache import KVCache
from scalelab.models.precision import autocast
from scalelab.models.transformer import Transformer
from scalelab.tasks.tokenizer import VOCAB_SIZE


class DecodeCase(NamedTuple):
    """A step of `model` generating a token for `batch` sequences, `past` tokens in."""

    model: ModelConfig
    batch: int
    past: int


class StepMeasurement(StrictModel):
    """A whole step, timed, and what the cost model needs to predict it."""

    model: str  # the config's name ("m1")
    spec: ModelSpec
    workload: WorkloadSpec
    timing: Timing

    @property
    def ops(self) -> list[Op]:
        return op_graph(self.spec, self.workload)

    @property
    def seconds(self) -> float:
        """One step's time: the median sample."""
        return self.timing.median


def decode_cases(
    models: Iterable[ModelConfig],
    batches: Sequence[int] = BATCHES,
    lengths: Sequence[int] = LENGTHS,
) -> list[DecodeCase]:
    """For each model and batch, the step that generates a sequence's last token at each
    length, with the length - 1 tokens before it in the cache."""
    return [DecodeCase(model, b, n - 1) for model in models for b in batches for n in lengths]


def _step(model: Transformer, cache: KVCache, tokens: Tensor, past: int) -> Tensor:
    """Decodes the token at position `past`. decode() moves the cache's length on by one, so it
    is moved back first: every call repeats the same step."""
    cache.length = past
    return model.decode(tokens, cache)


def _decode_step(
    case: DecodeCase, device: torch.device, generator: torch.Generator
) -> Callable[[], Tensor]:
    """The step on random weights, random tokens and a cache of zeros: its speed doesn't depend
    on the numbers. Once timed, nothing holds the model or the cache, so each is freed before
    the next case's is allocated: a GPU holds one cache at a time."""
    config, batch, past = case
    with torch.random.fork_rng(devices=[]):  # the caller's random numbers stay as they were
        torch.manual_seed(0)
        model = Transformer(config).to(device)
    tokens = torch.randint(VOCAB_SIZE, (batch, 1), generator=generator, device=device)
    return partial(_step, model, model.new_cache(batch), tokens, past)


def bench_decode(
    cases: Iterable[DecodeCase],
    device: torch.device,
    *,
    precision: Precision = "fp32",
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    min_sample_s: float = MIN_SAMPLE_S,
) -> list[StepMeasurement]:
    """Times Transformer.decode recording no graph, as the sampler runs it. In fp16 (m3s19)
    each case is one autocast region, as a generate call will be: the cache is float16, and
    the weights are cast once, while warming up."""
    generator = torch.Generator(device).manual_seed(0)
    results: list[StepMeasurement] = []
    for case in cases:
        # the step is only an argument: freed as soon as it is timed
        with torch.no_grad(), autocast(precision, device):
            timing = measure(
                _decode_step(case, device, generator),
                device,
                warmup=warmup,
                repeats=repeats,
                min_sample_s=min_sample_s,
            )
        workload = WorkloadSpec(
            phase="decode", batch=case.batch, tokens=1, past=case.past, precision=precision
        )
        spec = ModelSpec.from_config(case.model)
        results.append(
            StepMeasurement(model=case.model.name, spec=spec, workload=workload, timing=timing)
        )
    return results
