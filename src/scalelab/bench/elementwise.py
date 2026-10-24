"""The memory benchmarks: copies of growing size, and the models' own elementwise ops (norms,
RoPE, the MLP's activation, residual adds) at the sizes a training step runs them."""

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from scalelab.bench.gemm import ROWS
from scalelab.bench.timing import (
    MIN_SAMPLE_S,
    REPEATS,
    WARMUP,
    Measurement,
    measure,
    measure_forward_backward,
)
from scalelab.costmodel.ops import VALUE_BYTES, Op
from scalelab.costmodel.specs import ModelSpec
from scalelab.models.norms import RMSNorm
from scalelab.models.rope import RotaryEmbedding

# From 16 KB, so small that starting the kernel takes longer than moving the bytes, to 256 MB,
# far past any GPU's on-chip cache (the T4's L2 holds 4 MB): 4096 to 67M values, ×4 a size.
COPY_SIZES = tuple(4**k for k in range(6, 14))
TOKENS = 128  # tokens a sequence: RoPE reads a cos and a sin for each position

Kind = Literal["norm", "rope", "swiglu", "gelu", "residual"]


class Case(NamedTuple):
    """One elementwise op at one size: it reads `rows` vectors of `width` values."""

    kind: Kind
    rows: int
    width: int
    head_dim: int = 0  # RoPE only: it turns each head's vector on its own


def copy_op(values: int) -> Op:
    """Copying `values` numbers: each read once and written once, with no arithmetic."""
    size = VALUE_BYTES * values
    return Op(
        name=f"copy {values}",
        layer=0,
        op_class="elementwise",
        flops=0,
        bytes_read=size,
        bytes_written=size,
    )


def elementwise_cases(models: Iterable[ModelSpec], rows: Sequence[int] = ROWS) -> list[Case]:
    """The elementwise ops of a training step of these models, each size once, in first-seen
    order: every forward op op_graph counts but the embedding lookup (grad_op gives their
    gradients)."""
    cases: dict[Case, None] = {}  # a dict keeps the order; a set wouldn't
    for model in models:
        d, d_head = model.d_model, model.head_dim
        q_width, kv_width = model.n_heads * d_head, model.n_kv_heads * d_head
        act: Kind = "swiglu" if model.ffn == "swiglu" else "gelu"
        for r in rows:
            found = [Case("norm", r, d), Case(act, r, model.d_ff), Case("residual", r, d)]
            for width in (q_width, kv_width):
                if model.qk_norm:  # a norm over each head's vector: rows × heads vectors
                    found.append(Case("norm", r * width // d_head, d_head))
                found.append(Case("rope", r, width, d_head))
            cases.update(dict.fromkeys(found))
    return list(cases)


def case_op(case: Case) -> Op:
    """One benchmarked op as op_graph counts it (layer 0: it is in no model)."""
    kind, rows, width, head_dim = case
    values = VALUE_BYTES * rows * width  # the bytes of one input, and of the output
    extra = {  # what it reads besides that input
        "norm": VALUE_BYTES * width,  # the weight
        "rope": VALUE_BYTES * 2 * TOKENS * head_dim,  # each position's cos and sin
        "swiglu": values,  # silu(gate) * up: a second input
        "gelu": 0,
        "residual": values,  # x + the branch's output
    }[kind]
    return Op(
        name=f"{kind} {rows}x{width}",
        layer=0,
        op_class="elementwise",
        flops=0,
        bytes_read=values + extra,
        bytes_written=values,
    )


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """What the SwiGLU MLP computes between its matmuls."""
    return F.silu(gate) * up


def _kernel(
    case: Case, device: torch.device, generator: torch.Generator
) -> tuple[Callable[[], Tensor], list[Tensor]]:
    """The op as the models run it, on random inputs, and the tensors its backward pass gives
    gradients to: none for an add, whose gradient is its output's, passed on with no kernel."""
    kind, rows, width, head_dim = case

    def randn(*shape: int) -> Tensor:
        return torch.randn(shape, generator=generator, device=device, requires_grad=True)

    if kind == "norm":
        norm = RMSNorm(width).to(device)
        x = randn(rows, width)
        return partial(norm, x), [x, norm.weight]
    if kind == "rope":
        rope = RotaryEmbedding(head_dim, TOKENS).to(device)
        x = randn(rows // TOKENS, width // head_dim, TOKENS, head_dim)  # (batch, heads, seq, dim)
        return partial(rope, x, torch.arange(TOKENS, device=device)), [x]
    if kind == "residual":
        return partial(torch.add, randn(rows, width), randn(rows, width)), []
    up = randn(rows, width)
    if kind == "gelu":
        return partial(F.gelu, up), [up]
    gate = randn(rows, width)
    return partial(swiglu, gate, up), [gate, up]


def bench_copy(
    sizes: Iterable[int],
    device: torch.device,
    *,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    min_sample_s: float = MIN_SAMPLE_S,
) -> list[Measurement]:
    """Times copying each many float32 values from one buffer into another."""
    generator = torch.Generator(device).manual_seed(0)
    results: list[Measurement] = []
    for values in sizes:
        source = torch.randn(values, generator=generator, device=device)
        target = torch.empty_like(source)
        timing = measure(
            partial(target.copy_, source),
            device,
            warmup=warmup,
            repeats=repeats,
            min_sample_s=min_sample_s,
        )
        results.append(Measurement(op=copy_op(values), timing=timing))
    return results


def bench_elementwise(
    cases: Iterable[Case],
    device: torch.device,
    *,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    min_sample_s: float = MIN_SAMPLE_S,
) -> list[Measurement]:
    """Times each op forward, then its backward pass alone, when it has one."""
    generator = torch.Generator(device).manual_seed(0)
    results: list[Measurement] = []
    for case in cases:
        forward, inputs = _kernel(case, device, generator)
        results += measure_forward_backward(
            forward,
            inputs,
            case_op(case),
            device,
            generator,
            warmup=warmup,
            repeats=repeats,
            min_sample_s=min_sample_s,
        )
    return results

    """Times each op forward, then its backward pass alone, when it has one."""
    generator = torch.Generator(device).manual_seed(0)
    results: list[Measurement] = []
    for case in cases:
        forward, inputs = _kernel(case, device, generator)
        results += measure_forward_backward(
            forward,
            inputs,
            case_op(case),
            device,
            generator,
            warmup=warmup,
            repeats=repeats,
            min_sample_s=min_sample_s,
        )
    return results
