"""The gemm benchmark: matrix multiplies of the shapes the models run, timed on a device."""

from collections.abc import Iterable, Sequence
from functools import partial

import torch
import torch.nn.functional as F

from scalelab.bench.timing import MIN_SAMPLE_S, REPEATS, WARMUP, Measurement, measure
from scalelab.costmodel.ops import Op, matmul_op
from scalelab.costmodel.specs import ModelSpec

# Tokens in a matmul's activations: prefill, log-prob and training passes, from a batch of
# prompts up to a micro-batch of whole sequences (64 × 256). Decoding's matmuls, one token a
# sequence, belong to the decode-step bench (m3s4).
ROWS = (1024, 4096, 16384)

Shape = tuple[int, int, int]  # (rows, inner, cols): (rows × inner) @ (inner × cols)


def gemm_shapes(models: Iterable[ModelSpec], rows: Sequence[int] = ROWS) -> list[Shape]:
    """Every matmul shape in a training step of these models, each once, in first-seen order:
    each weight matrix's forward matmul and its two gradients' (as op_graph counts them)."""
    shapes: dict[Shape, None] = {}  # a dict keeps the order; a set wouldn't
    for model in models:
        d, d_ff, vocab = model.d_model, model.d_ff, model.vocab
        q_width, kv_width = model.n_heads * model.head_dim, model.n_kv_heads * model.head_dim
        weights = [(d, q_width), (d, kv_width), (q_width, d), (d, d_ff), (d_ff, d), (d, vocab)]
        for r in rows:
            for inner, cols in weights:
                # forward, the weight's gradient, the input's gradient
                for shape in ((r, inner, cols), (inner, r, cols), (r, cols, inner)):
                    shapes[shape] = None
    return list(shapes)


def gemm_op(shape: Shape) -> Op:
    """One benchmarked matmul as the cost model counts it (layer 0: it is in no model)."""
    rows, inner, cols = shape
    return matmul_op(f"gemm {rows}x{inner}x{cols}", 0, rows, inner, cols, "gemm")


def bench_gemm(
    shapes: Iterable[Shape],
    device: torch.device,
    *,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    min_sample_s: float = MIN_SAMPLE_S,
) -> list[Measurement]:
    """Times each matmul the way the models' linear layers run it: float32 activations
    (rows × inner) times a weight stored as (cols × inner)."""
    generator = torch.Generator(device).manual_seed(0)
    results: list[Measurement] = []
    for shape in shapes:
        rows, inner, cols = shape
        x = torch.randn(rows, inner, generator=generator, device=device)
        weight = torch.randn(cols, inner, generator=generator, device=device)
        timing = measure(
            partial(F.linear, x, weight),
            device,
            warmup=warmup,
            repeats=repeats,
            min_sample_s=min_sample_s,
        )
        results.append(Measurement(op=gemm_op(shape), timing=timing))
    return results
