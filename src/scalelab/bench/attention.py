"""The attention benchmark: scaled dot-product attention as the models call it, causal over
whole sequences (training), and a new token reading the KV cache (generation)."""

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from scalelab.bench.gemm import ROWS
from scalelab.bench.timing import (
    MIN_SAMPLE_S,
    REPEATS,
    WARMUP,
    Measurement,
    measure_forward_backward,
)
from scalelab.costmodel.ops import VALUE_BYTES, Op
from scalelab.costmodel.specs import ModelSpec
from scalelab.models.kvcache import LayerCache

CONTEXT = 256  # the models' context (ModelConfig's default): the longest sequence, a cache's size
LENGTHS = (16, 64, 128, CONTEXT)  # tokens a sequence
BATCHES = (1, 8, 64, 512)  # sequences generated together: an RL step's 64 prompts × 8 at most


class AttentionCase(NamedTuple):
    """Attention for `batch` sequences and `heads` heads of `head_dim`: `tokens` new queries a
    sequence, after `past` keys and values already in the cache."""

    batch: int
    heads: int
    head_dim: int
    tokens: int  # the whole sequence when training, 1 when generating
    past: int = 0  # 0: training's causal attention; more: a generation step's


def attention_cases(
    models: Iterable[ModelSpec],
    rows: Sequence[int] = ROWS,
    batches: Sequence[int] = BATCHES,
    lengths: Sequence[int] = LENGTHS,
) -> list[AttentionCase]:
    """Each size once, in first-seen order. For each model: training's attention over `rows`
    tokens cut into sequences of each length, then for each batch a generation step at the end
    of each length (the new token sees the length - 1 before it, and itself)."""
    cases: dict[AttentionCase, None] = {}  # a dict keeps the order; a set wouldn't
    for model in models:
        if model.n_kv_heads != model.n_heads:
            raise NotImplementedError("attention with shared keys and values (GQA) is M6's")
        h, d_head = model.n_heads, model.head_dim
        training = (AttentionCase(r // n, h, d_head, n) for r in rows for n in lengths)
        generating = (AttentionCase(b, h, d_head, 1, n - 1) for b in batches for n in lengths)
        cases.update(dict.fromkeys(training))
        cases.update(dict.fromkeys(generating))
    return list(cases)


def attention_op(case: AttentionCase) -> Op:
    """One benchmarked attention as op_graph counts attn.sdpa (layer 0: it is in no model)."""
    batch, heads, head_dim, tokens, past = case
    width = heads * head_dim
    queries, keys = batch * tokens, batch * (past + tokens)
    pairs = batch * (tokens * past + tokens * (tokens + 1) // 2)  # none of the masked ones
    return Op(
        name=f"sdpa b{batch} h{heads} q{tokens} k{past + tokens} d{head_dim}",
        layer=0,
        op_class="attention",
        flops=4 * pairs * width,
        bytes_read=VALUE_BYTES * (queries + 2 * keys) * width,
        bytes_written=VALUE_BYTES * queries * width,
    )


def causal(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Training's attention, called as MultiHeadAttention.forward calls it."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def cached(q: Tensor, k: Tensor, v: Tensor, allowed: Tensor) -> Tensor:
    """A generation step's attention, called as MultiHeadAttention.decode calls it: with an
    explicit mask, which can change the kernel PyTorch picks even where it allows every key."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)


def _kernel(
    case: AttentionCase, device: torch.device, generator: torch.Generator
) -> tuple[Callable[[], Tensor], list[Tensor]]:
    """The attention on random inputs laid out in memory as the models lay theirs out, and the
    tensors training's backward pass gives gradients to (none while generating)."""
    batch, heads, head_dim, tokens, past = case

    def split_heads(length: int, requires_grad: bool = False) -> Tensor:
        """(batch, heads, length, head_dim) as a view of (batch, length, heads, head_dim), the
        way _split_heads leaves it: a kernel may treat a view's strides differently."""
        shape = (batch, length, heads, head_dim)
        x = torch.randn(shape, generator=generator, device=device, requires_grad=requires_grad)
        return x.transpose(1, 2)

    if past == 0:
        q, k, v = (split_heads(tokens, requires_grad=True) for _ in range(3))
        return partial(causal, q, k, v), [q, k, v]
    # The cache is allocated at full length, as new_cache does, and read through a view.
    cache = LayerCache(batch, heads, CONTEXT, head_dim, torch.float32, device)
    k, v = cache.store(split_heads(past + tokens), split_heads(past + tokens), 0)
    positions = torch.arange(past, past + tokens, device=device)
    allowed = torch.arange(past + tokens, device=device) <= positions[:, None]  # as decode's
    return partial(cached, split_heads(tokens), k, v, allowed), []


def bench_attention(
    cases: Iterable[AttentionCase],
    device: torch.device,
    *,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    min_sample_s: float = MIN_SAMPLE_S,
) -> list[Measurement]:
    """Times each attention forward, then training's backward pass alone."""
    generator = torch.Generator(device).manual_seed(0)
    results: list[Measurement] = []
    for case in cases:
        forward, inputs = _kernel(case, device, generator)
        results += measure_forward_backward(
            forward,
            inputs,
            attention_op(case),
            device,
            generator,
            warmup=warmup,
            repeats=repeats,
            min_sample_s=min_sample_s,
        )
    return results
