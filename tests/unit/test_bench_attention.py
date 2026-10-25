"""The attention benchmark: its ops are the op graph's, and SDPA is called as the models call it."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from scalelab.bench.attention import (
    BATCHES,
    CONTEXT,
    LENGTHS,
    AttentionCase,
    attention_cases,
    attention_op,
    bench_attention,
)
from scalelab.bench.gemm import ROWS
from scalelab.core.config import load_config
from scalelab.costmodel.ops import Op, grad_op, op_graph
from scalelab.costmodel.specs import ModelSpec, WorkloadSpec
from scalelab.models.attention.mha import MultiHeadAttention
from scalelab.models.rope import RotaryEmbedding

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
LADDER = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
CPU = torch.device("cpu")
# What a test sees of one call to SDPA: the inputs' shapes, their layouts, whether it was told
# to be causal, and the mask's shape and whether it allows every key.
Call = tuple[
    tuple[torch.Size, ...], tuple[tuple[int, ...], ...], bool, tuple[torch.Size, bool] | None
]


def cost(op: Op) -> tuple[int, int, int]:
    return op.flops, op.bytes_read, op.bytes_written


def spec(name: str) -> ModelSpec:
    return ModelSpec.from_config(load_config(HELLO, [f"model=ladder_s/{name}"]).model)


def layout(t: Tensor) -> tuple[int, ...]:
    """How a tensor lies in memory: the strides of its dimensions longer than 1 (a length-1
    dimension is only ever read at index 0, so its stride is never used)."""
    return tuple(stride for stride, size in zip(t.stride(), t.shape, strict=True) if size > 1)


def sdpa(model: ModelSpec, workload: WorkloadSpec, name: str = "attn.sdpa") -> Op:
    """The first layer's attention in a workload's op graph (every layer's is the same)."""
    return next(op for op in op_graph(model, workload) if op.name == name)


def test_ops_by_hand() -> None:
    training = attention_op(AttentionCase(batch=2, heads=3, head_dim=4, tokens=5))
    assert training.name == "sdpa b2 h3 q5 k5 d4" and training.op_class == "attention"
    # 2 × 15 query-key pairs (5 + 4 + 3 + 2 + 1 a sequence), 4 FLOPs a pair and dimension of
    # 12; read 10 queries, 10 keys and 10 values; write 10 outputs
    assert cost(training) == (4 * 30 * 12, 4 * 30 * 12, 4 * 10 * 12)
    generating = attention_op(AttentionCase(2, 3, 4, tokens=1, past=9))
    assert generating.name == "sdpa b2 h3 q1 k10 d4"
    # 2 queries see 10 keys each; read 2 queries, 20 keys and 20 values; write 2 outputs
    assert cost(generating) == (4 * 20 * 12, 4 * 42 * 12, 4 * 2 * 12)


@pytest.mark.parametrize("name", LADDER)
def test_the_cases_are_the_attention_of_training_and_generation(name: str) -> None:
    model = spec(name)
    cases = attention_cases([model])
    training = [WorkloadSpec(phase="train", batch=r // n, tokens=n) for r in ROWS for n in LENGTHS]
    generating = [
        WorkloadSpec(phase="decode", batch=b, tokens=1, past=n - 1)
        for b in BATCHES
        for n in LENGTHS
    ]
    assert len(cases) == len(training) + len(generating) == 28
    assert [cost(attention_op(case)) for case in cases] == [
        cost(sdpa(model, workload)) for workload in training + generating
    ]
    assert [cost(grad_op(attention_op(case))) for case in cases[: len(training)]] == [
        cost(sdpa(model, workload, "attn.sdpa.grad")) for workload in training
    ]


def test_shared_keys_and_values_wait_for_gqa() -> None:
    gqa = spec("m2").model_copy(update={"attn": "gqa", "n_kv_heads": 2})
    with pytest.raises(NotImplementedError, match="GQA"):
        attention_cases([gqa])


def test_sdpa_is_called_as_the_models_call_it(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Call] = []
    real = F.scaled_dot_product_attention

    def spy(
        q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None, is_causal: bool = False
    ) -> Tensor:
        mask = None if attn_mask is None else (attn_mask.shape, bool(attn_mask.all()))
        inputs = (q, k, v)
        calls.append(
            (tuple(t.shape for t in inputs), tuple(layout(t) for t in inputs), is_causal, mask)
        )
        return real(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    generator = torch.Generator().manual_seed(0)
    with torch.random.fork_rng():
        attn = MultiHeadAttention(d_model=128, n_heads=2)
    rope = RotaryEmbedding(64, CONTEXT)
    with torch.no_grad():
        attn(torch.randn(3, 16, 128, generator=generator), rope, torch.arange(16))  # training
        cache = attn.new_cache(3, CONTEXT, torch.float32, CPU)
        before = torch.randn(3, 2, 15, 64, generator=generator)
        cache.store(before, before, 0)  # 15 tokens in the cache
        attn.decode(torch.randn(3, 1, 128, generator=generator), rope, cache, 15)  # the 16th
    model_calls = set(calls)
    calls.clear()
    cases = [AttentionCase(3, 2, 64, tokens=16), AttentionCase(3, 2, 64, tokens=1, past=15)]
    bench_attention(cases, CPU, warmup=0, repeats=1, min_sample_s=0)
    assert len(model_calls) == 2 and set(calls) == model_calls


def test_training_is_timed_forward_then_backward_generation_forward_only() -> None:
    cases = [AttentionCase(2, 2, 4, tokens=8), AttentionCase(2, 2, 4, tokens=1, past=7)]
    results = bench_attention(cases, CPU, warmup=1, repeats=3, min_sample_s=1e-4)
    training, generating = (attention_op(case) for case in cases)
    assert [result.op for result in results] == [training, grad_op(training), generating]
    assert all(len(result.timing.samples) == 3 and result.seconds > 0 for result in results)


def test_fp16_attention_moves_half_the_bytes() -> None:
    case = AttentionCase(batch=2, heads=3, head_dim=4, tokens=5)
    full, half = attention_op(case), attention_op(case, "fp16")
    assert half.dtype == "fp16" and half.flops == full.flops
    assert (2 * half.bytes_read, 2 * half.bytes_written) == (full.bytes_read, full.bytes_written)


@pytest.mark.parametrize("name", LADDER)
def test_fp16_cases_are_the_attention_of_fp16_graphs(name: str) -> None:
    model = spec(name)
    for case in attention_cases([model]):
        workload = WorkloadSpec(
            phase="train" if case.past == 0 else "decode",
            batch=case.batch,
            tokens=case.tokens,
            past=case.past,
            precision="fp16",
        )
        op = attention_op(case, "fp16")
        assert op.dtype == sdpa(model, workload).dtype == "fp16"
        assert cost(op) == cost(sdpa(model, workload))


def test_fp16_is_timed_in_float16(monkeypatch: pytest.MonkeyPatch) -> None:
    dtypes: set[torch.dtype] = set()
    real = F.scaled_dot_product_attention

    def spy(
        q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None, is_causal: bool = False
    ) -> Tensor:
        dtypes.update(t.dtype for t in (q, k, v))
        return real(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    cases = [AttentionCase(2, 2, 4, tokens=8), AttentionCase(2, 2, 4, tokens=1, past=7)]
    results = bench_attention(cases, CPU, precision="fp16", warmup=1, repeats=3, min_sample_s=0)
    training, generating = (attention_op(case, "fp16") for case in cases)
    assert [result.op for result in results] == [training, grad_op(training), generating]
    assert dtypes == {torch.float16}
