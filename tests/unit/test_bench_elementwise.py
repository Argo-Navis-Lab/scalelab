"""The copy and elementwise benchmarks: their ops are the op graph's, and each one is timed."""

from pathlib import Path

import pytest
import torch

from scalelab.bench.elementwise import (
    COPY_SIZES,
    TOKENS,
    Case,
    bench_copy,
    bench_elementwise,
    case_op,
    copy_op,
    elementwise_cases,
    swiglu,
)
from scalelab.bench.gemm import ROWS
from scalelab.core.config import load_config
from scalelab.costmodel.ops import VALUE_BYTES, Op, grad_op, op_graph
from scalelab.costmodel.specs import ModelSpec, WorkloadSpec
from scalelab.models.transformer import SwiGLU

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
LADDER = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
CPU = torch.device("cpu")


def cost(op: Op) -> tuple[int, int, int]:
    return op.flops, op.bytes_read, op.bytes_written


def test_copies_by_hand() -> None:
    assert copy_op(10).name == "copy 10" and cost(copy_op(10)) == (0, 40, 40)
    assert VALUE_BYTES * COPY_SIZES[0] == 16 * 2**10  # 16 KB
    assert VALUE_BYTES * COPY_SIZES[-1] == 256 * 2**20  # 256 MB


@pytest.mark.parametrize(
    ("case", "read"),
    [
        (Case("norm", 3, 5), 4 * (15 + 5)),  # x and the weight
        (Case("rope", 3, 8, 4), 4 * (24 + 2 * TOKENS * 4)),  # x and each position's cos, sin
        (Case("swiglu", 3, 5), 4 * 2 * 15),  # gate and up
        (Case("gelu", 3, 5), 4 * 15),
        (Case("residual", 3, 5), 4 * 2 * 15),  # x and the branch's output
    ],
)
def test_ops_by_hand(case: Case, read: int) -> None:
    op = case_op(case)
    assert op.name == f"{case.kind} 3x{case.width}" and op.op_class == "elementwise"
    assert cost(op) == (0, read, 4 * 3 * case.width)


def test_a_gradient_by_the_rule_of_thumb() -> None:
    op = grad_op(case_op(Case("norm", 3, 5)))  # read 80 bytes, wrote 60
    assert op.name == "norm 3x5.grad" and cost(op) == (0, 80 + 60, 80)


@pytest.mark.parametrize(
    "overrides",
    [[f"model=ladder_s/{name}"] for name in LADDER] + [["model=ladder_s/m2", "model.qk_norm=true"]],
)
def test_the_cases_are_the_elementwise_ops_of_a_training_step(overrides: list[str]) -> None:
    model = ModelSpec.from_config(load_config(HELLO, overrides).model)
    cases = elementwise_cases([model])
    assert elementwise_cases([model, model]) == cases  # a case seen twice is kept once
    assert len(set(cases)) == len(cases)
    forward: set[tuple[int, int, int]] = set()
    backward: set[tuple[int, int, int]] = set()
    for rows in ROWS:
        workload = WorkloadSpec(phase="train", batch=rows // TOKENS, tokens=TOKENS)
        for op in op_graph(model, workload):
            if op.op_class == "elementwise" and not op.name.startswith("embed"):
                (backward if op.name.endswith(".grad") else forward).add(cost(op))
    assert {cost(case_op(case)) for case in cases} == forward
    assert {cost(grad_op(case_op(case))) for case in cases} == backward


def test_swiglu_is_what_the_mlp_computes() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        mlp = SwiGLU(8, 32)
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(mlp(x), mlp.down(swiglu(mlp.gate(x), mlp.up(x))))


def test_copies_are_timed_in_order() -> None:
    results = bench_copy([16, 1024], CPU, warmup=1, repeats=3, min_sample_s=1e-4)
    assert [result.op for result in results] == [copy_op(16), copy_op(1024)]
    assert all(len(result.timing.samples) == 3 and result.seconds > 0 for result in results)


def test_each_op_is_timed_forward_then_backward() -> None:
    cases = [
        Case("norm", 4, 8),
        Case("rope", TOKENS, 8, 4),  # one sequence, 2 heads of 4
        Case("swiglu", 4, 8),
        Case("gelu", 4, 8),
        Case("residual", 4, 8),
    ]
    results = bench_elementwise(cases, CPU, warmup=1, repeats=3, min_sample_s=1e-4)
    expected: list[Op] = []
    for case in cases:
        expected.append(case_op(case))
        if case.kind != "residual":  # an add's gradient takes no kernel
            expected.append(grad_op(case_op(case)))
    assert [result.op for result in results] == expected
    assert all(len(result.timing.samples) == 3 and result.seconds > 0 for result in results)
