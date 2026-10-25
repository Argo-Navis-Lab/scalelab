"""The gemm benchmark: its shapes are the op graph's matmuls, and it times each one."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from scalelab.bench.gemm import ROWS, bench_gemm, gemm_op, gemm_shapes
from scalelab.core.config import load_config
from scalelab.costmodel.ops import Op, op_graph
from scalelab.costmodel.specs import ModelSpec, Precision, WorkloadSpec

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
LADDER = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]


def spec(name: str) -> ModelSpec:
    return ModelSpec.from_config(load_config(HELLO, [f"model=ladder_s/{name}"]).model)


def cost(op: Op) -> tuple[int, int, int]:
    return op.flops, op.bytes_read, op.bytes_written


def test_a_gemm_op_by_hand() -> None:
    op = gemm_op((2, 3, 4))  # (2 × 3) @ (3 × 4)
    assert op.name == "gemm 2x3x4" and op.op_class == "gemm"
    assert cost(op) == (2 * 2 * 3 * 4, 4 * (2 * 3 + 3 * 4), 4 * 2 * 4)


def test_m1_has_9_shapes_a_row_count() -> None:
    # q, k, v and o share one shape (MHA); the MLP's down repeats two of the up's three:
    # 2 + 3 + 1, and 3 for the head
    assert len(gemm_shapes([spec("m1")])) == 9 * len(ROWS)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("name", LADDER)
def test_the_shapes_are_the_matmuls_of_a_training_step(name: str, precision: Precision) -> None:
    model = spec(name)
    shapes = gemm_shapes([model])
    assert gemm_shapes([model, model]) == shapes  # a shape seen twice is kept once
    assert len(set(shapes)) == len(shapes)  # m3's d_ff and m7's d_model are 1024: fewer than 27
    used: set[tuple[int, int, int]] = set()
    for rows in ROWS:
        workload = WorkloadSpec(phase="train", batch=rows // 128, tokens=128, precision=precision)
        used |= {cost(op) for op in op_graph(model, workload) if op.op_class == "gemm"}
    assert {cost(gemm_op(shape, precision)) for shape in shapes} == used


def test_each_shape_is_timed_in_order() -> None:
    shapes = [(8, 16, 4), (64, 32, 48)]
    results = bench_gemm(shapes, torch.device("cpu"), warmup=1, repeats=3, min_sample_s=1e-4)
    assert [result.op for result in results] == [gemm_op(shape) for shape in shapes]
    assert all(len(result.timing.samples) == 3 and result.seconds > 0 for result in results)


def test_an_fp16_gemm_moves_half_the_bytes() -> None:
    op = gemm_op((2, 3, 4), "fp16")
    assert op.dtype == "fp16" and cost(op) == (2 * 2 * 3 * 4, 2 * (2 * 3 + 3 * 4), 2 * 2 * 4)


def test_fp16_times_float16_matmuls(monkeypatch: pytest.MonkeyPatch) -> None:
    dtypes: set[torch.dtype] = set()
    real = F.linear

    def spy(x: Tensor, weight: Tensor) -> Tensor:
        dtypes.update((x.dtype, weight.dtype))
        return real(x, weight)

    monkeypatch.setattr(F, "linear", spy)
    cpu = torch.device("cpu")
    results = bench_gemm([(8, 16, 4)], cpu, precision="fp16", warmup=1, repeats=3, min_sample_s=0)
    assert [result.op for result in results] == [gemm_op((8, 16, 4), "fp16")]
    assert dtypes == {torch.float16}
