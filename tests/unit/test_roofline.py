"""Roofline timing tests: each term by hand, the ridge point, and what holds back decoding
and training on a T4."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from scalelab.core.config import load_config
from scalelab.costmodel.ops import Op, OpClass, op_graph
from scalelab.costmodel.roofline import IDEAL, Efficiency, op_time, predict, time_ops
from scalelab.costmodel.specs import HARDWARE, HardwareSpec, ModelSpec, WorkloadSpec

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
LADDER = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
T4 = HARDWARE["t4"]
N = 4096  # a 4096 × 4096 × 4096 matmul: 683 FLOPs a byte, far above any ridge
MATMUL = Op(
    name="mm", layer=1, op_class="gemm", flops=2 * N**3, bytes_read=8 * N**2, bytes_written=4 * N**2
)
ADD = Op(name="add", layer=1, op_class="elementwise", flops=0, bytes_read=8000, bytes_written=4000)
TRAIN = WorkloadSpec(phase="train", batch=64, tokens=128)


def spec(name: str = "m1") -> ModelSpec:
    return ModelSpec.from_config(load_config(HELLO, [f"model=ladder_s/{name}"]).model)


def decode(batch: int) -> WorkloadSpec:
    return WorkloadSpec(phase="decode", batch=batch, tokens=1, past=100)


# --- the datasheets ---------------------------------------------------------------------


def test_the_t4_ridge_points():
    assert T4.ridge("fp16") == pytest.approx(203.125)  # the blueprint's "about 200"
    assert T4.ridge("fp32") == pytest.approx(25.3125)  # every run so far is fp32


@pytest.mark.parametrize("name", list(HARDWARE))
def test_every_profile_has_fp32_and_bf16_only_where_supported(name: str):
    device = HARDWARE[name]
    assert device.name == name and "fp32" in device.peak_flops
    assert ("bf16" in device.peak_flops) == device.supports_bf16


@pytest.mark.parametrize("peak_flops", [{"fp8": 1e12}, {"fp32": 0.0}])
def test_unknown_formats_and_zero_speeds_are_refused(peak_flops: dict[str, float]):
    fields = {"memory_bytes": 1, "supports_bf16": False, "peak_flops": peak_flops, "memory_bw": 1}
    with pytest.raises(ValidationError):
        HardwareSpec.model_validate({"name": "x", **fields})


# --- one op, term by term ---------------------------------------------------------------


def test_a_big_matmul_is_compute_bound():
    t = op_time(MATMUL, T4)
    assert t.compute_s == pytest.approx(2 * N**3 / 8.1e12)  # 17 ms
    assert t.memory_s == pytest.approx(12 * N**2 / 320e9)  # 0.6 ms
    assert t.launch_s == 0 and t.seconds == t.compute_s and t.bound == "compute"


def test_an_add_is_memory_bound():
    t = op_time(ADD, T4)
    assert t.compute_s == 0 and t.memory_s == pytest.approx(12_000 / 320e9)
    assert t.seconds == t.memory_s and t.bound == "memory"


def test_efficiencies_slow_each_term():
    ideal, slow = op_time(MATMUL, T4), op_time(MATMUL, T4, Efficiency(compute=0.5, memory=0.25))
    assert slow.compute_s == pytest.approx(2 * ideal.compute_s)
    assert slow.memory_s == pytest.approx(4 * ideal.memory_s)
    late = op_time(ADD, T4, Efficiency(launch_s=1e-5))  # 10 µs, much more than 37.5 ns of bytes
    assert late.seconds == pytest.approx(late.memory_s + 1e-5) and late.bound == "launch"


@pytest.mark.parametrize("field", [{"compute": 0}, {"memory": 1.5}, {"launch_s": -1e-6}])
def test_efficiencies_are_fractions_and_launches_are_not_negative(field: dict[str, float]):
    with pytest.raises(ValidationError):
        Efficiency.model_validate(field)


@pytest.mark.parametrize("workload", [TRAIN, decode(8), decode(512)], ids=["train", "8", "512"])
@pytest.mark.parametrize("name", LADDER)
def test_an_ideal_op_is_compute_bound_exactly_above_the_ridge(name: str, workload: WorkloadSpec):
    for t in predict(spec(name), workload, T4).times:
        assert (t.bound == "compute") == (t.op.intensity >= T4.ridge("fp32")), t.op.name


# --- a whole workload -------------------------------------------------------------------


def test_a_prediction_adds_up():
    ops = op_graph(spec("m3"), TRAIN)
    prediction = predict(spec("m3"), TRAIN, T4)
    assert prediction == time_ops(ops, T4) and prediction.hardware == "t4"
    assert [t.op for t in prediction.times] == ops
    assert prediction.seconds == pytest.approx(sum(op_time(op, T4).seconds for op in ops))
    assert sum(prediction.by_bound.values()) == pytest.approx(prediction.seconds)
    classes = prediction.by_class
    assert list(classes) == ["elementwise", "gemm", "attention"]  # in the order they first run
    assert sum(c.ops for c in classes.values()) == len(ops)
    assert sum(c.seconds for c in classes.values()) == pytest.approx(prediction.seconds)


def test_efficiencies_apply_by_op_class():
    ops = op_graph(spec(), TRAIN)
    every: dict[OpClass, Efficiency] = {
        "gemm": IDEAL,
        "gemv-decode": IDEAL,
        "attention": IDEAL,
        "elementwise": IDEAL,
    }
    assert time_ops(ops, T4, every) == time_ops(ops, T4)
    slow = time_ops(ops, T4, {**every, "gemm": Efficiency(compute=0.5)})
    for s, i in zip(slow.times, time_ops(ops, T4).times, strict=True):
        assert s.compute_s == pytest.approx((2 if s.op.op_class == "gemm" else 1) * i.compute_s)
        assert s.memory_s == i.memory_s
    del every["attention"]
    with pytest.raises(KeyError):  # a class without an efficiency is an error
        time_ops(ops, T4, every)


@pytest.mark.parametrize("name", LADDER)
def test_training_is_compute_bound_unless_the_batch_is_tiny(name: str):
    assert predict(spec(name), TRAIN, T4).bottleneck == "compute"
    tiny = WorkloadSpec(phase="train", batch=2, tokens=10)
    assert predict(spec(name), tiny, T4).bottleneck == "memory"


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("name", LADDER)
def test_decoding_a_few_sequences_is_memory_bound(name: str, batch: int):
    prediction = predict(spec(name), decode(batch), T4)
    assert prediction.bottleneck == "memory"
    assert all(c.bound == "memory" for c in prediction.by_class.values())


@pytest.mark.parametrize("name", LADDER)
def test_fp32_decoding_at_a_big_batch_crosses_the_ridge(name: str):
    # at batch 512 the decode matmuls do 20 to 150 FLOPs a byte, mostly past fp32's ridge of
    # 25 (but below fp16's 203); reading the KV cache stays memory-bound
    classes = predict(spec(name), decode(512), T4).by_class
    assert classes["gemv-decode"].bound == "compute" and classes["attention"].bound == "memory"
