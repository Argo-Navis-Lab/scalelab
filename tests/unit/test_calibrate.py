"""Calibration tests: every op has a group the benchmarks fit, the fit finds the efficiencies
that made its data, whole steps fit as sums, an outlier stays one, and a file warns when stale."""

import warnings
from pathlib import Path

import pytest

from scalelab.bench.attention import attention_cases, attention_op
from scalelab.bench.elementwise import COPY_SIZES, case_op, copy_op, elementwise_cases
from scalelab.bench.gemm import gemm_op, gemm_shapes
from scalelab.core.config import load_config
from scalelab.core.envcapture import GpuInfo, HardwareInfo, capture_env
from scalelab.costmodel.calibrate import (
    GROUPS,
    Sample,
    StaleCalibration,
    calibrate,
    env_versions,
    load_calibration,
    save_calibration,
)
from scalelab.costmodel.ops import Op, grad_op, matmul_op, op_graph
from scalelab.costmodel.roofline import Efficiency, Group, group, time_ops
from scalelab.costmodel.specs import HARDWARE, ModelSpec, WorkloadSpec

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
T4 = HARDWARE["t4"]
TRAIN = WorkloadSpec(phase="train", batch=64, tokens=128)
VERSIONS = {"torch": "2.14.1", "cuda": "12.8", "device": "Tesla T4"}
TRUE: dict[Group, Efficiency] = {
    "gemm": Efficiency(compute=0.6, memory=0.7, launch_s=8e-6),
    "copy": Efficiency(memory=0.8, launch_s=5e-6),
}


def spec(*overrides: str) -> ModelSpec:
    return ModelSpec.from_config(load_config(HELLO, list(overrides)).model)


def decode(batch: int, past: int) -> WorkloadSpec:
    return WorkloadSpec(phase="decode", batch=batch, tokens=1, past=past)


def single_ops() -> list[Op]:
    """Matmuls bound by each term in turn (launch, memory, compute), and copies."""
    widths = (64, 256, 1024, 4096)
    gemms = [matmul_op("gemm", 0, r, w, w, "gemm") for r in (1, 16, 256, 4096) for w in widths]
    return gemms + [copy_op(values) for values in COPY_SIZES]


def timed(
    ops: list[Op], efficiencies: dict[Group, Efficiency], suite: str = "bench"
) -> list[Sample]:
    return [Sample(suite, [op], time_ops([op], T4, efficiencies).seconds) for op in ops]


@pytest.mark.parametrize(
    "overrides",
    [[f"model=ladder_s/m{size}"] for size in range(1, 8)]
    + [["model=ladder_s/m2", "model.qk_norm=true"]],
)
def test_every_op_has_a_group_the_benchmarks_fit(overrides: list[str]) -> None:
    model = spec(*overrides)
    needed = {group(op) for op in op_graph(model, TRAIN) + op_graph(model, decode(64, 100))}
    benched = [gemm_op(shape) for shape in gemm_shapes([model])]
    benched += [copy_op(values) for values in COPY_SIZES]
    for case in elementwise_cases([model]):  # a residual add's backward isn't timed
        benched += (
            [case_op(case)] if case.kind == "residual" else [case_op(case), grad_op(case_op(case))]
        )
    benched += [attention_op(case) for case in attention_cases([model])]
    benched += op_graph(model, decode(1, 15))  # a whole decode step
    fitted = {group(op) for op in benched}
    assert fitted == set(GROUPS)
    assert needed <= fitted


def test_the_fit_finds_the_efficiencies_that_made_the_times() -> None:
    calibration = calibrate(timed(single_ops(), TRUE), T4, VERSIONS)
    assert calibration.efficiencies.keys() == TRUE.keys()  # only the groups it has samples of
    for name, true in TRUE.items():
        fitted = calibration.efficiencies[name]
        assert fitted.compute == pytest.approx(true.compute, rel=1e-3)
        assert fitted.memory == pytest.approx(true.memory, rel=1e-3)
        assert fitted.launch_s == pytest.approx(true.launch_s, rel=1e-3)
    assert calibration.efficiencies["copy"].compute == 1.0  # no FLOPs: nothing to fit
    assert calibration.errors["bench"].samples == len(single_ops())
    assert calibration.errors["bench"].mape < 1e-4


def test_a_term_no_sample_was_limited_by_stays_ideal() -> None:
    big = [matmul_op("gemm", 0, n, n, n, "gemm") for n in (2048, 4096, 8192)]  # compute-bound
    fitted = calibrate(timed(big, TRUE), T4, VERSIONS).efficiencies["gemm"]
    assert fitted.memory == 1.0  # nothing to fit it to: the roofline's bound
    assert fitted.compute == pytest.approx(TRUE["gemm"].compute, rel=1e-3)


def test_whole_steps_fit_as_the_sum_of_their_ops() -> None:
    true: dict[Group, Efficiency] = {
        name: Efficiency(compute=0.9 - 0.05 * i, memory=0.3 + 0.05 * i, launch_s=(2 + i) * 1e-6)
        for i, name in enumerate(GROUPS)
    }
    model = spec("model=ladder_s/m1")
    ops = single_ops() + [case_op(case) for case in elementwise_cases([model])]
    ops += [attention_op(case) for case in attention_cases([model])]
    steps = [
        Sample("decode", op_graph(model, decode(batch, past)), 0.0)
        for batch in (1, 8, 64, 512)
        for past in (15, 63, 127, 255)
    ]
    steps = [step._replace(seconds=time_ops(step.ops, T4, true).seconds) for step in steps]
    calibration = calibrate(timed(ops, true) + steps, T4, VERSIONS)
    assert calibration.efficiencies.keys() == {
        "gemv-decode",
        *TRUE,
        "attention",
        "norm",
        "rope",
        "act",
        "residual",
    }
    assert calibration.errors["decode"].samples == 16
    assert calibration.errors["decode"].mape < 1e-3
    assert calibration.errors["bench"].mape < 1e-3


def test_an_outlier_stays_an_outlier() -> None:
    samples = timed(single_ops(), TRUE)
    samples[15] = samples[15]._replace(seconds=10 * samples[15].seconds)  # 4096³: a hiccup
    calibration = calibrate(samples, T4, VERSIONS)
    gaps = sorted(
        abs(time_ops(s.ops, T4, calibration.efficiencies).seconds / s.seconds - 1) for s in samples
    )
    assert gaps[-2] < 0.05  # every other sample still fits
    assert gaps[-1] > 0.85  # and the outlier is predicted near its true time, a tenth
    assert calibration.efficiencies["gemm"].compute == pytest.approx(0.6, rel=0.05)


def test_versions_are_torch_cuda_and_the_device() -> None:
    env = capture_env()
    software = env.software.model_copy(update={"packages": {"torch": "2.14.1"}, "cuda": None})
    mac = env.model_copy(
        update={"software": software, "hardware": HardwareInfo(cpu_model="Apple M2", cpu_count=8)}
    )
    assert env_versions(mac) == {"torch": "2.14.1", "cuda": "none", "device": "Apple M2"}
    t4 = HardwareInfo(
        cpu_model="Intel(R) Xeon(R) CPU @ 2.20GHz",
        cpu_count=2,
        gpus=(GpuInfo(name="Tesla T4", memory_mib=15360, driver="550.54.15"),),
    )
    colab = mac.model_copy(
        update={"software": software.model_copy(update={"cuda": "12.8"}), "hardware": t4}
    )
    assert env_versions(colab) == VERSIONS


def test_a_calibration_file_round_trips_and_warns_when_versions_change(tmp_path: Path) -> None:
    calibration = calibrate(timed(single_ops(), TRUE), T4, VERSIONS)
    path = tmp_path / "calibration.json"
    save_calibration(path, calibration)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        assert load_calibration(path) == calibration
        assert load_calibration(path, VERSIONS) == calibration
    with pytest.warns(StaleCalibration, match="fitted on torch 2.14.1, this is 2.15.0"):
        load_calibration(path, {**VERSIONS, "torch": "2.15.0"})
