"""Tests for the suites together: what each times, what fits a device, the Parquet files."""

from pathlib import Path

import pytest

from scalelab.bench.attention import LENGTHS, attention_cases
from scalelab.bench.decode import StepMeasurement, decode_cases
from scalelab.bench.elementwise import COPY_SIZES, copy_op, elementwise_cases
from scalelab.bench.gemm import gemm_op, gemm_shapes
from scalelab.bench.suites import (
    HALF_SUITES,
    SUITES,
    bench_folder,
    decode_fits,
    load_env,
    load_results,
    load_samples,
    plan_suite,
    results_file,
    save_results,
)
from scalelab.bench.timing import Measurement, Timing
from scalelab.core.config import load_config
from scalelab.core.envcapture import capture_env
from scalelab.costmodel.calibrate import Sample, env_versions
from scalelab.costmodel.specs import HARDWARE, ModelSpec, WorkloadSpec

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
LADDER = [load_config(HELLO, [f"model=ladder_s/m{size}"]).model for size in range(1, 8)]


def test_each_suite_plans_its_own_cases() -> None:
    m1 = LADDER[0]
    spec = ModelSpec.from_config(m1)
    expected = {
        "gemm": len(gemm_shapes([spec])),
        "copy": len(COPY_SIZES),
        "elementwise": len(elementwise_cases([spec])),
        "attention": len(attention_cases([spec])),
        "decode": len(decode_cases([m1])),
    }
    assert {suite: plan_suite(suite, [m1], None).cases for suite in SUITES} == expected
    assert [plan_suite(suite, [m1], None).skipped for suite in SUITES] == [0] * 5


def test_only_m7s_biggest_batch_is_too_big_for_a_16_gb_gpu() -> None:
    """At batch 512 the fp32 cache of 256 positions is 17 GB for m7 and 11 GB for m6 (m3s5);
    90% of 16 GB is usable. A step allocates its whole cache, so every position is too big."""
    cases = decode_cases(LADDER)
    for name in ("t4", "p100"):
        too_big = [
            (case.model.name, case.batch, case.past)
            for case in cases
            if not decode_fits(case, HARDWARE[name])
        ]
        assert too_big == [("m7", 512, n - 1) for n in LENGTHS]
    assert all(decode_fits(case, HARDWARE["a100-40g"]) for case in cases)
    plan = plan_suite("decode", LADDER, HARDWARE["t4"])
    assert (plan.cases, plan.skipped) == (len(cases) - 4, 4)
    assert plan_suite("decode", LADDER, None).skipped == 0  # the CPU: no memory model


def test_results_round_trip_through_parquet(tmp_path: Path) -> None:
    spec = ModelSpec.from_config(LADDER[0])
    measurements = [
        Measurement(op=copy_op(4096), timing=Timing(number=40, samples=(1.5e-6, 1.25e-6))),
        Measurement(op=gemm_op(gemm_shapes([spec])[0]), timing=Timing(number=1, samples=(3e-5,))),
    ]
    workload = WorkloadSpec(phase="decode", batch=8, tokens=1, past=63)
    timing = Timing(number=2, samples=(7.1e-4, 7.3e-4, 7e-4))
    steps = [StepMeasurement(model="m1", spec=spec, workload=workload, timing=timing)]
    env = capture_env()
    folder = bench_folder(tmp_path, "t4")
    assert folder == tmp_path / "bench" / "t4"
    save_results(folder / "copy.parquet", measurements, env)
    save_results(folder / "decode.parquet", steps, env)
    assert sorted(path.name for path in folder.iterdir()) == ["copy.parquet", "decode.parquet"]
    assert load_results(folder / "copy.parquet", Measurement) == measurements
    assert load_results(folder / "decode.parquet", StepMeasurement) == steps
    assert load_env(folder / "decode.parquet") == env


def test_samples_come_from_every_suite_saved(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_samples(tmp_path)
    copies = [Measurement(op=copy_op(4096), timing=Timing(number=40, samples=(2e-6,)))]
    workload = WorkloadSpec(phase="decode", batch=8, tokens=1, past=63)
    timing = Timing(number=2, samples=(7.1e-4, 7.3e-4, 7e-4))
    step = StepMeasurement(
        model="m1", spec=ModelSpec.from_config(LADDER[0]), workload=workload, timing=timing
    )
    env = capture_env()
    save_results(tmp_path / "copy.parquet", copies, env)
    save_results(tmp_path / "decode.parquet", [step], env)
    samples, versions = load_samples(tmp_path)
    assert samples == [Sample("copy", [copy_op(4096)], 2e-6), Sample("decode", step.ops, 7.1e-4)]
    assert versions == env_versions(env)
    newer = env.software.model_copy(update={"packages": {"torch": "9.9.9"}})
    save_results(tmp_path / "copy.parquet", copies, env.model_copy(update={"software": newer}))
    with pytest.raises(ValueError, match="different versions"):
        load_samples(tmp_path)


def test_fp16_plans_only_the_suites_it_changes() -> None:
    m1 = LADDER[0]
    for suite in HALF_SUITES:
        assert plan_suite(suite, [m1], None, "fp16").cases == plan_suite(suite, [m1], None).cases
    for suite in sorted(set(SUITES) - set(HALF_SUITES)):
        with pytest.raises(ValueError, match="fp32 only"):
            plan_suite(suite, [m1], None, "fp16")
    assert results_file("gemm") == "gemm.parquet"
    assert results_file("decode", "fp16") == "decode-fp16.parquet"


def test_fp16_samples_are_labelled_apart(tmp_path: Path) -> None:
    fp32 = Measurement(op=gemm_op((8, 16, 4)), timing=Timing(number=1, samples=(5e-5,)))
    fp16 = Measurement(op=gemm_op((8, 16, 4), "fp16"), timing=Timing(number=1, samples=(3e-5,)))
    env = capture_env()
    save_results(tmp_path / results_file("gemm", "fp16"), [fp16], env)
    save_results(tmp_path / results_file("gemm"), [fp32], env)
    samples, _ = load_samples(tmp_path)  # fp32's first, whichever was saved first
    assert samples == [Sample("gemm", [fp32.op], 5e-5), Sample("gemm-fp16", [fp16.op], 3e-5)]
