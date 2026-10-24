"""Tests for the suites together: what each times, what fits a device, the Parquet files."""

from pathlib import Path

from scalelab.bench.attention import LENGTHS, attention_cases
from scalelab.bench.decode import StepMeasurement, decode_cases
from scalelab.bench.elementwise import COPY_SIZES, copy_op, elementwise_cases
from scalelab.bench.gemm import gemm_op, gemm_shapes
from scalelab.bench.suites import (
    SUITES,
    bench_folder,
    decode_fits,
    load_env,
    load_results,
    plan_suite,
    save_results,
)
from scalelab.bench.timing import Measurement, Timing
from scalelab.core.config import load_config
from scalelab.core.envcapture import capture_env
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
