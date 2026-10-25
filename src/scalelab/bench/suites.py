"""Every benchmark at once: what each suite times for a set of models, which cases fit on the
device, and the results saved as Parquet, one file a suite."""

import io
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple, get_args

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scalelab.bench.attention import attention_cases, bench_attention
from scalelab.bench.decode import DecodeCase, StepMeasurement, bench_decode, decode_cases
from scalelab.bench.elementwise import COPY_SIZES, bench_copy, bench_elementwise, elementwise_cases
from scalelab.bench.gemm import bench_gemm, gemm_shapes
from scalelab.bench.timing import Measurement
from scalelab.core.config import ModelConfig, StrictModel
from scalelab.core.envcapture import EnvInfo
from scalelab.core.storage import atomic_write_bytes
from scalelab.costmodel.calibrate import Sample, env_versions
from scalelab.costmodel.memory import max_generation_batch
from scalelab.costmodel.ops import VALUE_BYTES
from scalelab.costmodel.specs import HardwareSpec, ModelSpec, Precision

Suite = Literal["gemm", "copy", "elementwise", "attention", "decode"]
SUITES: tuple[Suite, ...] = get_args(Suite)
ENV_KEY = b"env"  # the Parquet metadata key that holds where the results were measured
# m3s19: the suites whose kernels change in fp16. Copies (casts among them) and elementwise
# kernels keep fp32's efficiencies (m3s18), so they run in fp32 only.
HALF_SUITES: tuple[Suite, ...] = ("gemm", "attention", "decode")
PRECISIONS: tuple[Precision, ...] = get_args(Precision)


class SuitePlan(NamedTuple):
    """What a suite will time: how many cases, how many left out, and the call that times them."""

    cases: int
    skipped: int  # cases too big for the device's memory
    run: Callable[[torch.device], Sequence[StrictModel]]


def bench_folder(home: Path, hardware: str) -> Path:
    """Where a device's results go: <home>/bench/<hardware>/, one Parquet file a suite."""
    return home / "bench" / hardware


def results_file(suite: Suite, precision: Precision = "fp32") -> str:
    """A suite's file in a device's folder: <suite>.parquet, or <suite>-fp16.parquet."""
    return f"{suite}.parquet" if precision == "fp32" else f"{suite}-{precision}.parquet"


def decode_fits(case: DecodeCase, hardware: HardwareSpec) -> bool:
    """Whether a decode step's weights and KV cache fit in the device's usable memory, in fp32.
    The cache holds the model's whole context, whatever position the step decodes. An fp16
    step needs less where it matters (m3s19): half the cache, plus half the weights' bytes for
    their casts."""
    spec = ModelSpec.from_config(case.model)
    return case.batch <= max_generation_batch(spec, hardware, case.model.context, VALUE_BYTES)


def plan_suite(
    suite: Suite,
    models: Sequence[ModelConfig],
    hardware: HardwareSpec | None,
    precision: Precision = "fp32",
) -> SuitePlan:
    """A suite's cases for these models. Without a hardware profile (the CPU) all of them run:
    only whole decode steps (every layer's cache) can outgrow a GPU."""
    if precision != "fp32" and suite not in HALF_SUITES:
        raise ValueError(f"the {suite} suite runs in fp32 only")
    specs = [ModelSpec.from_config(model) for model in models]
    if suite == "gemm":
        shapes = gemm_shapes(specs)
        return SuitePlan(len(shapes), 0, partial(bench_gemm, shapes, precision=precision))
    if suite == "copy":
        return SuitePlan(len(COPY_SIZES), 0, partial(bench_copy, COPY_SIZES))
    if suite == "elementwise":
        ops = elementwise_cases(specs)
        return SuitePlan(len(ops), 0, partial(bench_elementwise, ops))
    if suite == "attention":
        calls = attention_cases(specs)
        return SuitePlan(len(calls), 0, partial(bench_attention, calls, precision=precision))
    every = decode_cases(models)
    steps = [case for case in every if hardware is None or decode_fits(case, hardware)]
    run = partial(bench_decode, steps, precision=precision)
    return SuitePlan(len(steps), len(every) - len(steps), run)


def save_results(path: Path, results: Sequence[StrictModel], env: EnvInfo) -> None:
    """A suite's results as one Parquet file, a row each (nested fields become structs), and
    where they were measured in the file's metadata. Replaces the file if it exists."""
    table = pa.Table.from_pylist([result.model_dump(mode="json") for result in results])
    table = table.replace_schema_metadata({ENV_KEY: env.model_dump_json()})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    atomic_write_bytes(path, buffer.getvalue())


def load_results[R: StrictModel](path: Path, kind: type[R]) -> list[R]:
    """What save_results wrote, read back as `kind` (Measurement or StepMeasurement)."""
    return [kind.model_validate(row) for row in pq.read_table(path).to_pylist()]


def load_env(path: Path) -> EnvInfo:
    """Where a suite's results were measured: the commit, the versions, the machine."""
    metadata = pq.read_schema(path).metadata or {}
    return EnvInfo.model_validate_json(metadata[ENV_KEY])


def load_samples(folder: Path) -> tuple[list[Sample], dict[str, str]]:
    """Every suite's results in a device's folder as calibration samples (a whole decode step
    is one sample of all its ops), fp32's then fp16's, each labelled by its file's name
    ("gemm", m3s19: "gemm-fp16"), and the versions they were measured with."""
    samples: list[Sample] = []
    found: dict[str, dict[str, str]] = {}
    for precision in PRECISIONS:
        for suite in SUITES:
            path = folder / results_file(suite, precision)
            if not path.exists():
                continue
            found[path.stem] = env_versions(load_env(path))
            if suite == "decode":
                steps = load_results(path, StepMeasurement)
                samples += [Sample(path.stem, step.ops, step.seconds) for step in steps]
            else:
                results = load_results(path, Measurement)
                samples += [Sample(path.stem, [result.op], result.seconds) for result in results]
    if not found:
        raise FileNotFoundError(f"no benchmark results in {folder}")
    versions = next(iter(found.values()))
    if any(other != versions for other in found.values()):
        raise ValueError(f"the suites ran on different versions, so fit them apart: {found}")
    return samples, versions
