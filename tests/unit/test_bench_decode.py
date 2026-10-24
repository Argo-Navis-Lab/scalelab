"""The decode-step benchmark: whole generation steps, timed, with the ops that predict them."""

from pathlib import Path

import pytest
import torch
from torch import Tensor

from scalelab.bench.attention import BATCHES, LENGTHS
from scalelab.bench.decode import DecodeCase, StepMeasurement, bench_decode, decode_cases
from scalelab.bench.timing import Timing
from scalelab.core.config import ModelConfig, load_config
from scalelab.costmodel.ops import op_graph
from scalelab.costmodel.specs import ModelSpec, WorkloadSpec
from scalelab.models.kvcache import KVCache
from scalelab.models.transformer import Transformer

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
CPU = torch.device("cpu")
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1, context=32)


def test_a_step_for_every_batch_and_length() -> None:
    m1 = load_config(HELLO, ["model=ladder_s/m1"]).model
    cases = decode_cases([m1])
    assert [(case.batch, case.past + 1) for case in cases] == [
        (b, n) for b in BATCHES for n in LENGTHS
    ]
    assert cases[0] == DecodeCase(m1, 1, 15) and cases[-1] == DecodeCase(m1, 512, 255)


def test_a_measurement_knows_its_ops() -> None:
    spec = ModelSpec.from_config(TINY)
    workload = WorkloadSpec(phase="decode", batch=4, tokens=1, past=9)
    timing = Timing(number=2, samples=(3e-3, 1e-3, 2e-3))
    step = StepMeasurement(model="tiny", spec=spec, workload=workload, timing=timing)
    assert step.ops == op_graph(spec, workload) and step.seconds == 2e-3


def test_steps_are_timed_in_order() -> None:
    # the first position, and the last one the cache holds
    cases = [DecodeCase(TINY, 2, 0), DecodeCase(TINY, 3, TINY.context - 1)]
    state = torch.random.get_rng_state()
    results = bench_decode(cases, CPU, warmup=2, repeats=3, min_sample_s=1e-3)
    assert torch.equal(torch.random.get_rng_state(), state)  # the caller's random numbers
    assert [(r.model, r.workload.batch, r.workload.past) for r in results] == [
        ("tiny", 2, 0),
        ("tiny", 3, 31),
    ]
    assert all(len(r.timing.samples) == 3 and r.seconds > 0 for r in results)


def test_every_call_decodes_the_same_position(monkeypatch: pytest.MonkeyPatch) -> None:
    positions: list[int] = []  # the cache's length as each decode() starts
    real = Transformer.decode

    def spy(self: Transformer, tokens: Tensor, cache: KVCache) -> Tensor:
        positions.append(cache.length)
        return real(self, tokens, cache)

    monkeypatch.setattr(Transformer, "decode", spy)
    bench_decode([DecodeCase(TINY, 2, 5)], CPU, warmup=2, repeats=3, min_sample_s=1e-3)
    assert len(positions) >= 5 and set(positions) == {5}
