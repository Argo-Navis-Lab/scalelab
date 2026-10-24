"""The timing harness: warmup, enough calls a sample, the median, and waiting for the GPU."""

from typing import Any

import pytest
import torch
from pydantic import ValidationError

from scalelab.bench.timing import REPEATS, Timing, measure

CPU = torch.device("cpu")
TICK = 2.0**-12  # seconds a fake call takes: powers of 2 add up exactly


class FakeClock:
    """A clock that moves only when the fake op runs, and a log of what happened in order."""

    def __init__(self) -> None:
        self.now = 0.0
        self.log: list[str] = []

    def __call__(self) -> float:
        self.log.append("clock")
        return self.now

    def op(self, seconds: float = TICK) -> None:
        self.log.append("op")
        self.now += seconds


def test_warmup_then_twice_the_calls_until_a_sample_lasts_long_enough() -> None:
    clock = FakeClock()
    timing = measure(clock.op, CPU, warmup=2, repeats=5, min_sample_s=4 * TICK, clock=clock)
    assert clock.log[:3] == ["op", "op", "clock"]  # the warmup calls aren't timed
    assert timing.number == 4  # 1 call lasts 1 tick and 2 calls 2: too short
    assert timing.samples == (TICK,) * 5  # each sample is per call
    assert clock.log.count("op") == 2 + (1 + 2 + 4) + 5 * 4


def test_the_median_ignores_a_slow_sample() -> None:
    clock = FakeClock()
    lengths = iter([TICK, 3 * TICK, TICK, 64 * TICK, 2 * TICK, 4 * TICK])  # first: 1 call is enough
    timing = measure(
        lambda: clock.op(next(lengths)), CPU, warmup=0, repeats=5, min_sample_s=TICK, clock=clock
    )
    assert timing.samples == (3 * TICK, TICK, 64 * TICK, 2 * TICK, 4 * TICK)
    assert timing.median == 3 * TICK  # the mean would be 14.8 ticks
    assert timing.spread == 63  # the slowest sample took 64 times the fastest


def test_on_a_gpu_the_clock_is_read_once_the_device_has_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()

    def synchronize(device: torch.device) -> None:  # stands in for torch.cuda.synchronize
        clock.log.append(f"sync {device}")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    measure(clock.op, torch.device("cuda:0"), warmup=1, repeats=2, min_sample_s=TICK, clock=clock)
    # one warmup call, then 3 runs of 1 call: finding the number, then 2 samples
    assert clock.log == ["op"] + ["sync cuda:0", "clock", "op", "sync cuda:0", "clock"] * 3


def test_on_a_cpu_nothing_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    def synchronize(device: torch.device) -> None:
        raise AssertionError(f"a CPU has nothing to wait for, not even {device}")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    clock = FakeClock()
    assert measure(clock.op, CPU, warmup=0, repeats=1, min_sample_s=TICK, clock=clock).number == 1


def test_a_real_op_on_the_real_clock() -> None:
    values = torch.ones(1000)
    timing = measure(values.sum, CPU)
    assert len(timing.samples) == REPEATS
    assert timing.median > 0 and timing.spread >= 0


@pytest.mark.parametrize(
    "fields",
    [
        {"number": 0, "samples": (1.0,)},  # no calls
        {"number": 1, "samples": ()},  # no samples
        {"number": 1, "samples": (1.0, 0.0)},  # a sample that took no time
    ],
)
def test_a_timing_needs_calls_and_samples_that_took_time(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Timing.model_validate(fields)
