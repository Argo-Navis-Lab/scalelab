"""The communication model against real collectives: processes on this machine's CPU over gloo,
and `scalelab cost comms`, which times and checks them."""

import pytest

from scalelab.bench.comms import SIZES, bench_collectives
from scalelab.cli import main
from scalelab.costmodel.comms import collective_time, fit_link

TOLERANCE = 0.5  # loose: shared runners are noisy, and a CPU's caches change its bandwidth


def test_the_all_reduce_formula_tracks_gloo() -> None:
    measured = bench_collectives(2, collectives=["all_reduce"])
    assert [m.nbytes for m in measured] == list(SIZES)
    assert all(m.world == 2 and m.timing.number == 1 for m in measured)  # one call a sample
    link = fit_link("all_reduce", 2, [(m.nbytes, m.seconds) for m in measured[::2]])
    errors = [collective_time("all_reduce", m.nbytes, 2, link) / m.seconds - 1 for m in measured]
    held_out = errors[1::2]
    assert sum(abs(e) for e in held_out) / len(held_out) < TOLERANCE


def test_cost_comms_prints_a_link_and_a_check_for_each_collective(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["cost", "comms", "--world", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        "backend:       gloo",
        "fit:           all_reduce at 4 KB, 64 KB, 1 MB, 16 MB; checked on the rest",
    ]
    assert lines[2].startswith("world 2:       24 timings in ") and "GB/s" in lines[2]
    assert lines[3].split() == ["collective", "world", "sizes", "MAPE", "bias", "worst"]
    checks = [line.split()[:3] for line in lines[4:]]
    assert checks == [
        ["all_reduce", "2", "4"],
        ["all_gather", "2", "8"],
        ["reduce_scatter", "2", "8"],
    ]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--world", "1"], "a collective needs 2 processes or more"),
        (["--backend", "nccl", "--world", "64"], "--world 64 over nccl needs as many GPUs"),
    ],
)
def test_impossible_worlds_are_bad_input(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cost", "comms", *args]) == 2
    assert message in capsys.readouterr().err
