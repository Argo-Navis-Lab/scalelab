from pathlib import Path

import pytest

from scalelab.analysis.scaling import Family, Point, SizedRun
from scalelab.analysis.scalingreport import (
    BACKTEST_FIGURE,
    CURVES_FIGURE,
    SIZE_SHADES,
    VERSUS_FIGURE,
    Study,
    algo_name,
    by_size,
    shade,
    write_report,
)
from scalelab.analysis.stats import Interval

PNG = b"\x89PNG"
FAMILY = Family(a0=0.0, a1=1.0, c0=35.0, c1=2.0, b0=0.0, b1=0.0)
SIZES = {1_000_000: "m1", 3_000_000: "m2", 10_000_000: "m3"}


def run(params: int, seed: int) -> SizedRun:
    """A run of three gradings: R_0 at step 0, then two after it."""
    points = (
        Point(step=0, compute=0.0, rate=0.10 + 0.01 * seed, n=600),
        Point(step=10, compute=1e14, rate=0.20, n=600),
        Point(step=100, compute=1e15, rate=0.30, n=600),
    )
    return SizedRun(params=params, points=points)


def study(algo: str) -> Study:
    runs = tuple(run(params, seed) for seed in (1, 0) for params in reversed(SIZES))
    return Study(algo=algo, runs=runs, names=SIZES, family=FAMILY, held_out=FAMILY)


def test_runs_are_grouped_by_size_smallest_first() -> None:
    groups = by_size(study("grpo").runs)
    assert list(groups) == sorted(SIZES)
    assert all(len(runs) == 2 for runs in groups.values())


def test_the_smallest_and_largest_sizes_take_the_ends_of_the_ramp() -> None:
    assert [shade(k, 5) for k in range(5)] == list(SIZE_SHADES)
    assert (shade(0, 3), shade(2, 3)) == (SIZE_SHADES[0], SIZE_SHADES[-1])
    assert shade(0, 1) == SIZE_SHADES[-1]


def test_every_figure_is_saved_as_a_png(tmp_path: Path) -> None:
    gap = Interval(estimate=0.01, low=-0.01, high=0.03)
    gaps = {"samples": {"m1": gap, "m2": gap}, "flops": {"m1": gap, "m2": gap}}
    saved = write_report([study("grpo"), study("dapo")], gaps, tmp_path)
    assert [path.name for path in saved] == [CURVES_FIGURE, BACKTEST_FIGURE, VERSUS_FIGURE]
    assert all(path.read_bytes().startswith(PNG) for path in saved)


def test_the_report_compares_exactly_two_algorithms(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_report([study("grpo")], {}, tmp_path)


def test_the_figures_call_our_dapo_a_variant() -> None:
    # m4s24: our dapo never refills the batch it filtered, so it is not the paper's DAPO
    assert algo_name("dapo") == "DAPO-style (no refill)"
    assert algo_name("grpo") == "GRPO"
    assert algo_name("ppo") == "ppo"  # a preset without a name keeps its own
