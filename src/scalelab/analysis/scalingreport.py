"""The scaling study as figures: each algorithm's runs against its family of curves, its
largest size predicted from the smaller ones alone, and one algorithm minus the other, size
by size, at equal samples and at equal FLOPs. Drawn from what joint, backtest and versus
compute."""

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from scalelab.analysis.costreport import BLUE, INK, MUTED, ORANGE, panels, save_figure
from scalelab.analysis.scaling import Family, SizedRun
from scalelab.analysis.stats import Interval

CURVES_FIGURE = "scaling-curves.png"
BACKTEST_FIGURE = "scaling-backtest.png"
VERSUS_FIGURE = "scaling-versus.png"
# Light for small, dark for large: the sizes are ordered, so one blue ramp, not five hues
SIZE_SHADES = ("#a9cbf2", "#73a9e8", "#2a78d6", "#1c5aa8", "#0f3a72")
MEASURE_COLORS = {"samples": BLUE, "flops": ORANGE}
MEASURE_NAMES = {"samples": "samples", "flops": "FLOPs", "seconds": "hours"}
CURVE_POINTS = 200  # where a curve is drawn, evenly spaced in log C
R_LABEL = "R: greedy pass@1, pooled (%)"
C_LABEL = "C: RL FLOPs spent"


@dataclass(frozen=True)
class Study:
    """One algorithm's runs (m4s15): `family` fit to all of them, `held_out` to every size
    but the largest, as backtest fits it."""

    algo: str
    runs: tuple[SizedRun, ...]
    names: Mapping[int, str]  # N -> the size's name
    family: Family
    held_out: Family


def by_size(runs: Sequence[SizedRun]) -> dict[int, list[SizedRun]]:
    """The runs grouped by N, smallest first."""
    groups: dict[int, list[SizedRun]] = {}
    for run in sorted(runs, key=lambda run: run.params):
        groups.setdefault(run.params, []).append(run)
    return groups


def shade(index: int, count: int) -> str:
    """The ramp's color for the index-th of `count` sizes, both ends of the ramp used."""
    if count == 1:
        return SIZE_SHADES[-1]
    return SIZE_SHADES[round(index * (len(SIZE_SHADES) - 1) / (count - 1))]


def draw_size(panel: Axes, family: Family, runs: Sequence[SizedRun], color: str) -> None:
    """One size: every seed's gradings after step 0 as dots, the family's curve through
    them as a line (its R_0 the seeds' mean step 0)."""
    points = [point for run in runs for point in run.points[1:]]
    computes = [point.compute for point in points]
    rates = [100 * point.rate for point in points]
    panel.scatter(computes, rates, s=9, color=color, alpha=0.45, linewidths=0)
    curve = family.curve(runs[0].params, statistics.mean(run.points[0].rate for run in runs))
    grid = np.geomspace(min(computes), max(computes), CURVE_POINTS)
    panel.plot(grid, [100 * float(curve(c)) for c in grid], color=color, linewidth=1.8)


def label(panel: Axes, title: str) -> None:
    """A curves panel's axes: C on a log scale, R in percent."""
    panel.set_xscale("log")
    panel.set_xlabel(C_LABEL, color=INK)
    panel.set_ylabel(R_LABEL, color=INK)
    panel.set_title(title, color=INK, fontsize=10, loc="left")


def curves_figure(studies: Sequence[Study]) -> Figure:
    """A panel an algorithm: each size's seeds and its curve from the one family."""
    figure, axes = panels(len(studies))
    for panel, study in zip(axes, studies, strict=True):
        groups = by_size(study.runs)
        handles: list[Line2D] = []
        for index, (params, runs) in enumerate(groups.items()):
            color = shade(index, len(groups))
            draw_size(panel, study.family, runs, color)
            handles.append(Line2D([], [], color=color, marker="o", label=study.names[params]))
        label(panel, f"{study.algo}: one family through {len(study.runs)} runs")
        panel.legend(handles=handles, frameon=False, labelcolor=INK, fontsize=8)
    return figure


def backtest_figure(studies: Sequence[Study]) -> Figure:
    """A panel an algorithm: the largest size's seeds against the curve the smaller sizes'
    family predicts for it from its R_0 alone."""
    figure, axes = panels(len(studies))
    for panel, study in zip(axes, studies, strict=True):
        groups = by_size(study.runs)
        largest = max(groups)
        draw_size(panel, study.held_out, groups[largest], ORANGE)
        smaller = " ".join(study.names[params] for params in groups if params != largest)
        label(panel, f"{study.algo}: {study.names[largest]} predicted from {smaller}")
        handles = [
            Line2D([], [], color=ORANGE, marker="o", linestyle="", label="measured, each seed"),
            Line2D([], [], color=ORANGE, label="predicted before RL"),
        ]
        panel.legend(handles=handles, frameon=False, labelcolor=INK, fontsize=8)
    return figure


def versus_figure(gaps: Mapping[str, Mapping[str, Interval]], base: str, other: str) -> Figure:
    """other - base by size, one point and its 95% interval per measure (samples, FLOPs),
    side by side: above the zero line the other algorithm is ahead."""
    figure, (panel,) = panels(1)
    sizes = sorted({size for found in gaps.values() for size in found})
    offset = 0.12
    for k, (measure, found) in enumerate(gaps.items()):
        shift = (k - (len(gaps) - 1) / 2) * 2 * offset
        xs = [index + shift for index, size in enumerate(sizes) if size in found]
        shown = [found[size] for size in sizes if size in found]
        middle = [100 * gap.estimate for gap in shown]
        below = [100 * (gap.estimate - gap.low) for gap in shown]
        above = [100 * (gap.high - gap.estimate) for gap in shown]
        panel.errorbar(
            xs,
            middle,
            yerr=[below, above],
            fmt="o",
            color=MEASURE_COLORS.get(measure, MUTED),
            capsize=3,
            label=f"at equal {MEASURE_NAMES.get(measure, measure)}",
        )
    panel.axhline(0, color=MUTED, linewidth=0.8)
    panel.set_xticks(range(len(sizes)), sizes)
    panel.set_ylabel(f"{other} - {base}, R in pp (95% interval)", color=INK)
    panel.set_title(f"{other} - {base}, paired by size and seed", color=INK, fontsize=10)
    panel.legend(frameon=False, labelcolor=INK, fontsize=8)
    return figure


def write_report(
    studies: Sequence[Study],
    gaps: Mapping[str, Mapping[str, Interval]],
    out: Path,
) -> list[Path]:
    """report scaling's three figures in `out`; the versus one compares the second study's
    algorithm with the first's."""
    base, other = (study.algo for study in studies)
    return [
        save_figure(curves_figure(studies), out / CURVES_FIGURE),
        save_figure(backtest_figure(studies), out / BACKTEST_FIGURE),
        save_figure(versus_figure(gaps, base, other), out / VERSUS_FIGURE),
    ]
