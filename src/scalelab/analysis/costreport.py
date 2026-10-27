"""The cost model's record on some devices, as figures and a table file: each run's predicted
step against its measured one, where an RL step's time goes, and each measured change's predicted
speedup against its measured one. Drawn from the files cost validate and cost measure save."""

import io
from collections.abc import Collection, Sequence
from pathlib import Path

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scalelab.core.storage import atomic_write_bytes, atomic_write_text
from scalelab.costmodel.report import Change, Measurement
from scalelab.costmodel.rlstep import Phase
from scalelab.costmodel.roofline import Bound
from scalelab.costmodel.validate import TARGET_MAPE, Validation

MEASURED = "*-measured.json"  # cost measure's files in <home>/reports/ (m3s17)
STEPS_FIGURE = "steps.png"
PHASES_FIGURE = "rl-phases.png"
CHANGES_FIGURE = "changes.png"
TABLES = "tables.md"
RL_PHASES: tuple[Phase, ...] = ("generate", "logprobs", "train")
# The reference palette's first slots, always in this order; text stays in ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BOUND_COLORS: dict[Bound, str] = {"compute": BLUE, "memory": ORANGE, "launch": AQUA}
CHANGE_COLORS: dict[Change, str] = {"padding-cut": BLUE, "fp16": ORANGE, "fp16-cut": AQUA}
KIND_MARKERS = {"RL": "o", "warm start": "s"}


def load_measurements(folder: Path) -> list[Measurement]:
    """Every change cost measure saved in `folder` (<home>/reports/), in file name order."""
    return [Measurement.model_validate_json(p.read_text()) for p in sorted(folder.glob(MEASURED))]


def kind(parts: Collection[str]) -> str:
    """A step's kind from its parts or phases: an RL step generates, a warm start's doesn't."""
    return "RL" if "generate" in parts else "warm start"


def ordered(measurements: Sequence[Measurement]) -> list[Measurement]:
    """By device, change, kind, then the size of the step (its FLOPs)."""

    def key(measurement: Measurement) -> tuple[str, str, str, float]:
        report = measurement.report
        return report.hardware, report.change, kind(report.baseline.phases), report.baseline.flops

    return sorted(measurements, key=key)


def rl_baselines(measurements: Sequence[Measurement]) -> list[Measurement]:
    """The measured RL changes, one per device and experiment (each change measures the same
    baseline: fp32, without the cut), by device and then the size of the step."""
    seen: dict[tuple[str, str], Measurement] = {}
    for measurement in measurements:
        report = measurement.report
        if kind(report.baseline.phases) == "RL":
            seen.setdefault((report.hardware, report.config_hash), measurement)
    return sorted(seen.values(), key=lambda m: (m.report.hardware, m.report.baseline.flops))


def percent(value: float | None, sign: bool = False) -> str:
    if value is None:  # an Errors field saved before m3s25
        return "-"
    return f"{value:+.1%}" if sign else f"{value:.1%}"


def tables(validations: Sequence[Validation], measurements: Sequence[Measurement]) -> str:
    """The figures' numbers as Markdown tables."""
    lines = [
        "# The cost model's record",
        "",
        "Made by `scalelab report cost` from each device's validation.json (cost validate) and",
        "the measured changes in reports/ (cost measure). Errors are predicted / measured - 1.",
        "",
        "## Whole steps and phases, predicted vs measured",
        "",
        "| device | part | runs | MAPE | bias | worst | median | spread |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for validation in validations:
        for part, errors in validation.errors.items():
            lines.append(
                f"| {validation.hardware} | {part} | {errors.samples} | {percent(errors.mape)} "
                f"| {percent(errors.bias, sign=True)} | {percent(errors.worst, sign=True)} "
                f"| {percent(errors.median, sign=True)} | {percent(errors.spread)} |"
            )
    lines += [
        "",
        "## Where an RL step's time goes (predicted fp32, without the cut; measured laps)",
        "",
        "| device | FLOPs a step | phase | predicted s | compute | memory | launch | measured s |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for measurement in rl_baselines(measurements):
        report = measurement.report
        for phase in RL_PHASES:
            time = report.baseline.phases[phase]
            shares = " | ".join(
                percent(time.by_bound.get(bound, 0.0) / time.seconds) for bound in BOUND_COLORS
            )
            lines.append(
                f"| {report.hardware} | {report.baseline.flops:.3g} | {phase} "
                f"| {time.seconds:.4f} | {shares} | {measurement.baseline[phase]:.4f} |"
            )
    lines += [
        "",
        "## Changes, predicted vs measured speedups (held: within the device's validated MAPE)",
        "",
        "| device | change | kind | FLOPs a step | predicted | measured | error | held |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for measurement in ordered(measurements):
        report = measurement.report
        lines.append(
            f"| {report.hardware} | {report.change} | {kind(report.baseline.phases)} "
            f"| {report.baseline.flops:.3g} | {report.speedup:.2f}x | {measurement.speedup:.2f}x "
            f"| {percent(measurement.error, sign=True)} | {'yes' if measurement.held else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def panels(count: int) -> tuple[Figure, list[Axes]]:
    """A figure of `count` panels side by side: a light surface, a recessive grid, no top or
    right spine."""
    figure = Figure(figsize=(5.0 * count, 4.2), facecolor=SURFACE, layout="constrained")
    axes = [figure.add_subplot(1, count, index + 1) for index in range(count)]
    for panel in axes:
        panel.set_facecolor(SURFACE)
        panel.grid(color=GRID, linewidth=0.6)
        panel.set_axisbelow(True)
        panel.tick_params(colors=MUTED, labelcolor=INK)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
    return figure, axes


def steps_figure(validations: Sequence[Validation]) -> Figure:
    """Each run's measured seconds a step against its predicted ones, a panel a device: on the
    solid line they agree, between the dashed ones they are within TARGET_MAPE."""
    figure, axes = panels(len(validations))
    for panel, validation in zip(axes, validations, strict=True):
        checks = validation.checks
        seconds = [s for check in checks for s in (check.predicted["step"], check.measured["step"])]
        low, high = min(seconds) / 1.5, max(seconds) * 1.5
        panel.plot([low, high], [low, high], color=MUTED, linewidth=1)
        for factor in (1 - TARGET_MAPE, 1 + TARGET_MAPE):
            ends = [low * factor, high * factor]
            panel.plot([low, high], ends, color=MUTED, linewidth=1, linestyle="--")
        for name, color in (("RL", BLUE), ("warm start", ORANGE)):
            some = [check for check in checks if kind(check.predicted) == name]
            if some:
                panel.scatter(
                    [check.measured["step"] for check in some],
                    [check.predicted["step"] for check in some],
                    s=40,
                    color=color,
                    edgecolors=SURFACE,
                    linewidths=1,
                    label=f"{name} runs ({len(some)})",
                    zorder=3,
                )
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(low, high)
        panel.set_ylim(low, high)
        panel.set_xlabel("measured seconds a step", color=INK)
        panel.set_ylabel("predicted seconds a step", color=INK)
        step = validation.errors["step"]
        title = f"{validation.hardware}: step MAPE {step.mape:.1%}, bias {step.bias:+.1%}"
        panel.set_title(title, color=INK, fontsize=11)
        panel.legend(frameon=False, labelcolor=INK)
    return figure


def phases_figure(measurements: Sequence[Measurement]) -> Figure:
    """Each measured RL experiment's predicted phases split by what bounds them, its measured
    laps as diamonds: a panel a device, a group of bars an experiment (smallest step first)."""
    baselines = rl_baselines(measurements)
    devices = list(dict.fromkeys(m.report.hardware for m in baselines))
    figure, axes = panels(len(devices))
    for panel, device in zip(axes, devices, strict=True):
        here = [m for m in baselines if m.report.hardware == device]
        ticks: list[float] = []
        for group, measurement in enumerate(here):
            prediction = measurement.report.baseline
            start = group * (len(RL_PHASES) + 1)
            for index, phase in enumerate(RL_PHASES):
                bottom = 0.0
                for bound, color in BOUND_COLORS.items():
                    height = prediction.phases[phase].by_bound.get(bound, 0.0)
                    panel.bar(
                        start + index,
                        height,
                        bottom=bottom,
                        width=0.8,
                        color=color,
                        edgecolor=SURFACE,
                        linewidth=1,
                    )
                    bottom += height
                ticks.append(start + index)
            laps = [measurement.baseline[phase] for phase in RL_PHASES]
            panel.scatter(
                [start + index for index in range(len(RL_PHASES))],
                laps,
                marker="D",
                s=30,
                color=INK,
                zorder=3,
            )
            flops = f"{prediction.flops:.1e} FLOPs".replace("e+", "e")
            panel.text(
                start + (len(RL_PHASES) - 1) / 2,
                -0.13,
                flops,
                transform=panel.get_xaxis_transform(),
                ha="center",
                va="top",
                color=MUTED,
                fontsize=9,
            )
        panel.set_xticks(ticks, ["gen", "logp", "train"] * len(here), fontsize=8)
        panel.set_ylabel("seconds a step", color=INK)
        panel.set_title(f"{device}: an RL step's phases, fp32", color=INK, fontsize=11)
        handles: list[Artist] = [Patch(color=c, label=b) for b, c in BOUND_COLORS.items()]
        handles.append(Line2D([], [], marker="D", color=INK, linestyle="none", label="measured"))
        panel.legend(handles=handles, frameon=False, labelcolor=INK)
    return figure


def changes_figure(measurements: Sequence[Measurement]) -> Figure:
    """Each measured change's predicted speedup against its measured one, a panel a device:
    color the change, a circle an RL step and a square a warm start's, hollow where the
    prediction missed the model's validated error on that device."""
    devices = list(dict.fromkeys(m.report.hardware for m in ordered(measurements)))
    figure, axes = panels(len(devices))
    for panel, device in zip(axes, devices, strict=True):
        here = [m for m in measurements if m.report.hardware == device]
        speedups = [s for m in here for s in (m.report.speedup, m.speedup)]
        low, high = min(speedups) / 1.15, max(speedups) * 1.15
        panel.plot([low, high], [low, high], color=MUTED, linewidth=1)
        for measurement in here:
            color = CHANGE_COLORS[measurement.report.change]
            panel.scatter(
                [measurement.report.speedup],
                [measurement.speedup],
                marker=KIND_MARKERS[kind(measurement.report.baseline.phases)],
                s=48,
                facecolors=color if measurement.held else SURFACE,
                edgecolors=color,
                linewidths=1.5,
                zorder=3,
            )
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(low, high)
        panel.set_ylim(low, high)
        panel.set_xlabel("predicted speedup (x)", color=INK)
        panel.set_ylabel("measured speedup (x)", color=INK)
        held = sum(m.held for m in here)
        panel.set_title(f"{device}: {held} of {len(here)} changes held", color=INK, fontsize=11)
        handles: list[Artist] = [
            Line2D([], [], marker="o", color=c, linestyle="none", label=change)
            for change, c in CHANGE_COLORS.items()
        ]
        handles += [
            Line2D([], [], marker=m, color=MUTED, linestyle="none", label=name)
            for name, m in KIND_MARKERS.items()
        ]
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor=SURFACE,
                markeredgecolor=MUTED,
                linestyle="none",
                label="missed its band",
            )
        )
        panel.legend(handles=handles, frameon=False, labelcolor=INK, fontsize=8)
    return figure


def save_figure(figure: Figure, path: Path) -> Path:
    """A figure as a PNG, written atomically like every plain file."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150)
    return atomic_write_bytes(path, buffer.getvalue())


def write_report(
    validations: Sequence[Validation], measurements: Sequence[Measurement], out: Path
) -> list[Path]:
    """report cost's files in `out`: the figures with something to draw, then TABLES. Every
    validation needs its runs' steps (files saved since m3s26)."""
    saved = [save_figure(steps_figure(validations), out / STEPS_FIGURE)]
    if rl_baselines(measurements):
        saved.append(save_figure(phases_figure(measurements), out / PHASES_FIGURE))
    if measurements:
        saved.append(save_figure(changes_figure(measurements), out / CHANGES_FIGURE))
    saved.append(atomic_write_text(out / TABLES, tables(validations, measurements)))
    return saved
