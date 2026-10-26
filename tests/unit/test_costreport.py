from pathlib import Path

from scalelab.analysis.costreport import (
    CHANGES_FIGURE,
    PHASES_FIGURE,
    STEPS_FIGURE,
    TABLES,
    load_measurements,
    rl_baselines,
    tables,
    write_report,
)
from scalelab.costmodel.calibrate import Errors
from scalelab.costmodel.report import Change, DecisionReport, Measurement
from scalelab.costmodel.rlstep import Phase, PhaseTime, StepPrediction
from scalelab.costmodel.roofline import Bound
from scalelab.costmodel.validate import StepCheck, Validation

PNG = b"\x89PNG"
BAND = Errors(samples=5, mape=0.04, bias=0.01, worst=-0.1, median=0.0, spread=0.03)
OLD = Errors(samples=2, mape=0.2, bias=-0.2, worst=-0.3)  # saved before m3s25: no band


def phase(compute: float, memory: float, launch: float) -> PhaseTime:
    seconds = compute + memory + launch
    by_bound: dict[Bound, float] = {"compute": compute, "memory": memory, "launch": launch}
    return PhaseTime(by_bound=by_bound, by_class={"gemm": seconds}, flops=1e9)


def step(rl: bool, scale: float) -> StepPrediction:
    """An RL step (1.6 s: launch-bound generate 1.0, logprobs 0.2, train 0.4) or a warm start's
    (train 0.4 s), every time `scale` times as long."""
    train = phase(0.2 * scale, 0.1 * scale, 0.1 * scale)
    if not rl:
        return StepPrediction(hardware="t4", phases={"train": train})
    generate = phase(0.1 * scale, 0.2 * scale, 0.7 * scale)
    logprobs = phase(0.1 * scale, 0.05 * scale, 0.05 * scale)
    phases: dict[Phase, PhaseTime] = {"generate": generate, "logprobs": logprobs, "train": train}
    return StepPrediction(hardware="t4", phases=phases)


def measured(
    change: Change, predicted: float, actual: float, rl: bool = True, config: str = "a"
) -> Measurement:
    """A change predicted to speed the step up `predicted` times, measured at `actual`."""
    baseline = step(rl, 1.0)
    report = DecisionReport(
        change=change,
        experiment="configs/experiments/rl.yaml",
        config_hash=config * 16,
        hardware="t4",
        baseline=baseline,
        candidate=step(rl, 1 / predicted),
        validated=BAND,
    )
    laps = {name: time.seconds * 1.1 for name, time in baseline.phases.items()}
    laps["step"] = sum(laps.values())
    slower = {part: seconds / actual for part, seconds in laps.items()}
    return Measurement(report=report, steps=20, baseline=laps, candidate=slower)


def validation() -> Validation:
    rl = StepCheck(
        run_id="rl-run",
        steps=10,
        predicted={"generate": 1.0, "logprobs": 0.2, "train": 0.4, "step": 1.6},
        measured={"generate": 1.1, "logprobs": 0.2, "train": 0.4, "step": 1.7},
    )
    warm = StepCheck(
        run_id="ws-run",
        steps=10,
        predicted={"train": 0.4, "step": 0.4},
        measured={"train": 0.5, "step": 0.5},
    )
    return Validation(hardware="t4", errors={"step": BAND, "train": OLD}, checks=[rl, warm])


def test_measurements_are_read_from_cost_measures_files_alone(tmp_path: Path) -> None:
    measurement = measured("fp16", 1.5, 1.4)
    (tmp_path / "fp16-aaaa-t4-measured.json").write_text(measurement.model_dump_json())
    (tmp_path / "fp16-aaaa-t4.json").write_text(measurement.report.model_dump_json())  # compare's
    assert load_measurements(tmp_path) == [measurement]


def test_a_baseline_measured_by_both_changes_is_drawn_once() -> None:
    cut, fp16, other = (
        measured("padding-cut", 1.1, 1.1),
        measured("fp16", 1.5, 1.4),
        measured("fp16", 1.5, 1.4, config="b"),
    )
    warm = measured("fp16", 2.0, 2.0, rl=False)
    assert rl_baselines([cut, fp16, other, warm]) == [cut, other]


def test_the_tables_hold_each_part_phase_and_change() -> None:
    missed, held = measured("fp16", 1.5, 1.4), measured("padding-cut", 2.0, 2.0, rl=False)
    text = tables([validation()], [missed, held])
    assert "| t4 | step | 5 | 4.0% | +1.0% | -10.0% | +0.0% | 3.0% |\n" in text
    assert "| t4 | train | 2 | 20.0% | -20.0% | -30.0% | - | - |\n" in text
    assert "| t4 | 3e+09 | generate | 1.0000 | 10.0% | 20.0% | 70.0% | 1.1000 |\n" in text
    assert "| t4 | fp16 | RL | 3e+09 | 1.50x | 1.40x | +7.1% | no |\n" in text
    assert "| t4 | padding-cut | warm start | 1e+09 | 2.00x | 2.00x | +0.0% | yes |\n" in text
    assert text.index("| fp16 |") < text.index("| padding-cut |")  # by device, then change


def test_every_figure_is_saved_as_a_png(tmp_path: Path) -> None:
    changes = [measured("fp16", 1.5, 1.4), measured("padding-cut", 2.0, 2.0, rl=False)]
    saved = write_report([validation()], changes, tmp_path / "figs")
    assert [path.name for path in saved] == [STEPS_FIGURE, PHASES_FIGURE, CHANGES_FIGURE, TABLES]
    assert all(path.read_bytes().startswith(PNG) for path in saved[:3])


def test_figures_with_nothing_to_draw_are_left_out(tmp_path: Path) -> None:
    warm_only = write_report([validation()], [measured("fp16", 2.0, 2.0, rl=False)], tmp_path)
    assert [path.name for path in warm_only] == [STEPS_FIGURE, CHANGES_FIGURE, TABLES]
    nothing_measured = write_report([validation()], [], tmp_path / "bare")
    assert [path.name for path in nothing_measured] == [STEPS_FIGURE, TABLES]
