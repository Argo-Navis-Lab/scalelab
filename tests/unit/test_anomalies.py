"""Tests for the anomaly events: one per finding, where it starts, with what the run does about
it; and the detectors' memory carried through a checkpoint."""

import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from scalelab.core.checkpoint import load_checkpoint, save_checkpoint
from scalelab.telemetry.anomalies import AnomalyEvent, AnomalyMonitor, severity
from scalelab.telemetry.collectors import Cadence
from scalelab.telemetry.detectors import Collapse, Divergence, Spike
from scalelab.telemetry.sink import MetricsSink


def new_monitor(home: Path) -> AnomalyMonitor:
    return AnomalyMonitor(MetricsSink(home, "run"), Cadence())


def run(
    monitor: AnomalyMonitor, steps: Sequence[Mapping[str, float]], first: int = 1
) -> list[AnomalyEvent]:
    """Record each step's values in the monitor's sink and check the step, numbering the steps
    from `first`, as a run does. Returns every event."""
    events: list[AnomalyEvent] = []
    for step, values in enumerate(steps, start=first):
        for metric, value in values.items():
            monitor.sink.record(step, metric, value)
        events += monitor.check(step)
    return events


def test_the_severity_says_what_the_run_does():
    assert severity(Divergence(5, "numerics/loss_nan", 3.0)) == "HALT"
    assert severity(Divergence(5, "optim/grad_norm", 0.02)) == "SNAPSHOT"
    assert severity(Spike(5, "optim/loss", 9.0, 1.0, 0.1, 54.0)) == "WARN"
    assert severity(Collapse(5, "policy/entropy", 0.001, 0.01)) == "WARN"


def test_a_lasting_finding_is_one_event_where_it_starts(tmp_path: Path):
    monitor = new_monitor(tmp_path)
    events = run(monitor, [{"policy/entropy": 0.001}] * 40)  # found from step 20 to step 40
    assert [(e.step, e.detector, e.metric, e.severity) for e in events] == [
        (20, "collapse", "policy/entropy", "WARN")
    ]
    assert events[0].window_metrics == pytest.approx({"mean": 0.001, "limit": 0.01})


def test_a_finding_that_stops_and_starts_again_is_two_events(tmp_path: Path):
    counts = [0, 2, 0, 1, 1]  # NaNs in the loss at steps 2, 4 and 5
    events = run(new_monitor(tmp_path), [{"numerics/loss_nan": count} for count in counts])
    assert [(e.step, e.severity, e.window_metrics) for e in events] == [
        (2, "HALT", {"value": 2.0}),
        (4, "HALT", {"value": 1.0}),
    ]


def test_an_event_alerts_the_cadence(tmp_path: Path):
    quiet, alarmed = new_monitor(tmp_path / "quiet"), new_monitor(tmp_path / "alarmed")
    run(quiet, [{"policy/entropy": 0.5}] * 40)
    run(alarmed, [{"policy/entropy": 0.001}] * 40)  # the event is at step 20
    assert quiet.cadence.until == 0
    assert alarmed.cadence.until == 70  # every step measured in detail for the next 50


def test_a_step_with_no_metrics_finds_nothing(tmp_path: Path):
    assert new_monitor(tmp_path).check(1) == []


def healthy_then_troubled(steps: int) -> list[dict[str, float]]:
    """A run's metrics: noisy but healthy, except that entropy collapses from step 121, the
    loss spikes at step 250, and the gradient norm grows e-fold every 33 steps from step 200."""
    rng = random.Random(0)
    values: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        grad_norm = rng.gauss(1.0, 0.05) * math.exp(0.03 * max(0, step - 200))
        values.append(
            {
                "optim/loss": 2.0 if step == 250 else rng.gauss(1.0, 0.01),
                "optim/grad_norm": grad_norm,
                "policy/entropy": 0.001 if step > 120 else 0.5,
                "policy/flat_groups": 0.5,
            }
        )
    return values


def test_the_detectors_carry_on_through_a_checkpoint(tmp_path: Path):
    steps = healthy_then_troubled(300)
    whole = new_monitor(tmp_path / "whole")
    events = run(whole, steps)
    assert {e.detector for e in events} == {"spike", "divergence", "collapse"}

    before = new_monitor(tmp_path / "before")
    run(before, steps[:150])  # stopped at step 150, mid-collapse
    manifest = save_checkpoint(tmp_path / "ckpt", 150, {"anomalies": before.state()}, tmp_path)
    after = new_monitor(tmp_path / "after")
    after.load_state(load_checkpoint(manifest, tmp_path)[1]["anomalies"])
    resumed = run(after, steps[150:], first=151)

    assert resumed == [e for e in events if e.step > 150]  # the collapse is not found anew
    assert resumed  # spikes and a divergence
    assert after.state() == whole.state()
