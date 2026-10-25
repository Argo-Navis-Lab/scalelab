"""Anomaly events: what the online detectors find in a run, one event where each finding
starts, each with a severity that says what the run does about it. WARN: the event is recorded
and the detailed collectors measure every step for a while. SNAPSHOT: a forensic checkpoint is
kept as well. HALT: the run also stops."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from scalelab.telemetry.collectors import Cadence
from scalelab.telemetry.detectors import (
    NONFINITE_PREFIX,
    Collapse,
    CollapseDetector,
    Divergence,
    DivergenceDetector,
    Regression,
    Spike,
    SpikeDetector,
    ThroughputDetector,
)
from scalelab.telemetry.sink import MetricsSink

Severity = Literal["WARN", "SNAPSHOT", "HALT"]  # each does what the one before it does, and more


@dataclass(frozen=True)
class AnomalyEvent:
    """A detector's finding, at the step it started. `metric` is what the detector watched;
    `window_metrics` are the numbers it judged by, such as a spike's value, median, mad and z."""

    step: int
    detector: str  # "spike", "divergence", "collapse" or "throughput" (m3s25)
    metric: str
    severity: Severity
    window_metrics: dict[str, float]


class RunHalted(Exception):
    """A HALT event stopped the run."""

    def __init__(self, event: AnomalyEvent) -> None:
        super().__init__(f"step {event.step}: {event.detector} in {event.metric}")
        self.event = event


Finding = Spike | Divergence | Collapse | Regression


def severity(finding: Finding) -> Severity:
    """What the run does about a finding. NaNs or infinities have ruined it: HALT. A gradient
    norm growing steadily is on its way there: SNAPSHOT, the state kept while it is still
    finite. A spike may pass, and a collapse is WARN too: halting would end the run short of
    its FLOP budget, and the runs compared are compute-matched. So is a throughput regression
    (m3s25): slow steps compute what fast ones would."""
    if isinstance(finding, Divergence):
        return "HALT" if finding.metric.startswith(NONFINITE_PREFIX) else "SNAPSHOT"
    return "WARN"


def to_event(detector: str, finding: Finding) -> AnomalyEvent:
    """The event for a finding: its numbers, less the step and the metric, are the window's."""
    numbers = dataclasses.asdict(finding)
    del numbers["step"], numbers["metric"]
    return AnomalyEvent(finding.step, detector, finding.metric, severity(finding), numbers)


def ignore(event: AnomalyEvent) -> None:
    """The default `report`: events go nowhere."""


class AnomalyMonitor:
    """Runs the three detectors on the metrics `sink` holds for each step, and turns what they
    find into events; the throughput detector too, given one (m3s25: a run whose step the cost
    model can predict). A detector keeps finding a problem while it lasts (a lasting jump is a
    spike for 100 steps), so an event marks only where a finding starts: a (detector, metric)
    that found nothing at the step before. Each event goes to `report`, and alerts `cadence`
    so the detailed collectors measure the steps after it."""

    def __init__(
        self,
        sink: MetricsSink,
        cadence: Cadence,
        report: Callable[[AnomalyEvent], None] = ignore,
        throughput: ThroughputDetector | None = None,
    ) -> None:
        self.sink = sink
        self.cadence = cadence
        self.report = report
        self.spike = SpikeDetector()
        self.divergence = DivergenceDetector()
        self.collapse = CollapseDetector()
        self.throughput = throughput
        self.found: set[tuple[str, str]] = set()  # (detector, metric) found at the last step

    def check(self, step: int) -> list[AnomalyEvent]:
        """The events that start at `step`, once its metrics are all in the sink."""
        values = self.sink.values(step)
        findings: list[tuple[str, Finding]] = [
            *(("spike", finding) for finding in self.spike.check(step, values)),
            *(("divergence", finding) for finding in self.divergence.check(step, values)),
            *(("collapse", finding) for finding in self.collapse.check(step, values)),
        ]
        if self.throughput is not None:  # m3s25
            findings += [("throughput", finding) for finding in self.throughput.check(step, values)]
        found = {(detector, finding.metric) for detector, finding in findings}
        events = [
            to_event(detector, finding)
            for detector, finding in findings
            if (detector, finding.metric) not in self.found
        ]
        self.found = found
        for event in events:
            self.report(event)
            self.cadence.alert(step)
        return events

    def state(self) -> dict[str, Any]:
        """Everything the detectors remember, for a checkpoint: a resumed run's detectors
        carry on as if it had never stopped."""
        return {
            "spike": {metric: list(window) for metric, window in self.spike.windows.items()},
            "divergence": list(self.divergence.window),
            "collapse": {metric: list(window) for metric, window in self.collapse.windows.items()},
            "found": sorted(self.found),
            "alert_until": self.cadence.until,
            # m3s25: the slow steps in a row and their sum, None with no throughput detector
            "throughput": None
            if self.throughput is None
            else [self.throughput.slow, self.throughput.slow_seconds],
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Carry on from `state` (what state() returned), in a new monitor."""
        for metric, values in state["spike"].items():
            self.spike.windows[metric].extend(values)
        self.divergence.window.extend(state["divergence"])
        for metric, values in state["collapse"].items():
            self.collapse.windows[metric].extend(values)
        self.found = {(detector, metric) for detector, metric in state["found"]}
        self.cadence.until = state["alert_until"]
        slow = state.get("throughput")  # m3s25: absent from older checkpoints
        if self.throughput is not None and slow is not None:
            self.throughput.slow, self.throughput.slow_seconds = slow
