"""Two ways to run the same steps, side by side: the cost model's step for each on one device, and
whether the change is worth building, judged against how close the model's steps have come to
measured ones there."""

from pathlib import Path
from typing import Literal

from scalelab.core.config import StrictModel
from scalelab.costmodel.calibrate import Errors
from scalelab.costmodel.rlstep import StepPrediction

Change = Literal["padding-cut", "fp16"]
CHANGES: dict[Change, str] = {  # what each change does to a step
    "padding-cut": "rows sorted by length, then run in micro-batches, each padded only to its "
    "own longest row (RL's log-probs too, in rl.micro_batch rows)",
    "fp16": "autocast float16 (m3s18's graph): matmuls, attention, RoPE, the MLP's activation and "
    "the KV cache in fp16; weights, norms and the residual stream stay fp32; a GradScaler "
    "unscales the gradients before clipping",
}
FLOPS_NOTES: dict[Change, str] = {  # m3s20: what the FLOPs line means for budgets
    "padding-cut": "budgets still count the padding (m3s16)",
    "fp16": "the same matmuls (casts do none), so budgets buy the same steps",
}


def reports_folder(home: Path) -> Path:
    """Where `scalelab cost compare` saves its reports: <home>/reports/."""
    return home / "reports"


class DecisionReport(StrictModel):
    """What the cost model expects a change to do to one experiment's steps on one device."""

    change: Change
    experiment: str
    config_hash: str
    hardware: str
    baseline: StepPrediction  # the loop as it runs now
    candidate: StepPrediction  # with the change
    validated: Errors | None  # cost validate's whole steps on this device (validation.json)

    @property
    def speedup(self) -> float:
        return self.baseline.seconds / self.candidate.seconds

    @property
    def saving(self) -> float:
        """The share of a step's time the change saves."""
        return 1 - self.candidate.seconds / self.baseline.seconds

    @property
    def worth_building(self) -> bool:
        """The saving beats the model's validated error on this device, so it isn't the model's
        noise. Never on a device the model wasn't validated on."""
        return self.validated is not None and self.saving > self.validated.mape

    def markdown(self) -> str:
        before, after = self.baseline, self.candidate
        lines = [
            f"# DecisionReport: {self.change}, {self.experiment} on {self.hardware}",
            "",
            f"Config {self.config_hash}. The change: {CHANGES[self.change]}.",
            "",
            "| phase | baseline s | candidate s | change | its bound | its slowest op class |",
            "|---|---:|---:|---:|---|---|",
        ]
        for name, phase in before.phases.items():
            cut = after.phases[name]
            top = max(cut.by_class, key=cut.by_class.__getitem__)
            change = cut.seconds / phase.seconds - 1
            lines.append(
                f"| {name} | {phase.seconds:.4f} | {cut.seconds:.4f} | {change:+.1%} "
                f"| {cut.bottleneck} | {top} |"
            )
        lines += [
            f"| step | {before.seconds:.4f} | {after.seconds:.4f} | {-self.saving:+.1%} | | |",
            "",
            f"- speedup: {self.speedup:.2f}x, {3600 / before.seconds:,.0f} -> "
            f"{3600 / after.seconds:,.0f} steps an hour",
            f"- FLOPs a step: {before.flops:.3g} -> {after.flops:.3g} "
            f"({after.flops / before.flops - 1:+.1%}): {FLOPS_NOTES[self.change]}",
        ]
        if self.validated is None:
            lines.append(
                f"- error band: none, {self.hardware} has no validation (cost validate saves one)"
            )
            verdict = "not worth building on this evidence: the model is unvalidated here"
        else:
            errors = self.validated
            low, high = after.seconds * (1 - errors.mape), after.seconds * (1 + errors.mape)
            lines.append(
                f"- error band: whole steps here land within {errors.mape:.1%} on average (bias "
                f"{errors.bias:+.1%}, worst {errors.worst:+.1%}, {errors.samples} runs): the "
                f"candidate's step {low:.4f}-{high:.4f} s"
            )
            if self.worth_building:
                verdict = f"worth building: the saving, {self.saving:.1%}, is above the error"
            elif self.saving < -errors.mape:  # m3s16: slower, by more than the error
                verdict = f"not worth building: it is {-self.saving:.1%} slower, beyond the error"
            else:
                verdict = f"not worth building: the saving, {self.saving:.1%}, is within the error"
        lines.append(f"- verdict: {verdict}")
        return "\n".join(lines) + "\n"


class Measurement(StrictModel):
    """A DecisionReport's change, measured (m3s17): the same steps run on its device without the
    change and with it, each part's mean seconds a step from telemetry's laps. The report is made
    and saved before the steps run, so the prediction can't lean on them."""

    report: DecisionReport
    steps: int  # timed steps of each run
    baseline: dict[str, float]  # part -> mean seconds a step, without the change
    candidate: dict[str, float]  # with it

    @property
    def speedup(self) -> float:
        return self.baseline["step"] / self.candidate["step"]

    @property
    def saving(self) -> float:
        """The share of a step's time the change saved."""
        return 1 - self.candidate["step"] / self.baseline["step"]

    @property
    def error(self) -> float:
        """How far the predicted speedup is from the measured one, as a share of it."""
        return self.report.speedup / self.speedup - 1

    @property
    def held(self) -> bool:
        """The predicted speedup lands within the model's validated error on this device."""
        validated = self.report.validated
        return validated is not None and abs(self.error) <= validated.mape

    def markdown(self) -> str:
        report = self.report
        predicted = {
            name: (phase.seconds, report.candidate.phases[name].seconds)
            for name, phase in report.baseline.phases.items()
        }
        predicted["step"] = (report.baseline.seconds, report.candidate.seconds)
        lines = [
            f"# Measured: {report.change}, {report.experiment} on {report.hardware}",
            "",
            f"Config {report.config_hash}. The same {self.steps} steps run without the change and "
            "with it, each part's mean seconds a step, beside the prediction made before they ran.",
            "",
            "| part | predicted without s | with s | change "
            "| measured without s | with s | change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for part, (before, after) in predicted.items():
            off, on = self.baseline[part], self.candidate[part]
            lines.append(
                f"| {part} | {before:.4f} | {after:.4f} | {after / before - 1:+.1%} "
                f"| {off:.4f} | {on:.4f} | {on / off - 1:+.1%} |"
            )
        lines += [
            "",
            f"- speedup: predicted {report.speedup:.2f}x, measured {self.speedup:.2f}x, so the "
            f"prediction is off by {self.error:+.1%}",
        ]
        if report.validated is None:
            verdict = f"unjudged: {report.hardware} has no validation (cost validate saves one)"
        elif self.held:
            verdict = f"the prediction held: within the {report.validated.mape:.1%} error band"
        else:
            verdict = f"the prediction missed: beyond the {report.validated.mape:.1%} error band"
        lines.append(f"- verdict: {verdict}")
        return "\n".join(lines) + "\n"
