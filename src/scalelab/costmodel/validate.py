"""How close the cost model's whole steps come to measured ones: each run's predicted and
measured seconds a step, phase by phase, and the errors over many runs."""

from collections.abc import Mapping, Sequence

from scalelab.core.config import StrictModel
from scalelab.costmodel.calibrate import Errors, errors_between
from scalelab.costmodel.rlstep import StepPrediction

PARTS = ("generate", "logprobs", "train", "step")  # the phases the model times, then the step
TARGET_MAPE = 0.15  # M3's exit: whole steps within 15% on average
VALIDATION_FILE = "validation.json"  # m3s15: cost validate's errors, beside calibration.json


class Validation(StrictModel):
    """cost validate's errors on one device, saved for cost compare's error band (m3s15)."""

    hardware: str
    errors: dict[str, Errors]  # each part's, over the runs that timed it
    # m3s23: with runs of several runtimes, also each one's step, as "step, fp16 cut"


class StepCheck(StrictModel):
    """One run's steps: the predicted and the measured seconds a step, for each part it has (a
    warm start's: train and step, m3s11)."""

    run_id: str
    steps: int  # how many measured steps the measured means cover
    predicted: dict[str, float]
    measured: dict[str, float]

    @property
    def errors(self) -> dict[str, float]:
        """Each part's error, predicted / measured - 1: below 0, the model runs fast."""
        return {part: self.predicted[part] / self.measured[part] - 1 for part in self.predicted}


def check_step(
    run_id: str, prediction: StepPrediction, steps: int, measured: Mapping[str, float]
) -> StepCheck:
    """A run's measured seconds a step (each phase the prediction has, then the step) against
    its predicted step. The measured step also holds what the model leaves out: drawing
    problems, scoring, telemetry."""
    predicted = {name: phase.seconds for name, phase in prediction.phases.items()}
    predicted["step"] = prediction.seconds
    missing = sorted(set(predicted) - set(measured))
    if missing:
        raise ValueError(f"{run_id}: no measured {', '.join(missing)}")
    return StepCheck(
        run_id=run_id,
        steps=steps,
        predicted=predicted,
        measured={part: measured[part] for part in predicted},
    )


def summarize(checks: Sequence[StepCheck]) -> dict[str, Errors]:
    """Each part's MAPE, bias and worst error over the runs that have it (generate and logprobs:
    RL runs alone), each run once. A part no run has is left out."""
    summary: dict[str, Errors] = {}
    for part in PARTS:
        timed = [check for check in checks if part in check.predicted]
        if timed:
            predicted = [check.predicted[part] for check in timed]
            summary[part] = errors_between(predicted, [check.measured[part] for check in timed])
    return summary
