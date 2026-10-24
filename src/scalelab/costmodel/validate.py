"""How close the cost model's whole steps come to measured ones: each run's predicted and
measured seconds a step, phase by phase, and the errors over many runs."""

from collections.abc import Mapping, Sequence

from scalelab.core.config import StrictModel
from scalelab.costmodel.calibrate import Errors, errors_between
from scalelab.costmodel.rlstep import StepPrediction

PARTS = ("generate", "logprobs", "train", "step")  # the phases the model times, then the step
TARGET_MAPE = 0.15  # M3's exit: whole steps within 15% on average


class StepCheck(StrictModel):
    """One run's steps: the predicted and the measured seconds a step, for each part."""

    run_id: str
    steps: int  # how many measured steps the measured means cover
    predicted: dict[str, float]
    measured: dict[str, float]

    @property
    def errors(self) -> dict[str, float]:
        """Each part's error, predicted / measured - 1: below 0, the model runs fast."""
        return {part: self.predicted[part] / self.measured[part] - 1 for part in PARTS}


def check_step(
    run_id: str, prediction: StepPrediction, steps: int, measured: Mapping[str, float]
) -> StepCheck:
    """A run's measured seconds a step (every part in PARTS) against its predicted step. The
    measured step also holds what the model leaves out: drawing problems, scoring, telemetry."""
    missing = sorted(set(PARTS) - set(measured))
    if missing:
        raise ValueError(f"{run_id}: no measured {', '.join(missing)}")
    predicted = {name: phase.seconds for name, phase in prediction.phases.items()}
    predicted["step"] = prediction.seconds
    return StepCheck(
        run_id=run_id,
        steps=steps,
        predicted=predicted,
        measured={part: measured[part] for part in PARTS},
    )


def summarize(checks: Sequence[StepCheck]) -> dict[str, Errors]:
    """Each part's MAPE, bias and worst error over the runs (at least one), each run once."""
    return {
        part: errors_between(
            [check.predicted[part] for check in checks], [check.measured[part] for check in checks]
        )
        for part in PARTS
    }
