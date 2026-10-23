"""Gradings kept problem by problem: each grading of an RL run is one JSON file under
<home>/gradings/<run_id>/, holding every result with its per-problem outcomes. The evals
table keeps a rate, n and standard error per result; comparing two gradings problem by
problem (a paired comparison) needs the outcomes themselves."""

from pathlib import Path

from scalelab.core.config import StrictModel
from scalelab.core.storage import atomic_write_text
from scalelab.rl.evaluation import EvalResult, SampledResult


class Grading(StrictModel):
    """One grading of one run at one step: its greedy and sampled results, whole eval sets
    and parts, as graded (the results its rows in the evals table sum up)."""

    run_id: str
    step: int  # 0 is the warm start's own grading
    greedy: tuple[EvalResult, ...]
    sampled: tuple[SampledResult, ...]


def gradings_folder(home: Path, run_id: str) -> Path:
    return home / "gradings" / run_id


def grading_path(home: Path, run_id: str, step: int) -> Path:
    # Not "step-*.json": checkpoint pruning reads every file so named under <home> as a
    # checkpoint manifest.
    return gradings_folder(home, run_id) / f"grading-{step:09d}.json"


def save_grading(home: Path, grading: Grading) -> Path:
    """Write one grading, replacing the file of a step graded again (a resumed run redoes
    the steps after its last checkpoint). Compact JSON on one line: pretty JSON would give
    every problem a line of its own."""
    path = grading_path(home, grading.run_id, grading.step)
    return atomic_write_text(path, grading.model_dump_json() + "\n")


def load_gradings(home: Path, run_id: str) -> list[Grading]:
    """Every grading of a run, in step order (the zero-padded names sort that way); none if
    the run has no folder."""
    paths = sorted(gradings_folder(home, run_id).glob("grading-*.json"))
    return [Grading.model_validate_json(path.read_text()) for path in paths]
