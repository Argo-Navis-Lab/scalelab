import json
from pathlib import Path

from scalelab.rl.evaluation import EvalResult, SampledResult
from scalelab.rl.gradings import Grading, grading_path, gradings_folder, load_gradings, save_grading

RUN = "0123456789abcdef-s0-01234567"


def grading(
    step: int, run_id: str = RUN, solved: tuple[bool, ...] = (True, False, True)
) -> Grading:
    """A small grading: 3 problems, the first of them a multiplication."""
    greedy = (
        EvalResult(name="arithmetic-3", solved=solved),
        EvalResult(name="arithmetic-3/*", solved=solved[:1]),
    )
    sampled = (SampledResult(name="arithmetic-3", samples=4, correct=(4, 0, 2)),)
    return Grading(run_id=run_id, step=step, greedy=greedy, sampled=sampled)


def test_a_grading_comes_back_as_it_was_saved(tmp_path: Path):
    saved = grading(7)
    path = save_grading(tmp_path, saved)
    assert path == tmp_path / "gradings" / RUN / "grading-000000007.json"
    assert path == grading_path(tmp_path, RUN, 7)
    assert load_gradings(tmp_path, RUN) == [saved]


def test_the_file_is_plain_json_on_one_line(tmp_path: Path):
    text = save_grading(tmp_path, grading(0)).read_text()
    assert text.count("\n") == 1  # one line, and its end
    data = json.loads(text)  # any tool can read it
    assert (data["run_id"], data["step"]) == (RUN, 0)
    assert data["greedy"][0] == {"name": "arithmetic-3", "solved": [True, False, True]}
    assert data["sampled"][0] == {"name": "arithmetic-3", "samples": 4, "correct": [4, 0, 2]}


def test_no_gradings_file_looks_like_a_checkpoint_manifest(tmp_path: Path):
    save_grading(tmp_path, grading(0))
    assert list(tmp_path.rglob("step-*.json")) == []  # pruning reads those as manifests


def test_gradings_come_in_step_order_and_only_the_runs_own(tmp_path: Path):
    for step in (10, 2, 0):
        save_grading(tmp_path, grading(step))
    save_grading(tmp_path, grading(5, run_id="fedcba9876543210-s0-01234567"))
    assert [g.step for g in load_gradings(tmp_path, RUN)] == [0, 2, 10]


def test_a_step_graded_again_replaces_its_file(tmp_path: Path):
    save_grading(tmp_path, grading(5, solved=(False, False, False)))
    again = grading(5, solved=(True, True, False))
    save_grading(tmp_path, again)
    assert load_gradings(tmp_path, RUN) == [again]
    assert list(gradings_folder(tmp_path, RUN).iterdir()) == [grading_path(tmp_path, RUN, 5)]


def test_a_run_without_gradings_has_none(tmp_path: Path):
    assert load_gradings(tmp_path, RUN) == []
