"""Running an experiment: a warm start built from nothing but its config and seed."""

from pathlib import Path

import torch

from scalelab.core.checkpoint import latest_checkpoint
from scalelab.core.config import ExperimentConfig, load_config
from scalelab.models.transformer import Transformer
from scalelab.rl.evaluation import EvalResult
from scalelab.rl.experiment import grade_by_part, load_eval_sets, make_task, run_experiment
from scalelab.rl.warmstart_loop import StepMetrics
from scalelab.tasks.arithmetic import ArithmeticProblem
from scalelab.tasks.countdown import CountdownProblem
from scalelab.tasks.evalsets import read_eval_set

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"


def run_after_noise(
    config: ExperimentConfig, seed: int, home: Path, folder: str, noise: int
) -> list[StepMetrics]:
    """A run that starts with PyTorch's generator wherever `noise` left it, as in a fresh
    process: the run's own seed must decide everything anyway."""
    with torch.random.fork_rng():  # put the generator back afterwards, for the other tests
        torch.manual_seed(noise)
        return run_experiment(config, seed, home / folder, home)


def test_hello_warm_starts_m1(tmp_path: Path):
    metrics = run_experiment(load_config(HELLO), 0, tmp_path / "ckpt", tmp_path)
    assert metrics[-1].tokens >= 20_000  # hello's budget
    assert metrics[-1].loss < metrics[0].loss
    last = tmp_path / "ckpt" / f"step-{len(metrics):09d}.json"
    assert latest_checkpoint(tmp_path / "ckpt") == last


def test_the_seed_decides_everything(tmp_path: Path):
    # Two CPU runs with the same seed match exactly; another seed gives another run.
    config = load_config(HELLO)
    first = run_after_noise(config, 0, tmp_path, "a", noise=1)
    assert run_after_noise(config, 0, tmp_path, "b", noise=2) == first
    assert run_after_noise(config, 1, tmp_path, "c", noise=1) != first


def test_countdown_warm_starts_too(tmp_path: Path):
    config = load_config(HELLO, ["task=countdown"])
    assert config.task.difficulties == (3, 4, 5, 6)  # from configs/tasks/countdown.yaml
    metrics = run_experiment(config, 0, tmp_path / "ckpt", tmp_path)
    assert metrics[-1].tokens >= 20_000


def test_each_task_comes_with_the_type_of_its_problems():
    task, problem_type = make_task("arithmetic")
    assert (task.name, problem_type) == ("arithmetic", ArithmeticProblem)
    task, problem_type = make_task("countdown")
    assert (task.name, problem_type) == ("countdown", CountdownProblem)


def test_the_eval_sets_are_those_of_the_training_difficulties():
    task, problem_type = make_task("arithmetic")
    sets = load_eval_sets(task, problem_type, [2, 5, 9], problems=20)
    assert list(sets) == ["arithmetic-2", "arithmetic-5"]  # there's no 9-digit eval set
    assert sets["arithmetic-2"] == read_eval_set("arithmetic-2", ArithmeticProblem)[:20]


def test_arithmetic_is_graded_by_operation_and_countdown_whole():
    with torch.random.fork_rng():
        model = Transformer(load_config(HELLO).model)  # untrained: only the counts matter here
    task, problem_type = make_task("arithmetic")
    results = grade_by_part(model, task, load_eval_sets(task, problem_type, [2, 3], 10))
    assert [(r.name, r.n) for r in results] == [
        ("arithmetic-2", 10),
        ("arithmetic-2/*", 3),
        ("arithmetic-2/+", 5),
        ("arithmetic-2/-", 2),
        ("arithmetic-3", 10),
        ("arithmetic-3/*", 1),
        ("arithmetic-3/+", 6),
        ("arithmetic-3/-", 3),
    ]
    task, problem_type = make_task("countdown")
    results = grade_by_part(model, task, load_eval_sets(task, problem_type, [3], 10))
    assert [(r.name, r.n) for r in results] == [("countdown-3", 10)]


# --- grading --------------------------------------------------------------------------------


def test_grading_comes_every_so_often_and_after_the_last_step(tmp_path: Path):
    graded: list[tuple[int, list[str], list[int]]] = []

    def report(metrics: StepMetrics, results: list[EvalResult]) -> None:
        graded.append((metrics.step, [r.name for r in results], [r.n for r in results]))

    config = load_config(HELLO, ["eval.every=5", "eval.problems=5"])
    run_experiment(config, 0, tmp_path / "ckpt", tmp_path, report=report)
    names = ["arithmetic-2", "arithmetic-3", "arithmetic-4", "arithmetic-5"]
    assert graded == [(step, names, [5] * 4) for step in (5, 10, 15, 16)]  # 16 is the last


def test_grading_changes_nothing_about_training(tmp_path: Path):
    config = load_config(HELLO, ["eval.every=5", "eval.problems=5"])
    graded = run_experiment(config, 0, tmp_path / "a", tmp_path, report=lambda m, r: None)
    assert graded == run_experiment(config, 0, tmp_path / "b", tmp_path)  # never graded
