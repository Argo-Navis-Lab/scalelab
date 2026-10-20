"""Running an RL experiment: from a warm start's model, graded at step 0 (the warm start's
own grade), every so often and at the end, greedy and sampled, whole and by operation."""

from pathlib import Path

import pytest
import torch

from scalelab.core.config import ExperimentConfig, ModelConfig, load_config
from scalelab.models.transformer import Transformer
from scalelab.rl.evaluation import EvalResult, SampledResult
from scalelab.rl.experiment import run_rl_experiment
from scalelab.rl.loop import RLStepMetrics

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"
# rl.yaml (GRPO, so a KL reference too) made small: a tiny model, 4 problems x 2 answers a
# step, 10 problems of each eval set, graded every 2 steps.
SMALL = [
    "model.name=tiny",
    "model.n_layers=2",
    "model.d_model=64",
    "model.n_heads=1",
    "model.context=32",
    "task.difficulties=[2]",
    "rl.prompts_per_step=4",
    "rl.group_size=2",
    "rl.flops=2e9",
    "eval.every=2",
    "eval.problems=10",
]

Grading = tuple[int, list[EvalResult], list[SampledResult]]


def config() -> ExperimentConfig:
    return load_config(EXPERIMENTS / "rl.yaml", SMALL)


def warm_start(model: ModelConfig) -> Transformer:
    """Stands in for rl.from_run's model: the same weights every time, made without
    disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(model)


def run(home: Path, folder: str, seed: int = 0) -> tuple[list[RLStepMetrics], list[Grading]]:
    """An RL run, and every grading it reported: (step, greedy results, sampled results)."""
    gradings: list[Grading] = []

    def report(
        metrics: RLStepMetrics | None, greedy: list[EvalResult], sampled: list[SampledResult]
    ) -> None:
        gradings.append((0 if metrics is None else metrics.step, greedy, sampled))

    small = config()
    metrics = run_rl_experiment(
        small, seed, warm_start(small.model), home / folder, home, report=report
    )
    return metrics, gradings


def test_gradings_come_at_step_0_every_so_often_and_at_the_end(tmp_path: Path):
    metrics, gradings = run(tmp_path, "a")
    last = len(metrics)
    expected = [0] + [step for step in range(1, last + 1) if step % 2 == 0 or step == last]
    assert [step for step, _, _ in gradings] == expected


def test_each_grading_is_greedy_and_sampled_by_operation(tmp_path: Path):
    _, gradings = run(tmp_path, "a")
    step, greedy, sampled = gradings[0]
    names = ["arithmetic-2", "arithmetic-2/*", "arithmetic-2/+", "arithmetic-2/-"]
    assert step == 0
    assert [(r.name, r.n) for r in greedy] == list(zip(names, [10, 3, 5, 2], strict=True))
    assert [(r.name, r.n, r.samples) for r in sampled] == [
        (name, n, 4) for name, n in zip(names, [10, 3, 5, 2], strict=True)
    ]


def test_grading_changes_nothing_about_rl(tmp_path: Path):
    graded, _ = run(tmp_path, "a")
    small = config()
    assert run_rl_experiment(small, 0, warm_start(small.model), tmp_path / "b", tmp_path) == graded


def test_every_seed_gets_the_same_grade_for_the_warm_start(tmp_path: Path):
    # R_0 is one number per warm start: the step-0 grade doesn't depend on the RL seed.
    _, seed_0 = run(tmp_path, "a", seed=0)
    _, seed_1 = run(tmp_path, "b", seed=1)
    assert seed_0[0] == seed_1[0]


def test_a_finished_run_is_not_graded_again(tmp_path: Path):
    run(tmp_path, "a")
    assert run(tmp_path, "a") == ([], [])


def test_a_warm_start_of_another_model_is_refused(tmp_path: Path):
    other = ModelConfig(name="other", n_layers=1, d_model=64, n_heads=1, context=32)
    with pytest.raises(ValueError, match="the warm start is model other, the config's is tiny"):
        run_rl_experiment(config(), 0, warm_start(other), tmp_path / "a", tmp_path)


def test_a_warm_start_experiment_is_not_run_as_rl(tmp_path: Path):
    hello = load_config(EXPERIMENTS / "hello.yaml")
    with pytest.raises(ValueError, match="hello is not an RL experiment"):
        run_rl_experiment(hello, 0, warm_start(hello.model), tmp_path / "a", tmp_path)
