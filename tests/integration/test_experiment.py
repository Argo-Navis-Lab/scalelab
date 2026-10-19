"""Running an experiment: a warm start built from nothing but its config and seed."""

from pathlib import Path

import torch

from scalelab.core.checkpoint import latest_checkpoint
from scalelab.core.config import ExperimentConfig, load_config
from scalelab.rl.experiment import make_task, run_experiment
from scalelab.rl.warmstart_loop import StepMetrics

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


def test_each_task_keeps_its_own_eval_questions_out():
    for name in ("arithmetic", "countdown"):
        task, excluded = make_task(name)
        assert task.name == name
        assert len(excluded) == 4 * 1000  # 4 bands of 1000
