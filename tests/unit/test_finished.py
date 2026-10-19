"""Finished runs: found by run_id, loaded exactly as they ended, refused when they can't be."""

from pathlib import Path

import pytest
import torch

from scalelab.core.checkpoint import save_checkpoint
from scalelab.core.config import ExperimentConfig, ModelConfig, TaskConfig
from scalelab.core.envcapture import EnvInfo, GitInfo, HardwareInfo, SoftwareInfo
from scalelab.core.registry import Registry, RunStatus
from scalelab.models.transformer import Transformer
from scalelab.rl.finished import FinishedRun, load_finished_run

TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1)
TASK = TaskConfig(name="arithmetic", difficulties=(2, 3))
CONFIG = ExperimentConfig(name="tiny", model=TINY, task=TASK)
ENV = EnvInfo(  # made up, so these tests never depend on your machine
    git=GitInfo(sha="0123456789abcdef0123456789abcdef01234567", dirty=False),
    software=SoftwareInfo(python="3.12.14", os="test-os", packages={}),
    hardware=HardwareInfo(cpu_model="Test CPU", cpu_count=4),
)


def make(seed: int) -> Transformer:
    """A tiny model whose weights depend on `seed`, without disturbing PyTorch's generator."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return Transformer(TINY)


def record(home: Path, saved: dict[int, Transformer], status: RunStatus = "done") -> str:
    """Record a run of CONFIG in the registry under `home`, save each model in `saved` (step
    -> model) in the run's checkpoint folder, and leave the run `status`. Returns its run_id."""
    registry = Registry.sqlite(home / "registry.db")
    run_id = registry.add_run(CONFIG, seed=0, env=ENV)
    for step, model in saved.items():
        save_checkpoint(home / "checkpoints" / run_id, step, {"model": model.state_dict()}, home)
    if status != "running":
        registry.finish_run(run_id, status)
    return run_id


def load(home: Path, run_id: str) -> FinishedRun:
    return load_finished_run(Registry.sqlite(home / "registry.db"), home, run_id)


# --- loading --------------------------------------------------------------------------------


def test_a_finished_run_comes_back_as_it_ended(tmp_path: Path):
    trained = make(1)
    finished = load(tmp_path, record(tmp_path, {7: trained}))
    assert finished.step == 7
    assert finished.task == TASK
    assert finished.model.config == TINY
    loaded = finished.model.state_dict()
    for name, tensor in trained.state_dict().items():
        assert torch.equal(loaded[name], tensor), name


def test_the_last_checkpoint_is_the_one_loaded(tmp_path: Path):
    finished = load(tmp_path, record(tmp_path, {3: make(1), 7: make(2)}))
    assert finished.step == 7
    assert torch.equal(finished.model.head.weight, make(2).head.weight)


def test_loading_leaves_the_random_generator_alone(tmp_path: Path):
    run_id = record(tmp_path, {7: make(1)})
    before = torch.get_rng_state()
    load(tmp_path, run_id)
    assert torch.equal(torch.get_rng_state(), before)


# --- refusing -------------------------------------------------------------------------------


def test_a_run_the_registry_never_saw_is_refused(tmp_path: Path):
    record(tmp_path, {7: make(1)})
    with pytest.raises(ValueError, match="the registry has no run abc-s0-01234567"):
        load(tmp_path, "abc-s0-01234567")


@pytest.mark.parametrize("status", ["running", "failed"])
def test_an_unfinished_run_is_refused(tmp_path: Path, status: RunStatus):
    run_id = record(tmp_path, {7: make(1)}, status)
    with pytest.raises(ValueError, match=f"is {status}, not done"):
        load(tmp_path, run_id)


def test_a_run_whose_checkpoints_are_not_in_home_is_refused(tmp_path: Path):
    run_id = record(tmp_path, {7: make(1)})
    registry = Registry.sqlite(tmp_path / "registry.db")
    with pytest.raises(ValueError, match="has no checkpoint in"):
        load_finished_run(registry, tmp_path / "elsewhere", run_id)  # a home without them
