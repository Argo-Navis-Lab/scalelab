"""A run that `scalelab run` finished loads by its run_id: the command and the loader agree on
where a run's checkpoints live."""

from pathlib import Path

import pytest

from scalelab.cli import main
from scalelab.core.config import load_config
from scalelab.core.registry import Registry
from scalelab.rl.finished import load_finished_run

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"


def test_a_run_finished_by_scalelab_run_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    home = tmp_path.resolve()
    monkeypatch.setenv("SCALELAB_HOME", str(home))
    assert main(["run", str(HELLO)]) == 0
    registry = Registry.sqlite(home / "registry.db")
    (run,) = registry.list_runs()
    finished = load_finished_run(registry, home, run.run_id)
    config = load_config(HELLO)
    assert finished.step == 16  # hello's last step
    assert (finished.model.config, finished.task) == (config.model, config.task)
