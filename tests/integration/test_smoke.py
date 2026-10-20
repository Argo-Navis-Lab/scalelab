"""The CI smoke test: a model of real size, m1, through 20 RL steps on the CPU, on every push,
in under 3 minutes. It catches a change that breaks RL at a real size before any GPU run."""

import math
import time
from pathlib import Path

import pytest
import torch

from scalelab.cli import main
from scalelab.core.checkpoint import latest_checkpoint, load_checkpoint
from scalelab.core.registry import Registry

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"
HELLO, RL = EXPERIMENTS / "hello.yaml", EXPERIMENTS / "rl.yaml"
# rl.yaml as it is (GRPO, a KL reference, m1 with its 256-token context, every arithmetic
# band), but 8 problems x 8 answers a step instead of 64 x 8, so that 20 steps fit in CI, and
# graded on the first 20 problems of each band.
SMOKE = ["rl.max_steps=20", "rl.prompts_per_step=8", "eval.problems=20"]


def test_twenty_rl_steps_of_m1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path))
    assert main(["run", str(HELLO)]) == 0  # the warm start: hello, m1 after 16 steps
    (warm_start,) = Registry.sqlite(tmp_path / "registry.db").list_runs()

    began = time.monotonic()
    assert main(["run", str(RL), f"rl.from_run={warm_start.run_id}", *SMOKE]) == 0
    seconds = time.monotonic() - began
    output = capsys.readouterr().out
    assert "done: ran 20 steps" in output
    assert "about 0 s to go" in output  # the time left counts down to the step limit
    assert seconds < 180, f"20 RL steps of m1 took {seconds:.0f} s: more than 3 minutes"

    registry = Registry.sqlite(tmp_path / "registry.db")
    (rl,) = [run for run in registry.list_runs() if run.run_id != warm_start.run_id]
    assert rl.status == "done"
    graded = registry.list_evals(rl.run_id)
    assert {e.step for e in graded} == {0, 20}  # the warm start, and the last step
    assert {e.task for e in graded if "/" not in e.task} == {f"arithmetic-{d}" for d in range(2, 6)}
    assert all(math.isfinite(e.value) and 0 <= e.value <= 1 for e in graded)
    latest = latest_checkpoint(tmp_path / "checkpoints" / rl.run_id)
    assert latest is not None and latest.name == "step-000000020.json"
    weights = load_checkpoint(latest, tmp_path)[1]["model"]
    assert all(torch.isfinite(tensor).all() for tensor in weights.values())  # no NaN, no inf
