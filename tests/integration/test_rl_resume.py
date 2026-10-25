"""The resume contract for RL: interrupted and resumed on CPU, an RL run is bit-identical to
one that ran straight through, sampling included."""

import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from scalelab.core.checkpoint import CheckpointPolicy, latest_checkpoint, load_checkpoint
from scalelab.core.config import AlgoConfig, ModelConfig, Precision, RLConfig
from scalelab.core.determinism import seed_everything
from scalelab.models.transformer import Transformer
from scalelab.rl.loop import RLStepMetrics, rl_train
from scalelab.tasks.arithmetic import ArithmeticProblem, ArithmeticTask
from scalelab.tasks.rewards import Reward

GRPO = Path(__file__).resolve().parents[2] / "configs" / "algos" / "grpo.yaml"
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1, context=32)
CONFIG = RLConfig(from_run="0123456789abcdef-s0-01234567", flops=2e10, prompts_per_step=8, lr=1e-2)


class Hurry(ArithmeticTask):
    """Arithmetic's prompts with a made-up reward: 1 for stopping at once (<eos> as the first
    token), 0 for anything else, so that a tiny untrained model has something to learn."""

    def score(self, problem: ArithmeticProblem, completion: str) -> Reward:
        if completion == "":
            return Reward(value=1.0, format=0.0, parsed=None, flags=())
        return Reward(value=0.0, format=0.0, parsed=None, flags=("no_answer",))


def session(
    home: Path, folder: str, stop_after: int | None = None, precision: Precision = "fp32"
) -> list[RLStepMetrics]:
    """One session of a run, as a fresh process would start it: everything built again from
    the seed (the model as the warm start would hand it over), then the latest checkpoint (if
    any) picks up where the last session stopped. In `precision` (m3s22)."""
    seeds = seed_everything(0, deterministic=True)
    model = Transformer(TINY)
    return rl_train(
        model,
        Hurry(),
        [2],
        AlgoConfig.model_validate(yaml.safe_load(GRPO.read_text())),
        CONFIG,
        np.random.default_rng(seeds.data),
        torch.Generator().manual_seed(seeds.sampling),
        home / folder,
        home=home,
        policy=CheckpointPolicy(every_steps=3),
        reference=copy.deepcopy(model),  # the starting weights, as in a real run
        stop_after=stop_after,
        precision=precision,
    )


def final_weights(home: Path, folder: str) -> dict[str, torch.Tensor]:
    latest = latest_checkpoint(home / folder)
    assert latest is not None
    return load_checkpoint(latest, home)[1]["model"]


def test_stop_and_resume_is_bit_identical(tmp_path: Path):
    uninterrupted = session(tmp_path, "a")
    before = session(tmp_path, "b", stop_after=3)
    after = session(tmp_path, "b")
    assert (len(before), len(after)) == (3, len(uninterrupted) - 3)
    assert before + after == uninterrupted  # every reward, loss and KL, exactly
    a, b = final_weights(tmp_path, "a"), final_weights(tmp_path, "b")
    assert all(torch.equal(a[name], b[name]) for name in a)


def test_interrupted_twice_is_still_bit_identical(tmp_path: Path):
    uninterrupted = session(tmp_path, "a")
    first = session(tmp_path, "b", stop_after=2)
    second = session(tmp_path, "b", stop_after=5)
    third = session(tmp_path, "b")
    assert first + second + third == uninterrupted


def test_a_finished_rl_run_has_nothing_left_to_do(tmp_path: Path):
    assert len(session(tmp_path, "a")) > 5
    assert session(tmp_path, "a") == []


def test_an_fp16_run_resumes_with_its_loss_scale(tmp_path: Path):  # m3s22
    uninterrupted = session(tmp_path, "a", precision="fp16")
    before = session(tmp_path, "b", stop_after=3, precision="fp16")
    after = session(tmp_path, "b", precision="fp16")
    assert before + after == uninterrupted
    assert uninterrupted != session(tmp_path, "c")  # fp32 rounds otherwise
    latest = latest_checkpoint(tmp_path / "b")
    assert latest is not None
    scaler = load_checkpoint(latest, home=tmp_path)[1]["scaler"]
    # No update overflowed, so the scale is still 2**16, counting every step towards doubling.
    assert (scaler["scale"], scaler["_growth_tracker"]) == (2.0**16, len(uninterrupted))
