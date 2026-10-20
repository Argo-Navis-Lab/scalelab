"""Hook point tests: when the trainer and the loop call their hooks, in what order, what each
hook point shows, and that watching changes nothing about training."""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scalelab.core.config import AlgoConfig, ModelConfig, RLConfig, SamplingParams
from scalelab.core.determinism import seed_everything
from scalelab.models.transformer import Transformer
from scalelab.rl.loop import RLStepMetrics, rl_train
from scalelab.rl.rollout import RolloutBatch, generate
from scalelab.rl.trainer import UpdateMetrics, rl_update
from scalelab.tasks.arithmetic import ArithmeticTask
from scalelab.telemetry.hooks import (
    BackwardEvent,
    ForwardEvent,
    Hook,
    OptimizerStepEvent,
    RolloutEvent,
)

ALGOS = Path(__file__).resolve().parents[2] / "configs" / "algos"
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1, context=32)
TASK = ArithmeticTask()
ADVANTAGES = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25])  # made up
SETTINGS = {
    "from_run": "0123456789abcdef-s0-01234567",
    "flops": 1e15,
    "max_steps": 3,
    "prompts_per_step": 8,
    "lr": 1e-2,
}


class Recorder(Hook):
    """A hook that keeps every event it is shown, and the order they came in."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []  # (hook point, step, epoch)
        self.rollouts: list[RolloutEvent] = []
        self.forwards: list[ForwardEvent] = []

    def on_rollout(self, event: RolloutEvent) -> None:
        self.calls.append(("rollout", event.step, -1))  # -1: a rollout has no epoch
        self.rollouts.append(event)

    def on_forward(self, event: ForwardEvent) -> None:
        self.calls.append(("forward", event.step, event.epoch))
        self.forwards.append(event)

    def on_backward(self, event: BackwardEvent) -> None:
        self.calls.append(("backward", event.step, event.epoch))

    def on_optimizer_step(self, event: OptimizerStepEvent) -> None:
        self.calls.append(("optimizer_step", event.step, event.epoch))


def preset(name: str) -> AlgoConfig:
    return AlgoConfig.model_validate(yaml.safe_load((ALGOS / f"{name}.yaml").read_text()))


def make() -> Transformer:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(TINY)


def roll(model: Transformer) -> RolloutBatch:
    """8 answers: 2 to each of 4 two-digit problems."""
    problems = TASK.sample(np.random.default_rng(0), 4, 2)
    prompts = [TASK.format_prompt(p) for p in problems]
    return generate(model, prompts, 2, SamplingParams(), torch.Generator().manual_seed(0))


def update(
    model: Transformer,
    hooks: list[Hook],
    *,
    epochs: int = 1,
    micro_batch: int = 64,
    grad_clip: float = 1.0,
    nothing_to_learn: bool = False,
) -> list[UpdateMetrics]:
    """rl_update with GRPO's settings (a KL penalty, so a reference) on 8 rolled-out rows,
    told it is step 7."""
    batch = roll(model)
    mask = batch.response_mask
    if nothing_to_learn:
        mask = torch.zeros_like(mask)
    return rl_update(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        batch,
        ADVANTAGES,
        mask,
        preset("grpo"),
        temperature=1.0,
        epochs=epochs,
        micro_batch=micro_batch,
        grad_clip=grad_clip,
        reference=copy.deepcopy(model),
        hooks=hooks,
        step=7,
    )


def train(hooks: list[Hook], home: Path) -> tuple[list[RLStepMetrics], Transformer]:
    """3 RL steps of RLOO from a tiny untrained model, 8 problems x 8 answers each."""
    seeds = seed_everything(0, deterministic=True)
    model = Transformer(TINY)
    metrics = rl_train(
        model,
        TASK,
        [2],
        preset("rloo"),
        RLConfig.model_validate(SETTINGS),
        np.random.default_rng(seeds.data),
        torch.Generator().manual_seed(seeds.sampling),
        home / "ckpt",
        home,
        hooks=hooks,
    )
    return metrics, model


# --- the trainer's hook points --------------------------------------------------------------


def test_the_trainer_calls_its_hooks_in_order():
    recorder = Recorder()
    update(make(), [recorder], epochs=2, micro_batch=4)  # 2 micro-batches an update
    one_update = ["forward", "forward", "backward", "optimizer_step"]
    assert [point for point, _, _ in recorder.calls] == one_update * 2
    assert {step for _, step, _ in recorder.calls} == {7}
    assert [epoch for _, _, epoch in recorder.calls] == [0] * 4 + [1] * 4


def test_on_forward_shows_numbers_not_the_graph_behind_them():
    recorder = Recorder()
    update(make(), [recorder], micro_batch=4)
    for event in recorder.forwards:
        assert not event.logp.requires_grad and not event.loss.requires_grad
        assert event.loss.dim() == 0 and math.isfinite(event.loss.item())
        assert event.logp.shape == event.old_logp.shape == event.mask.shape
        assert event.ref_logp is not None  # GRPO has a KL penalty
        assert torch.equal(event.mask, event.batch.response_mask)
        assert len(event.advantages) == len(event.batch.finish_reasons) == 4


def test_on_backward_sees_the_whole_gradient_before_clipping():
    class Norm(Hook):
        def on_backward(self, event: BackwardEvent) -> None:
            squares = [p.grad.pow(2).sum() for p in event.model.parameters() if p.grad is not None]
            self.norm = torch.stack(squares).sum().sqrt().item()

        def on_optimizer_step(self, event: OptimizerStepEvent) -> None:
            self.grad_norm = event.grad_norm

    norm = Norm()
    update(make(), [norm], micro_batch=4, grad_clip=1e-3)  # clipping will shrink it
    assert norm.norm > 1e-3  # seen before the clip...
    assert norm.norm == pytest.approx(norm.grad_norm, rel=1e-5)  # ...and whole


def test_on_optimizer_step_comes_after_the_weights_change():
    class Before(Hook):
        def on_backward(self, event: BackwardEvent) -> None:
            self.weights = {k: v.clone() for k, v in event.model.state_dict().items()}

        def on_optimizer_step(self, event: OptimizerStepEvent) -> None:
            now = event.model.state_dict()
            self.changed = any(not torch.equal(self.weights[k], now[k]) for k in now)

    before = Before()
    update(make(), [before])
    assert before.changed


def test_nothing_to_learn_calls_no_trainer_hooks():
    recorder = Recorder()
    assert update(make(), [recorder], nothing_to_learn=True) == []
    assert recorder.calls == []


# --- the loop's hook point ------------------------------------------------------------------


def test_the_loop_shows_each_steps_answers_before_training_on_them(tmp_path: Path):
    recorder = Recorder()
    train([recorder], tmp_path)
    one_step = ["rollout", "forward", "backward", "optimizer_step"]
    assert [point for point, _, _ in recorder.calls] == one_step * 3  # 64 rows: 1 micro-batch
    assert [step for _, step, _ in recorder.calls] == [1] * 4 + [2] * 4 + [3] * 4
    for event in recorder.rollouts:
        assert len(event.batch.finish_reasons) == 64  # 8 problems x 8 answers
        assert event.rewards.shape == (64,)


def test_watching_changes_nothing_about_training(tmp_path: Path):
    watched, model = train([Recorder()], tmp_path / "a")
    unwatched, other = train([], tmp_path / "b")
    assert watched == unwatched
    weights, others = model.state_dict(), other.state_dict()
    assert all(torch.equal(weights[name], others[name]) for name in weights)
