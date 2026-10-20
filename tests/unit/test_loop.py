"""RL loop tests: problems drawn as warm start draws them, a made-up reward learned within a
few steps, the FLOP budget, steps with nothing to learn, and the KL's reference."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scalelab.core.config import AlgoConfig, ModelConfig, RLConfig
from scalelab.core.determinism import seed_everything
from scalelab.models.transformer import Transformer
from scalelab.rl.loop import RLStepMetrics, rl_train, sample_problems
from scalelab.rl.warmstart import eval_prompts, sample_examples
from scalelab.tasks.arithmetic import ArithmeticProblem, ArithmeticTask
from scalelab.tasks.rewards import Reward

ALGOS = Path(__file__).resolve().parents[2] / "configs" / "algos"
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1, context=32)
SETTINGS = {"from_run": "0123456789abcdef-s0-01234567", "prompts_per_step": 8, "lr": 1e-2}


class Hurry(ArithmeticTask):
    """Arithmetic's prompts with a made-up reward: 1 for stopping at once (<eos> as the first
    token), 0 for anything else. An untrained model does that about 1 time in 22, so its
    groups differ now and then, and RL learns it within a few steps."""

    def score(self, problem: ArithmeticProblem, completion: str) -> Reward:
        if completion == "":
            return Reward(value=1.0, format=0.0, parsed=None, flags=())
        return Reward(value=0.0, format=0.0, parsed=None, flags=("no_answer",))


class Hopeless(ArithmeticTask):
    """Arithmetic's prompts, with every answer scored 0: nothing to learn. (Real arithmetic
    won't do for that: now and then an untrained model's answer is right by luck.)"""

    def score(self, problem: ArithmeticProblem, completion: str) -> Reward:
        return Reward(value=0.0, format=0.0, parsed=None, flags=("wrong_answer",))


def preset(name: str) -> AlgoConfig:
    return AlgoConfig.model_validate(yaml.safe_load((ALGOS / f"{name}.yaml").read_text()))


def train(
    home: Path,
    algo: AlgoConfig,
    flops: float,
    task: ArithmeticTask | None = None,
    reference: bool = True,
) -> tuple[Transformer, Transformer, list[RLStepMetrics]]:
    """RL from an untrained tiny model, seed 0. Returns the model before, after, and each
    step's metrics."""
    seeds = seed_everything(0, deterministic=True)
    model = Transformer(TINY)
    start = copy.deepcopy(model)
    metrics = rl_train(
        model,
        task or Hurry(),
        [2],
        algo,
        RLConfig.model_validate(SETTINGS | {"flops": flops}),
        np.random.default_rng(seeds.data),
        torch.Generator().manual_seed(seeds.sampling),
        home / "ckpt",
        home,
        reference=start if reference else None,
    )
    return start, model, metrics


# --- drawing problems -----------------------------------------------------------------------


def test_problems_are_drawn_as_warm_start_draws_them():
    task, excluded = ArithmeticTask(), eval_prompts(ArithmeticTask(), ArithmeticProblem)
    problems = sample_problems(task, np.random.default_rng(0), 50, [2, 3], excluded)
    examples = sample_examples(task, np.random.default_rng(0), 50, [2, 3], 256, excluded)
    assert [task.format_prompt(p) for p in problems] == [e.prompt for e in examples]


def test_an_excluded_prompt_is_skipped_and_another_drawn():
    task = ArithmeticTask()
    first = sample_problems(task, np.random.default_rng(0), 5, [2])
    excluded = frozenset({tuple(task.format_prompt(first[0]))})
    again = sample_problems(task, np.random.default_rng(0), 5, [2], excluded)
    assert again[:4] == first[1:]  # the excluded one is gone, the others move up


# --- learning -------------------------------------------------------------------------------


def test_rl_learns_a_made_up_reward(tmp_path: Path):
    _, _, metrics = train(tmp_path, preset("rloo"), flops=2e10)
    assert metrics[0].reward < 0.25  # an untrained model stops at once about 1 time in 22
    assert metrics[-1].reward > 0.9  # ...and within a few steps nearly always


def test_the_budget_ends_the_loop_at_the_step_that_crosses_it(tmp_path: Path):
    _, _, metrics = train(tmp_path, preset("rloo"), flops=1e10)
    flops = [m.flops for m in metrics]
    assert flops == sorted(set(flops))  # every step adds some
    assert flops[-2] < 1e10 <= flops[-1]
    assert [m.step for m in metrics] == list(range(1, len(metrics) + 1))


def test_the_kl_to_the_reference_is_reported(tmp_path: Path):
    _, _, metrics = train(tmp_path, preset("grpo"), flops=1e10)
    assert metrics[0].kl < 1e-6  # the first update starts at the reference
    assert metrics[-1].kl > 0


def test_a_kl_penalty_without_a_reference_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="needs the reference model"):
        train(tmp_path, preset("grpo"), flops=5e9, reference=False)


# --- nothing to learn -----------------------------------------------------------------------


def test_with_nothing_to_learn_nothing_changes(tmp_path: Path):
    # Every group's rewards are all 0, so dynamic sampling leaves nothing to train on.
    algo = preset("dapo").model_copy(update={"overlong": "none"})  # no penalty to vary
    before, after, metrics = train(tmp_path, algo, flops=5e9, task=Hopeless())
    assert {(m.reward, m.trained, m.loss, m.grad_norm) for m in metrics} == {(0.0, 0, 0.0, 0.0)}
    weights, now = before.state_dict(), after.state_dict()
    assert all(torch.equal(weights[name], now[name]) for name in weights)


# --- added in m2s21: a hook after each step -------------------------------------------------


def test_on_step_sees_each_steps_metrics_before_it_is_saved(tmp_path: Path):
    seen: list[tuple[RLStepMetrics, bool]] = []

    def on_step(metrics: RLStepMetrics) -> None:
        saved = (tmp_path / "ckpt" / f"step-{metrics.step:09d}.json").exists()
        seen.append((metrics, saved))

    seeds = seed_everything(0, deterministic=True)
    metrics = rl_train(
        Transformer(TINY),
        Hurry(),
        [2],
        preset("rloo"),
        RLConfig.model_validate(SETTINGS | {"flops": 5e9}),
        np.random.default_rng(seeds.data),
        torch.Generator().manual_seed(seeds.sampling),
        tmp_path / "ckpt",
        tmp_path,
        on_step=on_step,
    )
    assert [m for m, _ in seen] == metrics
    assert not any(saved for _, saved in seen)  # not even the last step, which is always saved


# --- added in m2s24: a step limit -----------------------------------------------------------


def test_max_steps_ends_a_run_before_its_budget(tmp_path: Path):
    def session() -> list[RLStepMetrics]:
        seeds = seed_everything(0, deterministic=True)
        return rl_train(
            Transformer(TINY),
            Hurry(),
            [2],
            preset("rloo"),
            RLConfig.model_validate(SETTINGS | {"flops": 1e15, "max_steps": 3}),
            np.random.default_rng(seeds.data),
            torch.Generator().manual_seed(seeds.sampling),
            tmp_path / "ckpt",
            tmp_path,
        )

    metrics = session()
    assert [m.step for m in metrics] == [1, 2, 3]
    assert metrics[-1].flops < 1e15  # most of the budget is left
    assert (tmp_path / "ckpt" / "step-000000003.json").exists()  # the last step is saved
    assert session() == []  # and the run counts as finished
