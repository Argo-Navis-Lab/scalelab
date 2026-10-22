"""The RL loop: draw problems, sample answers, score them and learn from them, step after
step, until the FLOP budget is spent."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from scalelab.core.checkpoint import (
    CheckpointPolicy,
    capture_rng,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    restore_rng,
    save_checkpoint,
)
from scalelab.core.config import AlgoConfig, RLConfig, StrictModel
from scalelab.costmodel.specs import ModelSpec
from scalelab.models.transformer import Transformer
from scalelab.rl.advantages import advantages
from scalelab.rl.filters import drop_flat_groups, loss_mask, overlong_penalty
from scalelab.rl.flops import rl_step_flops
from scalelab.rl.rollout import generate
from scalelab.rl.scoring import reward_values, score_rollouts
from scalelab.rl.trainer import UpdateMetrics, rl_update
from scalelab.rl.warmstart_loop import make_optimizer
from scalelab.tasks.base import Task
from scalelab.telemetry.hooks import CheckpointEvent, Hook, RolloutEvent, StepEndEvent

GENERATION_BATCH = 512  # answers decoded at once: a whole step of 64 problems x 8 answers


class RLStepMetrics(StrictModel):
    """What one RL step did. flops is a running total from the start."""

    step: int
    flops: int  # sampling, log-probs and training so far, as the cost model counts them
    reward: float  # the mean reward of the step's answers
    trained: int  # answers left to train on after the filters
    loss: float  # the step's last update's; this and the next three are 0 with no update
    grad_norm: float  # the gradient's size before clipping
    clip_fraction: float
    kl: float


def budget_spent(config: RLConfig, step: int, flops: float) -> bool:
    """Whether a run that has taken `step` steps and spent `flops` FLOPs is finished: its FLOP
    budget is spent, or it has taken config.max_steps steps."""
    return flops >= config.flops or (config.max_steps is not None and step >= config.max_steps)


def sample_problems[P](
    task: Task[P],
    rng: np.random.Generator,
    n: int,
    difficulties: Sequence[int],
    excluded: frozenset[tuple[int, ...]] = frozenset(),
) -> list[P]:
    """n problems, each at a difficulty drawn uniformly from `difficulties`, drawn as warm
    start draws them. A problem whose prompt is in `excluded` (the eval sets' prompts) is
    skipped and another is drawn."""
    problems: list[P] = []
    while len(problems) < n:
        difficulty = difficulties[int(rng.integers(len(difficulties)))]
        (problem,) = task.sample(rng, 1, difficulty)
        if tuple(task.format_prompt(problem)) not in excluded:
            problems.append(problem)
    return problems


def rl_train[P](
    model: Transformer,
    task: Task[P],
    difficulties: Sequence[int],
    algo: AlgoConfig,
    config: RLConfig,
    data_rng: np.random.Generator,
    generator: torch.Generator,
    checkpoint_dir: str | Path,
    home: Path | None = None,
    policy: CheckpointPolicy | None = None,
    reference: Transformer | None = None,
    excluded: frozenset[tuple[int, ...]] = frozenset(),
    stop_after: int | None = None,
    on_step: Callable[[RLStepMetrics], None] | None = None,
    hooks: Sequence[Hook] = (),
) -> list[RLStepMetrics]:
    """Train `model` with RL until config.flops FLOPs have been spent (or config.max_steps
    steps taken), resuming from the latest checkpoint in `checkpoint_dir` if there is one.

    Each step draws config.prompts_per_step problems with `data_rng`, samples
    config.group_size answers to each with `generator` (on the model's device), scores
    them, applies algo's filters and updates the model on what is left. `reference` is the
    model the KL penalty measures against (needed when algo.kl_coef > 0).

    Returns the metrics of every step this call ran. Saving follows `policy` (by default:
    every 15 minutes, keeping 2), and the last step is always saved. `stop_after=k`
    pretends the job is interrupted after step k. `on_step`, if given, is called with each
    step's metrics before any saving. Each of `hooks` is called once a step's answers are
    scored (on_rollout), by the updates (see trainer.rl_update), when the step is over
    (on_step_end) and before each checkpoint is saved (on_checkpoint).
    """
    policy = policy or CheckpointPolicy()
    optimizer = make_optimizer(model, config)
    spec = ModelSpec.from_config(model.config)
    context = model.config.context

    step, flops = 0, 0
    latest = latest_checkpoint(checkpoint_dir)
    if latest is not None:
        step, state = load_checkpoint(latest, home)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        data_rng.bit_generator.state = state["data_rng"]
        generator.set_state(state["sampling"])
        restore_rng(state["rng"])
        flops = state["flops"]

    metrics: list[RLStepMetrics] = []
    while not budget_spent(config, step, flops):
        step += 1  # the step being taken
        problems = sample_problems(task, data_rng, config.prompts_per_step, difficulties, excluded)
        prompts = [task.format_prompt(problem) for problem in problems]
        batch = generate(
            model,
            prompts,
            config.group_size,
            config.sampling,
            generator,
            policy_version=step - 1,  # the number of steps that have changed the weights
            batch_size=GENERATION_BATCH,
        )
        rewards = reward_values(score_rollouts(task, problems, batch), algo.format_weight)
        if algo.overlong == "soft_penalty":
            rewards = rewards + overlong_penalty(batch, context, algo.overlong_buffer)
        scored = RolloutEvent(step=step, batch=batch, rewards=rewards)
        for hook in hooks:
            hook.on_rollout(scored)
        trained, kept = batch, rewards
        if algo.dynamic_sampling:
            trained, kept = drop_flat_groups(batch, rewards)

        updates: list[UpdateMetrics] = []
        if trained.finish_reasons:  # dynamic sampling can leave nothing to learn from
            updates = rl_update(
                model,
                optimizer,
                trained,
                advantages(kept, trained.group_ids, algo.estimator),
                loss_mask(trained, algo.overlong),
                algo,
                temperature=config.sampling.temperature,
                epochs=config.epochs,
                micro_batch=config.micro_batch,
                grad_clip=config.grad_clip,
                reference=reference,
                hooks=hooks,
                step=step,
            )
        flops += rl_step_flops(
            spec,
            batch,
            trained if updates else None,  # no update: only the sampling was done
            batch_size=GENERATION_BATCH,
            micro_batch=config.micro_batch,
            epochs=config.epochs,
            reference=algo.kl_coef > 0,
        ).total
        last = updates[-1] if updates else None
        metrics.append(
            RLStepMetrics(
                step=step,
                flops=flops,
                reward=rewards.mean().item(),
                trained=len(trained.finish_reasons),
                loss=last.loss if last else 0.0,
                grad_norm=last.grad_norm if last else 0.0,
                clip_fraction=last.clip_fraction if last else 0.0,
                kl=last.kl if last else 0.0,
            )
        )
        ended = StepEndEvent(step=step)
        for hook in hooks:
            hook.on_step_end(ended)
        if on_step is not None:
            on_step(metrics[-1])
        if step == stop_after:
            policy.request_stop()
        if budget_spent(config, step, flops) or policy.due(step):
            saving = CheckpointEvent(step=step)
            for hook in hooks:
                hook.on_checkpoint(saving)  # before saving: see CheckpointEvent
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "data_rng": data_rng.bit_generator.state,
                "sampling": generator.get_state(),
                "rng": capture_rng(),
                "flops": flops,
            }
            save_checkpoint(checkpoint_dir, step, state, home)
            prune_checkpoints(checkpoint_dir, policy.keep, home)
            policy.saved()
            if policy.stop_requested:
                break
    return metrics
