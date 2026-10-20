"""The warm-start training loop: worked solutions, step after step, until the budget is spent."""

import math
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
from scalelab.core.config import RLConfig, StrictModel, WarmStartConfig
from scalelab.costmodel.ops import count_flops
from scalelab.costmodel.specs import ModelSpec, WorkloadSpec
from scalelab.models.transformer import Transformer
from scalelab.rl.warmstart import make_batch, sample_examples, warm_start_loss
from scalelab.tasks.base import Task

BETAS = (0.9, 0.95)  # how fast Adam's running averages forget, as in GPT-3


class StepMetrics(StrictModel):
    """What one training step did. tokens and flops are running totals from the start."""

    step: int
    tokens: int  # positions read so far: what the budget counts
    flops: int  # training FLOPs so far, as the cost model counts them
    loss: float
    lr: float
    grad_norm: float  # the gradient's size before clipping


def learning_rate(config: WarmStartConfig, progress: float) -> float:
    """The learning rate once `progress` (0 to 1) of the budget is used: a straight ramp up
    from 0 during the warmup, then half a cosine wave down to decay_to * lr at the end."""
    if progress < config.warmup:
        return config.lr * progress / config.warmup
    cosine = 0.5 * (1 + math.cos(math.pi * (progress - config.warmup) / (1 - config.warmup)))
    return config.lr * (config.decay_to + (1 - config.decay_to) * cosine)


def make_optimizer(model: Transformer, config: WarmStartConfig | RLConfig) -> torch.optim.AdamW:
    """AdamW, for warm starts and RL alike. Weight decay pulls the matrices toward zero; the
    norms' weights, which are vectors, are left alone (GPT-2's recipe)."""
    matrices = [p for p in model.parameters() if p.dim() >= 2]
    vectors = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": matrices, "weight_decay": config.weight_decay},
        {"params": vectors, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr, betas=BETAS)


def warm_start[P](
    model: Transformer,
    task: Task[P],
    difficulties: Sequence[int],
    config: WarmStartConfig,
    data_rng: np.random.Generator,
    checkpoint_dir: str | Path,
    home: Path | None = None,
    policy: CheckpointPolicy | None = None,
    excluded: frozenset[tuple[int, ...]] = frozenset(),
    stop_after: int | None = None,
    on_step: Callable[[StepMetrics], None] | None = None,
) -> list[StepMetrics]:
    """Train `model` on worked solutions until config.tokens positions have been read,
    resuming from the latest checkpoint in `checkpoint_dir` if there is one.

    Returns the metrics of every step this call ran. Saving follows `policy` (by default:
    every 15 minutes, keeping 2), and the last step is always saved, because its weights are
    what RL will start from. `stop_after=k` pretends the job is interrupted after step k.
    `on_step`, if given, is called with each step's metrics before any saving.
    """
    policy = policy or CheckpointPolicy()
    optimizer = make_optimizer(model, config)
    spec = ModelSpec.from_config(model.config)
    device = model.embed.weight.device

    step, tokens, flops = 0, 0, 0
    latest = latest_checkpoint(checkpoint_dir)
    if latest is not None:
        step, state = load_checkpoint(latest, home)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        data_rng.bit_generator.state = state["data_rng"]
        restore_rng(state["rng"])
        tokens, flops = state["tokens"], state["flops"]

    context = model.config.context
    metrics: list[StepMetrics] = []
    while tokens < config.tokens:
        batch = make_batch(
            sample_examples(task, data_rng, config.batch_size, difficulties, context, excluded)
        )
        step += 1
        tokens += batch.tokens
        sequences, width = batch.inputs.shape  # the padding is computed too, so it's counted
        flops += count_flops(spec, WorkloadSpec(phase="train", batch=sequences, tokens=width)).total
        lr = learning_rate(config, min(1.0, tokens / config.tokens))
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss = warm_start_loss(model(batch.inputs.to(device)), batch.targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        metrics.append(
            StepMetrics(
                step=step,
                tokens=tokens,
                flops=flops,
                loss=loss.item(),
                lr=lr,
                grad_norm=grad_norm.item(),
            )
        )
        if on_step is not None:
            on_step(metrics[-1])
        if step == stop_after:
            policy.request_stop()
        if tokens >= config.tokens or policy.due(step):
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "data_rng": data_rng.bit_generator.state,
                "rng": capture_rng(),
                "tokens": tokens,
                "flops": flops,
            }
            save_checkpoint(checkpoint_dir, step, state, home)
            prune_checkpoints(checkpoint_dir, policy.keep, home)
            policy.saved()
            if policy.stop_requested:
                break
    return metrics
