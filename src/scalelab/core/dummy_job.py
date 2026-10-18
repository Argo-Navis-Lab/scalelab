"""A tiny stand-in training job, used until the real trainer exists (milestones M1 and M2).

It does what every real run must do: seed everything, train, checkpoint, and resume.
"""

import argparse
import time
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
from scalelab.core.determinism import seed_everything


def run_dummy_job(
    seed: int,
    steps: int,
    checkpoint_dir: str | Path,
    home: Path | None = None,
    stop_after: int | None = None,
    policy: CheckpointPolicy | None = None,
    step_seconds: float = 0.0,
    deterministic: bool = True,
) -> list[float]:
    """Train a small network to add up 8 numbers, resuming from a checkpoint if there is one.

    Returns the loss of every step this call ran. Saving follows `policy` (by default: every
    15 minutes, keeping 2). `stop_after=k` pretends the job is interrupted after step k: it
    saves and returns early, just as a SIGTERM would. `step_seconds` makes each step slower,
    so demos and tests have time to interrupt it.
    """
    policy = policy or CheckpointPolicy()
    seeds = seed_everything(seed, deterministic)
    data_rng = np.random.default_rng(seeds.data)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    start = 0
    latest = latest_checkpoint(checkpoint_dir)
    if latest is not None:
        start, state = load_checkpoint(latest, home)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        data_rng.bit_generator.state = state["data_rng"]
        restore_rng(state["rng"])

    losses = []
    for step in range(start + 1, steps + 1):  # steps are counted from 1
        x = torch.from_numpy(data_rng.standard_normal((16, 8), dtype=np.float32))
        y = x.sum(dim=1, keepdim=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        time.sleep(step_seconds)

        if step == stop_after:
            policy.request_stop()
        if policy.due(step):
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "data_rng": data_rng.bit_generator.state,
                "rng": capture_rng(),
            }
            save_checkpoint(checkpoint_dir, step, state, home)
            prune_checkpoints(checkpoint_dir, policy.keep, home)
            policy.saved()
            if policy.stop_requested:
                break
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dummy training job (seed 0).")
    parser.add_argument("home", type=Path, help="folder for its checkpoints and artifacts")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--every-steps", type=int, default=None, help="also save every N steps")
    parser.add_argument("--step-seconds", type=float, default=0.0, help="slow each step down")
    args = parser.parse_args()

    policy = CheckpointPolicy(every_steps=args.every_steps)
    policy.stop_on_signals()
    losses = run_dummy_job(
        0,
        args.steps,
        args.home / "checkpoints",
        home=args.home,
        policy=policy,
        step_seconds=args.step_seconds,
    )
    how = "stopped early, checkpoint saved" if policy.stop_requested else "finished"
    last = losses[-1] if losses else None
    print(f"ran {len(losses)} steps ({how}); last loss {last!r}")
