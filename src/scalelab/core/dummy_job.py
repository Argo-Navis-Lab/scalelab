"""A tiny stand-in training job, used until the real trainer exists (milestones M1 and M2).

It does what every real run must do: seed everything, train, checkpoint, and resume.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from scalelab.core.checkpoint import (
    capture_rng,
    latest_checkpoint,
    load_checkpoint,
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
) -> list[float]:
    """Train a small network to add up 8 numbers, resuming from a checkpoint if there is one.

    Returns the loss of every step this call ran. `stop_after=k` pretends the job is
    interrupted: it saves a checkpoint after step k and returns early.
    """
    seeds = seed_everything(seed, deterministic=True)
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

        if step == stop_after:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "data_rng": data_rng.bit_generator.state,
                "rng": capture_rng(),
            }
            save_checkpoint(checkpoint_dir, step, state, home)
            break
    return losses


if __name__ == "__main__":
    # Demo: python -m scalelab.core.dummy_job <folder> [stop_after]
    home = Path(sys.argv[1])
    stop = int(sys.argv[2]) if len(sys.argv) > 2 else None
    losses = run_dummy_job(0, 50, home / "checkpoints", home=home, stop_after=stop)
    print(f"ran {len(losses)} steps; last loss {losses[-1]!r}")
