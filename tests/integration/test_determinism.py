"""Two CPU runs with the same seed must match exactly, step by step."""

import json
import subprocess
import sys

import numpy as np
import torch

from scalelab.core.determinism import seed_everything


def tiny_training_run(seed: int, steps: int = 50) -> list[float]:
    """A stand-in for the real trainer: a small network learning to add 8 numbers."""
    seeds = seed_everything(seed, deterministic=True)
    data_rng = np.random.default_rng(seeds.data)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(steps):
        x = torch.from_numpy(data_rng.standard_normal((16, 8), dtype=np.float32))
        y = x.sum(dim=1, keepdim=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_same_seed_gives_bit_identical_losses():
    first = tiny_training_run(seed=0)
    assert tiny_training_run(seed=0) == first  # exactly equal, not just close
    assert first[-1] < first[0]  # and it really trained


def test_different_seed_gives_different_losses():
    assert tiny_training_run(seed=0) != tiny_training_run(seed=1)


def test_same_seed_matches_across_separate_processes():
    # Two fresh Python processes, like two separate runs of the lab.
    def run_in_new_process() -> list[float]:
        result = subprocess.run(
            [sys.executable, __file__, "0"], capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)

    assert run_in_new_process() == run_in_new_process() == tiny_training_run(seed=0)


if __name__ == "__main__":
    print(json.dumps(tiny_training_run(seed=int(sys.argv[1]))))
