"""Seeds and determinism: the same seed must give the same run."""

import os
import random

import numpy as np
import torch

from scalelab.core.config import StrictModel

# Each stream gets its own child seed, numbered by position.
# Never renumber or reorder these: that would change every run's random numbers.
# A new stream may only be added at the end.
STREAMS = ("init", "data", "sampling")

MAX_SEED = 2**32 - 1  # the largest seed every library here accepts (NumPy's global limit)


class Seeds(StrictModel):
    """The run's seed and the independent child seed derived from it for each stream."""

    root: int
    init: int  # model weight initialization
    data: int  # which training problems get sampled, and in what order
    sampling: int  # token sampling during rollouts


def derive_seeds(seed: int) -> Seeds:
    """Split one seed into independent child seeds, one per stream."""
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}, got {seed}")
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    # 63 bits: big enough to never collide in practice, small enough for any database integer.
    values = [int(child.generate_state(1, dtype=np.uint64)[0] >> 1) for child in children]
    return Seeds(root=seed, **dict(zip(STREAMS, values, strict=False)))


def default_deterministic() -> bool:
    """Deterministic on CPU (your Mac, CI); off on GPU, where it costs speed."""
    return not torch.cuda.is_available()


def set_deterministic(enabled: bool) -> None:
    """Make PyTorch use only reproducible algorithms, or allow the faster nondeterministic ones."""
    if enabled:
        # cuBLAS (NVIDIA's matrix-multiply library) needs this to be reproducible.
        # It only takes effect if set before the first GPU matrix multiply, so set it early.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = False  # benchmarking picks kernels by timing, which varies


def seed_everything(seed: int, deterministic: bool | None = None) -> Seeds:
    """Seed Python, NumPy and PyTorch for one run, and return the per-stream child seeds.

    Our own code draws from generators built from the child seeds. The global generators
    are seeded too, as a safety net for library code that quietly uses them.
    """
    seeds = derive_seeds(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seeds.init)  # PyTorch initializes model weights from its global generator
    set_deterministic(default_deterministic() if deterministic is None else deterministic)
    return seeds


if __name__ == "__main__":
    print(derive_seeds(0).model_dump_json(indent=2))
