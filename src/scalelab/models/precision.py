"""The number formats a model runs in: float32 throughout, or float16 under autocast."""

from collections.abc import Generator
from contextlib import contextmanager

import torch

from scalelab.costmodel.specs import Precision

DTYPES: dict[Precision, torch.dtype] = {"fp32": torch.float32, "fp16": torch.float16}


@contextmanager
def autocast(precision: Precision, device: torch.device) -> Generator[None]:
    """A region where the models run in `precision`. fp16 is autocast's: the weights stay
    float32, and each is cast to float16 once a region (autocast keeps the casts of tensors
    that require grad until the region ends). fp32 changes nothing."""
    if precision == "fp32":
        yield
        return
    with torch.autocast(device.type, dtype=DTYPES[precision]):
        yield
