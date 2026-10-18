"""The KV cache: every layer's keys and values so far, so each new token reuses them."""

import torch
from torch import Tensor


class LayerCache:
    """One attention layer's keys and values, in tensors allocated once at full size,
    (batch, heads, max_len, head_dim), and filled in position by position."""

    def __init__(
        self,
        batch: int,
        heads: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.k = torch.zeros(batch, heads, max_len, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)

    def store(self, k: Tensor, v: Tensor, start: int) -> tuple[Tensor, Tensor]:
        """Write the keys and values of new positions start, start + 1, ..., then return the
        keys and values of every position so far."""
        end = start + k.shape[2]
        self.k[:, :, start:end] = k
        self.v[:, :, start:end] = v
        return self.k[:, :, :end], self.v[:, :, :end]


class KVCache:
    """Every layer's cache, and how many positions they hold so far."""

    def __init__(self, layers: list[LayerCache], max_len: int) -> None:
        self.layers = layers
        self.max_len = max_len
        self.length = 0  # positions 0 to length - 1 are filled
