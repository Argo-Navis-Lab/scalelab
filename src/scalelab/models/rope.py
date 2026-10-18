"""Rotary position embeddings (RoPE): tell attention where each token is, by rotating vectors."""

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Rotates query and key vectors by angles that grow with the token's position.

    The head_dim numbers are used as head_dim/2 pairs: number i goes with number
    i + head_dim/2. At position p, pair i turns by the angle p * base**(-2i/head_dim): the
    first pair by 1 radian per position, each later pair more slowly. Queries and keys both
    turn, so their dot product (the attention score) depends only on how far apart the two
    tokens are, not on where they are.
    """

    def __init__(self, head_dim: int, max_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        speeds = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        angles = torch.outer(torch.arange(max_len, dtype=torch.float64), speeds)
        angles = torch.cat([angles, angles], dim=-1)  # pair i's angle in columns i and i + half
        # One row per position. Buffers move to the GPU with the model, but aren't saved in
        # checkpoints (persistent=False): they never change, and are cheap to rebuild.
        self.cos: Tensor
        self.sin: Tensor
        self.register_buffer("cos", angles.cos().float(), persistent=False)
        self.register_buffer("sin", angles.sin().float(), persistent=False)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Rotate x, shaped (batch, heads, seq, head_dim). `positions` holds each token's
        position: shape (seq,) for the whole batch, or (batch, seq) for each sequence."""
        cos = self.cos[positions].unsqueeze(-3).to(x.dtype)  # every head turns alike
        sin = self.sin[positions].unsqueeze(-3).to(x.dtype)
        return x * cos + _rotate_half(x) * sin


def _rotate_half(x: Tensor) -> Tensor:
    """Turn every pair (a, b) into (-b, a), a quarter turn; x * cos + this * sin rotates x."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)
