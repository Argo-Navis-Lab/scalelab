"""RMSNorm: rescale each vector to a root mean square of 1, then by a learned weight."""

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension.

    For a vector x of length `dim`, it returns x / sqrt(mean(x**2) + eps) * weight. The
    weight starts at all ones and is learned. It works on any leading dimensions: a single
    vector, a sequence of them, or a batch of sequences.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps  # keeps an all-zero vector from dividing by zero
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()  # work in float32: in float16, squaring anything above 255 overflows
        rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 / rms * self.weight).to(x.dtype)
