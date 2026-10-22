"""Multi-head attention (MHA): the reference variant, with one key and value per head."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scalelab.models.attention.base import Attention, CacheSpec
from scalelab.models.kvcache import LayerCache
from scalelab.models.norms import RMSNorm
from scalelab.models.rope import RotaryEmbedding


class MultiHeadAttention(Attention):
    """n_heads heads, each working on head_dim = d_model / n_heads of the numbers.

    Every head projects each token to a query, a key and a value, scores every earlier
    token by query-key dot products, and returns a weighted mix of their values. The heads'
    results are joined and projected back to d_model.
    """

    def __init__(self, d_model: int, n_heads: int, qk_norm: bool = False) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        # QK-norm normalizes each head's queries and keys, which caps how large attention
        # scores can grow. Off by default; nn.Identity does nothing and has no parameters.
        self.q_norm: nn.Module = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm: nn.Module = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: Tensor, rope: RotaryEmbedding, positions: Tensor) -> Tensor:
        q, k, v = self._project(x, rope, positions)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(self._join_heads(out))

    def scores(self, x: Tensor, rope: RotaryEmbedding, positions: Tensor) -> Tensor:
        q, k, _ = self._project(x, rope, positions)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # SDPA's default scale
        future = positions[None, :] > positions[:, None]  # True where the key comes later
        return logits.masked_fill(future, float("-inf"))

    def decode(self, x: Tensor, rope: RotaryEmbedding, cache: LayerCache, start: int) -> Tensor:
        positions = torch.arange(start, start + x.shape[1], device=x.device)
        q, k, v = self._project(x, rope, positions)
        k, v = cache.store(k, v, start)  # now the keys and values of every position so far
        # New token i sits at position start + i, and may see every position up to that one.
        allowed = torch.arange(start + x.shape[1], device=x.device) <= positions[:, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.o(self._join_heads(out))

    def new_cache(
        self, batch: int, max_len: int, dtype: torch.dtype, device: torch.device
    ) -> LayerCache:
        return LayerCache(batch, self.n_heads, max_len, self.head_dim, dtype, device)

    def cache_spec(self) -> CacheSpec:
        return CacheSpec(elements_per_token=2 * self.n_heads * self.head_dim)  # a key and a value

    def _project(
        self, x: Tensor, rope: RotaryEmbedding, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Queries, keys and values for x, split into heads, with QK-norm and RoPE applied."""
        q = rope(self.q_norm(self._split_heads(self.q(x))), positions)
        k = rope(self.k_norm(self._split_heads(self.k(x))), positions)
        v = self._split_heads(self.v(x))
        return q, k, v

    def _split_heads(self, x: Tensor) -> Tensor:
        """(batch, seq, d_model) -> (batch, heads, seq, head_dim)."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

    def _join_heads(self, x: Tensor) -> Tensor:
        """(batch, heads, seq, head_dim) -> (batch, seq, d_model)."""
        batch, _, seq, _ = x.shape
        return x.transpose(1, 2).reshape(batch, seq, self.n_heads * self.head_dim)
