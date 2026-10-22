"""What every attention variant offers (MHA now; GQA, MLA and sliding-window later)."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from scalelab.core.config import StrictModel
from scalelab.models.kvcache import LayerCache
from scalelab.models.rope import RotaryEmbedding


class CacheSpec(StrictModel):
    """What one layer's KV cache keeps for every token. The cost model reads this from the
    module itself, so its memory predictions can't drift away from the code."""

    elements_per_token: int  # how many numbers are stored per token (keys and values for MHA)

    def bytes_per_token(self, bytes_per_element: int) -> int:
        """Bytes per token in this layer: 2 per number in float16, 4 in float32."""
        return self.elements_per_token * bytes_per_element


class Attention(nn.Module, ABC):
    """The interface every attention variant follows, so the transformer and the cost model
    can swap one for another without changing."""

    @abstractmethod
    def forward(self, x: Tensor, rope: RotaryEmbedding, positions: Tensor) -> Tensor:
        """Causal attention over a whole sequence (for training, and to read a prompt).

        x is (batch, seq, d_model); positions are the tokens' positions for RoPE. Each token
        sees itself and the tokens before it, never the ones after.
        """

    @abstractmethod
    def scores(self, x: Tensor, rope: RotaryEmbedding, positions: Tensor) -> Tensor:
        """The attention scores forward() works with, for telemetry (m2s31): (batch, heads,
        seq, seq), each query's pre-softmax logit for each key, and -inf for the keys after
        the query. A softmax over the last dimension turns them into the attention weights.
        forward() leaves them inside SDPA, which need never store them all at once."""

    @abstractmethod
    def decode(self, x: Tensor, rope: RotaryEmbedding, cache: LayerCache, start: int) -> Tensor:
        """Attention for new tokens at positions start, start + 1, ..., using the cache for
        every earlier position, and storing the new tokens' keys and values in it. Reading a
        whole prompt (start 0) and generating one token at a time both go through here."""

    @abstractmethod
    def new_cache(
        self, batch: int, max_len: int, dtype: torch.dtype, device: torch.device
    ) -> LayerCache:
        """An empty cache for this layer, with room for max_len positions."""

    @abstractmethod
    def cache_spec(self) -> CacheSpec:
        """What this layer keeps per token in its cache: new_cache's tensors must match it."""
