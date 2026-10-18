"""What every attention variant offers (MHA now; GQA, MLA and sliding-window later)."""

from abc import ABC, abstractmethod

from torch import Tensor, nn

from scalelab.core.config import StrictModel
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
    def cache_spec(self) -> CacheSpec:
        """What this layer would keep per token in a KV cache (used from step 13 on)."""
