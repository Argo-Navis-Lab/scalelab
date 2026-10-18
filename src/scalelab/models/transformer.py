"""The Track S transformer: token ids in, a score for every possible next token out."""

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scalelab.core.config import ModelConfig
from scalelab.models.attention.base import Attention
from scalelab.models.attention.mha import MultiHeadAttention
from scalelab.models.init import init_weights
from scalelab.models.kvcache import KVCache, LayerCache
from scalelab.models.norms import RMSNorm
from scalelab.models.rope import RotaryEmbedding
from scalelab.tasks.tokenizer import VOCAB_SIZE


class SwiGLU(nn.Module):
    """The MLP: widen each token's vector to d_ff, gate it, and narrow it back to d_model.

    Two parallel projections up: one passes through SiLU (a smooth on/off switch) and
    multiplies the other, number by number. Then one projection back down.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """One layer: attention, then the MLP. Each reads a normalized copy of the running vector
    (pre-norm) and adds its result back to it (a residual connection)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = make_attention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: Tensor, rope: RotaryEmbedding, positions: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), rope, positions)
        return x + self.mlp(self.mlp_norm(x))

    def decode(self, x: Tensor, rope: RotaryEmbedding, cache: LayerCache, start: int) -> Tensor:
        """Like forward, for new tokens at positions start, start + 1, ..., using the cache."""
        x = x + self.attn.decode(self.attn_norm(x), rope, cache, start)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    """Embedding, n_layers blocks, a final RMSNorm, and the output head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(VOCAB_SIZE, config.d_model)  # token id -> vector
        self.rope = RotaryEmbedding(config.head_dim, config.context)  # shared by every block
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, VOCAB_SIZE, bias=False)  # vector -> token scores
        init_weights(self, config.n_layers)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: (batch, seq) token ids. Returns (batch, seq, VOCAB_SIZE) logits: at every
        position, a score for each token that could come next (higher is likelier)."""
        seq = tokens.shape[1]
        if seq > self.config.context:
            raise ValueError(f"{seq} tokens is longer than the context of {self.config.context}")
        positions = torch.arange(seq, device=tokens.device)
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, self.rope, positions)
        return self.head(self.norm(x))

    def new_cache(self, batch: int) -> KVCache:
        """An empty KV cache for `batch` sequences of up to `context` tokens, in the model's
        own dtype and on its device."""
        weight = self.embed.weight
        layers = [
            block.attn.new_cache(batch, self.config.context, weight.dtype, weight.device)
            for block in self._typed_blocks()
        ]
        return KVCache(layers, self.config.context)

    def decode(self, tokens: Tensor, cache: KVCache) -> Tensor:
        """Logits for new tokens that follow the ones already in the cache: the first call
        reads a prompt, each later call the token just generated. The logits equal what
        forward() gives for the whole sequence, without recomputing earlier positions."""
        start, n = cache.length, tokens.shape[1]
        if start + n > cache.max_len:
            raise ValueError(f"the cache holds {cache.max_len} tokens; {start + n} don't fit")
        x = self.embed(tokens)
        for block, layer in zip(self._typed_blocks(), cache.layers, strict=True):
            x = block.decode(x, self.rope, layer, start)
        cache.length = start + n
        return self.head(self.norm(x))

    def num_parameters(self, embeddings: bool = False) -> int:
        """N, the size the scaling fits use: every parameter except the embedding and the
        head, which are tiny here (64 rows each). embeddings=True counts everything."""
        total = sum(p.numel() for p in self.parameters())
        if embeddings:
            return total
        return total - self.embed.weight.numel() - self.head.weight.numel()

    def _typed_blocks(self) -> list[Block]:
        """The blocks, as Blocks: an nn.ModuleList only tells pyright it holds some Module.
        cast() changes nothing when the code runs; it just states what we know."""
        return [cast(Block, block) for block in self.blocks]


def make_attention(config: ModelConfig) -> Attention:
    """The attention variant the config names. Only MHA exists so far."""
    if config.attn == "mha":
        return MultiHeadAttention(config.d_model, config.n_heads, qk_norm=config.qk_norm)
    raise NotImplementedError(f"attention variant {config.attn!r} arrives in milestone M6")
