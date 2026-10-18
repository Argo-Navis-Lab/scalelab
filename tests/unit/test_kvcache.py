"""KV cache tests: decoding with the cache gives the full pass's logits, at the promised size."""

import pytest
import torch
from torch import Tensor

from scalelab.core.config import ModelConfig
from scalelab.models.attention.mha import MultiHeadAttention
from scalelab.models.rope import RotaryEmbedding
from scalelab.models.transformer import Transformer
from scalelab.tasks.tokenizer import VOCAB_SIZE

M1 = ModelConfig(name="m1", n_layers=4, d_model=128, n_heads=2)
M1_QK_NORM = ModelConfig(name="m1", n_layers=4, d_model=128, n_heads=2, qk_norm=True)


def make(config: ModelConfig = M1) -> Transformer:
    """The same starting weights every time, without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(config)


def random_tokens(batch: int, seq: int, seed: int = 0) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (batch, seq), generator=generator)


def decode_in_pieces(model: Transformer, tokens: Tensor, prompt_length: int) -> Tensor:
    """Read the first prompt_length tokens in one call, then the rest one token at a time,
    the way generation works. Returns every position's logits, in order."""
    cache = model.new_cache(tokens.shape[0])
    pieces = [model.decode(tokens[:, :prompt_length], cache)]
    for t in range(prompt_length, tokens.shape[1]):
        pieces.append(model.decode(tokens[:, t : t + 1], cache))
    return torch.cat(pieces, dim=1)


# --- the same logits as a full pass ---------------------------------------------------------


@pytest.mark.parametrize("config", [M1, M1_QK_NORM], ids=["plain", "qk-norm"])
@pytest.mark.parametrize("prompt_length", [1, 7, 20])
def test_cached_decoding_gives_the_full_pass_logits(config: ModelConfig, prompt_length: int):
    model, tokens = make(config), random_tokens(2, 20)
    # Close, not bit-identical: attention adds up the same numbers in differently shaped
    # pieces, so the last digit of a float can differ.
    torch.testing.assert_close(decode_in_pieces(model, tokens, prompt_length), model(tokens))


def test_attention_decoding_in_chunks_matches_forward():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        attn = MultiHeadAttention(128, 2)
    rope = RotaryEmbedding(64, 256)
    x = torch.randn(2, 9, 128, generator=torch.Generator().manual_seed(0))
    cache = attn.new_cache(2, 256, x.dtype, x.device)
    chunks = [attn.decode(x[:, a:b], rope, cache, a) for a, b in [(0, 4), (4, 5), (5, 9)]]
    torch.testing.assert_close(torch.cat(chunks, dim=1), attn(x, rope, torch.arange(9)))


# --- how the cache fills up -----------------------------------------------------------------


def test_a_new_cache_is_empty_and_full_size():
    cache = make().new_cache(3)
    assert cache.length == 0
    assert len(cache.layers) == M1.n_layers
    for layer in cache.layers:
        assert layer.k.shape == layer.v.shape == (3, 2, 256, 64)  # batch, heads, context, head_dim
        assert not layer.k.any() and not layer.v.any()


def test_the_cache_counts_the_positions_it_holds():
    model, tokens = make(), random_tokens(1, 6)
    cache = model.new_cache(1)
    model.decode(tokens[:, :5], cache)
    assert cache.length == 5
    model.decode(tokens[:, 5:6], cache)
    assert cache.length == 6


def test_earlier_positions_are_never_rewritten():
    model, tokens = make(), random_tokens(2, 8)
    cache = model.new_cache(2)
    model.decode(tokens[:, :5], cache)
    before = [(layer.k[:, :, :5].clone(), layer.v[:, :, :5].clone()) for layer in cache.layers]
    model.decode(tokens[:, 5:8], cache)
    for (k, v), layer in zip(before, cache.layers, strict=True):
        assert torch.equal(layer.k[:, :, :5], k) and torch.equal(layer.v[:, :, :5], v)


def test_a_full_cache_refuses_more_tokens():
    model = make()
    cache = model.new_cache(1)
    model.decode(random_tokens(1, 256), cache)  # exactly the context
    with pytest.raises(ValueError, match="the cache holds 256 tokens; 257 don't fit"):
        model.decode(random_tokens(1, 1), cache)


# --- the size the cost model will predict ---------------------------------------------------


def test_each_position_holds_what_cache_spec_says():
    spec = MultiHeadAttention(128, 2).cache_spec()
    for layer in make().new_cache(1).layers:
        assert layer.k[0, :, 0].numel() + layer.v[0, :, 0].numel() == spec.elements_per_token


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["float32", "float16"])
def test_the_cache_uses_the_memory_cache_spec_predicts(dtype: torch.dtype):
    model = make().to(dtype)
    cache = model.new_cache(3)
    used = sum(layer.k.nbytes + layer.v.nbytes for layer in cache.layers)
    per_token = MultiHeadAttention(128, 2).cache_spec().bytes_per_token(dtype.itemsize)
    assert used == M1.n_layers * 3 * M1.context * per_token  # layers x batch x positions x bytes
