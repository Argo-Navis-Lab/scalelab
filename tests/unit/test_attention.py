"""Attention tests: the maths by hand, no peeking ahead, QK-norm, parameters and cache size,
and (m2s31) the scores telemetry reads."""

import math

import pytest
import torch
from torch import Tensor

from scalelab.models.attention.base import Attention, CacheSpec
from scalelab.models.attention.mha import MultiHeadAttention
from scalelab.models.rope import RotaryEmbedding

D_MODEL, N_HEADS, SEQ = 128, 2, 10  # m1's width and head count, so head_dim is 64
ROPE = RotaryEmbedding(64, 256)
POSITIONS = torch.arange(SEQ)


def randn(*shape: int, seed: int = 0) -> Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def make(qk_norm: bool = False) -> MultiHeadAttention:
    """The same starting weights every time, without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return MultiHeadAttention(D_MODEL, N_HEADS, qk_norm=qk_norm)


def attention_by_hand(attn: MultiHeadAttention, x: Tensor) -> Tensor:
    """The same attention written out step by step with plain matrix maths, as a check."""
    batch, seq, d_model = x.shape

    def heads(t: Tensor) -> Tensor:  # (batch, seq, d_model) -> (batch, heads, seq, head_dim)
        return t.view(batch, seq, attn.n_heads, attn.head_dim).transpose(1, 2)

    positions = torch.arange(seq)
    q = ROPE(attn.q_norm(heads(attn.q(x))), positions)
    k = ROPE(attn.k_norm(heads(attn.k(x))), positions)
    v = heads(attn.v(x))
    scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)  # every query with every key
    future = torch.ones(seq, seq, dtype=torch.bool).triu(diagonal=1)  # True above the diagonal
    weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)  # rows add up to 1
    mixed = weights @ v  # each token's weighted mix of the values it may see
    return attn.o(mixed.transpose(1, 2).reshape(batch, seq, d_model))


# --- the maths -----------------------------------------------------------------------------


def test_output_has_the_input_shape():
    assert make()(randn(2, SEQ, D_MODEL), ROPE, POSITIONS).shape == (2, SEQ, D_MODEL)


@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qk-norm"])
def test_matches_attention_written_out_by_hand(qk_norm: bool):
    attn, x = make(qk_norm), randn(2, SEQ, D_MODEL)
    torch.testing.assert_close(attn(x, ROPE, POSITIONS), attention_by_hand(attn, x))


def test_the_first_token_can_only_attend_to_itself():
    attn, x = make(), randn(2, SEQ, D_MODEL)
    out = attn(x, ROPE, POSITIONS)
    torch.testing.assert_close(out[:, 0], attn.o(attn.v(x[:, 0])))  # all weight on its own value


# --- no peeking ahead ----------------------------------------------------------------------


def test_a_token_never_sees_the_tokens_after_it():
    attn, x = make(), randn(2, SEQ, D_MODEL)
    changed = x.clone()
    changed[:, 6:] = randn(2, SEQ - 6, D_MODEL, seed=1)  # rewrite tokens 6 to 9
    before, after = attn(x, ROPE, POSITIONS), attn(changed, ROPE, POSITIONS)
    assert torch.equal(before[:, :6], after[:, :6])  # tokens 0 to 5: exactly the same
    assert not torch.allclose(before[:, 6:], after[:, 6:])


# --- QK-norm -------------------------------------------------------------------------------


def test_qk_norm_keeps_the_attention_pattern_when_inputs_grow():
    # With QK-norm, queries and keys are normalized, so scores can't grow with the input:
    # 1000 times the input just gives 1000 times the output (the values grow, nothing else).
    # Dividing by 1000 compares at the usual scale, where float32's rounding is tiny.
    attn, x = make(qk_norm=True), randn(2, SEQ, D_MODEL)
    torch.testing.assert_close(attn(1000 * x, ROPE, POSITIONS) / 1000, attn(x, ROPE, POSITIONS))


def test_without_qk_norm_scores_grow_and_the_pattern_changes():
    attn, x = make(qk_norm=False), randn(2, SEQ, D_MODEL)
    big = attn(1000 * x, ROPE, POSITIONS) / 1000
    assert not torch.allclose(big, attn(x, ROPE, POSITIONS), atol=1e-2)


# --- parameters, the cache and the interface -----------------------------------------------


def test_parameters_without_qk_norm():
    names = {name: p.numel() for name, p in make().named_parameters()}
    assert names == {
        "q.weight": 128 * 128,
        "k.weight": 128 * 128,
        "v.weight": 128 * 128,
        "o.weight": 128 * 128,
    }


def test_qk_norm_adds_one_weight_per_head_dimension_for_queries_and_keys():
    names = {name: p.numel() for name, p in make(qk_norm=True).named_parameters()}
    assert names["q_norm.weight"] == names["k_norm.weight"] == 64
    assert sum(names.values()) == 4 * 128 * 128 + 2 * 64


def test_gradients_reach_every_weight():
    attn = make(qk_norm=True)
    attn(randn(2, SEQ, D_MODEL), ROPE, POSITIONS).sum().backward()
    assert all(p.grad is not None for p in attn.parameters())


def test_the_cache_keeps_a_key_and_a_value_per_token():
    spec = make().cache_spec()
    assert spec == CacheSpec(elements_per_token=2 * D_MODEL)
    assert spec.bytes_per_token(2) == 512  # float16: 256 numbers of 2 bytes each


def test_heads_must_divide_the_width():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(128, 3)


def test_attention_itself_is_only_an_interface():
    assert isinstance(make(), Attention)
    with pytest.raises(TypeError, match="abstract"):
        Attention()  # pyright: ignore[reportAbstractUsage]


# --- added in m2s31: the scores telemetry reads ---------------------------------------------


@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qk-norm"])
def test_the_scores_are_the_ones_forward_weighs_the_values_by(qk_norm: bool):
    attn, x = make(qk_norm), randn(2, SEQ, D_MODEL)
    scores = attn.scores(x, ROPE, POSITIONS)
    assert scores.shape == (2, N_HEADS, SEQ, SEQ)
    future = torch.ones(SEQ, SEQ, dtype=torch.bool).triu(diagonal=1)
    assert bool(scores[..., future].isneginf().all())  # no peeking ahead
    assert bool(scores[..., ~future].isfinite().all())
    v = attn.v(x).view(2, SEQ, N_HEADS, attn.head_dim).transpose(1, 2)
    mixed = scores.softmax(dim=-1) @ v  # (batch, heads, seq, head_dim)
    joined = mixed.transpose(1, 2).reshape(2, SEQ, D_MODEL)
    torch.testing.assert_close(attn.o(joined), attn(x, ROPE, POSITIONS))
