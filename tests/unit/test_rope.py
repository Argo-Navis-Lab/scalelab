"""RoPE tests: a hand example, the rotation's properties, and a second way to compute it."""

import math

import pytest
import torch

from scalelab.models.rope import RotaryEmbedding

HEAD_DIM, MAX_LEN = 64, 256  # the size of every Track S head, and the context length


def randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def complex_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10_000.0) -> torch.Tensor:
    """The same rotation computed another way, as a check: pair i is the complex number
    x[i] + 1j * x[i + half], and turning it by angle a means multiplying by e^(1j * a)."""
    half = x.shape[-1] // 2
    z = torch.complex(x[..., :half].double(), x[..., half:].double())
    speeds = base ** (-2 * torch.arange(half, dtype=torch.float64) / x.shape[-1])
    angles = positions.double()[:, None] * speeds  # (seq, half)
    turned = z * torch.polar(torch.ones_like(angles), angles)  # e^(1j * angle)
    return torch.cat([turned.real, turned.imag], dim=-1).float()


# --- by hand -------------------------------------------------------------------------------


def test_a_hand_worked_example():
    # head_dim 2 is a single pair, turning by 1 radian per position: (1, 0) moves round
    # the unit circle to (cos p, sin p).
    rope = RotaryEmbedding(2, MAX_LEN)
    x = torch.tensor([1.0, 0.0]).repeat(1, 1, 3, 1)  # the same vector at positions 0, 1, 2
    expected = torch.tensor([[1.0, 0.0], [math.cos(1), math.sin(1)], [math.cos(2), math.sin(2)]])
    torch.testing.assert_close(rope(x, torch.arange(3)), expected.reshape(1, 1, 3, 2))


def test_each_pair_turns_at_its_own_speed():
    # head_dim 4: pair 0 turns 1 radian per position, pair 1 turns 10000**(-2/4) = 0.01.
    rope = RotaryEmbedding(4, MAX_LEN)
    angles = torch.tensor([1.0, 0.01, 1.0, 0.01])
    torch.testing.assert_close(rope.cos[1], angles.cos())
    torch.testing.assert_close(rope.sin[1], angles.sin())


# --- properties of a rotation --------------------------------------------------------------


def test_position_zero_changes_nothing():
    x = randn(2, 3, 1, HEAD_DIM)
    torch.testing.assert_close(RotaryEmbedding(HEAD_DIM, MAX_LEN)(x, torch.tensor([0])), x)


def test_turning_never_changes_a_vectors_length():
    x = randn(2, 3, MAX_LEN, HEAD_DIM)
    out = RotaryEmbedding(HEAD_DIM, MAX_LEN)(x, torch.arange(MAX_LEN))
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))


def test_scores_depend_only_on_how_far_apart_two_tokens_are():
    rope = RotaryEmbedding(HEAD_DIM, MAX_LEN)
    q, k = randn(1, 1, 1, HEAD_DIM, seed=1), randn(1, 1, 1, HEAD_DIM, seed=2)

    def score(query_position: int, key_position: int) -> torch.Tensor:
        turned_q = rope(q, torch.tensor([query_position]))
        turned_k = rope(k, torch.tensor([key_position]))
        return (turned_q * turned_k).sum()

    torch.testing.assert_close(score(10, 3), score(17, 10))  # 7 apart, at different places
    torch.testing.assert_close(score(10, 3), score(200, 193))
    assert not torch.allclose(score(10, 3), score(10, 4))  # 6 apart is a different score


# --- a second way to compute it ------------------------------------------------------------


def test_matches_rotating_complex_numbers():
    x = randn(2, 3, MAX_LEN, HEAD_DIM)
    positions = torch.arange(MAX_LEN)
    out = RotaryEmbedding(HEAD_DIM, MAX_LEN)(x, positions)
    torch.testing.assert_close(out, complex_rope(x, positions))


# --- positions -----------------------------------------------------------------------------


def test_each_sequence_can_have_its_own_positions():
    rope = RotaryEmbedding(HEAD_DIM, MAX_LEN)
    x = randn(2, 3, 4, HEAD_DIM)
    positions = torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]])  # the second sequence starts at 5
    out = rope(x, positions)
    torch.testing.assert_close(out[0], rope(x[:1], positions[0])[0])
    torch.testing.assert_close(out[1], rope(x[1:], positions[1])[0])


def test_positions_beyond_the_context_are_refused():
    rope = RotaryEmbedding(HEAD_DIM, MAX_LEN)
    with pytest.raises(IndexError):
        rope(randn(1, 1, 1, HEAD_DIM), torch.tensor([MAX_LEN]))


# --- types, parameters and mistakes --------------------------------------------------------


def test_float16_in_float16_out():
    x = randn(1, 2, 8, HEAD_DIM)
    rope = RotaryEmbedding(HEAD_DIM, MAX_LEN)
    out = rope(x.half(), torch.arange(8))
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), rope(x, torch.arange(8)), atol=1e-2, rtol=1e-2)


def test_nothing_is_learned_and_nothing_is_saved():
    rope = RotaryEmbedding(HEAD_DIM, MAX_LEN)
    assert list(rope.parameters()) == []
    assert dict(rope.state_dict()) == {}  # the tables are rebuilt, never stored
    assert rope.cos.shape == rope.sin.shape == (MAX_LEN, HEAD_DIM)


def test_head_dim_must_be_even():
    with pytest.raises(ValueError, match="must be even"):
        RotaryEmbedding(63, MAX_LEN)
