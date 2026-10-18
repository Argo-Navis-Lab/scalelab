"""RMSNorm tests: the formula by hand, PyTorch's own RMSNorm, shapes, float16 and gradients."""

import pytest
import torch

from scalelab.models.norms import RMSNorm


def randn(*shape: int) -> torch.Tensor:
    """Random numbers from a fixed seed, so every run of the tests sees the same input."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(0))


def rms(x: torch.Tensor) -> torch.Tensor:
    """Root mean square over the last dimension."""
    return x.pow(2).mean(dim=-1).sqrt()


# --- the formula ---------------------------------------------------------------------------


def test_a_hand_worked_example():
    # [3, 4]: mean of squares (9 + 16) / 2 = 12.5, root mean square 3.5355,
    # so the result is [3 / 3.5355, 4 / 3.5355] = [0.8485, 1.1314].
    out = RMSNorm(2)(torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(out, torch.tensor([0.8485, 1.1314]), atol=1e-4, rtol=0)


def test_every_output_vector_has_a_root_mean_square_of_one():
    out = RMSNorm(16)(randn(4, 7, 16) * 50)
    torch.testing.assert_close(rms(out), torch.ones(4, 7))


def test_scaling_the_input_changes_nothing():
    norm, x = RMSNorm(8), randn(3, 8)
    torch.testing.assert_close(norm(10 * x), norm(x))


def test_the_weight_scales_each_feature():
    norm, x = RMSNorm(8), randn(3, 8)
    before = norm(x)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    torch.testing.assert_close(norm(x), 2 * before)


def test_an_all_zero_vector_gives_zeros_not_nan():
    assert torch.equal(RMSNorm(3)(torch.zeros(3)), torch.zeros(3))


# --- agreeing with PyTorch -----------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(8,), (3, 8), (8, 8), (2, 5, 8)], ids=["vector", "sequence", "square", "batch"]
)
def test_matches_pytorchs_own_rmsnorm(shape: tuple[int, ...]):
    ours, theirs = RMSNorm(8), torch.nn.RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        weight = randn(8)
        ours.weight.copy_(weight)
        theirs.weight.copy_(weight)
    x = randn(*shape)
    out = ours(x)
    assert out.shape == x.shape
    # Close, not bit-identical: the two do the same maths in a slightly different order.
    torch.testing.assert_close(out, theirs(x))


# --- float16 -------------------------------------------------------------------------------


def test_float16_overflows_when_squaring_300():
    x = torch.tensor([300.0], dtype=torch.float16)
    assert torch.isinf(x * x).all()  # 90000 is beyond float16's largest number, 65504


def test_float16_input_still_works_and_stays_float16():
    out = RMSNorm(4)(torch.full((4,), 300.0, dtype=torch.float16))
    assert out.dtype == torch.float16
    assert torch.equal(out, torch.ones(4, dtype=torch.float16))


# --- learning ------------------------------------------------------------------------------


def test_the_weight_is_the_only_parameter():
    norm = RMSNorm(8)
    ((name, weight),) = norm.named_parameters()
    assert name == "weight"
    assert weight.shape == (8,)
    assert torch.equal(weight, torch.ones(8))
    assert weight.requires_grad


def test_gradients_reach_the_weight():
    norm = RMSNorm(8)
    norm(randn(3, 8)).sum().backward()
    assert norm.weight.grad is not None
    assert norm.weight.grad.shape == (8,)
