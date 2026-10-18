"""Seed and determinism tests."""

import os
import random

import numpy as np
import pytest
import torch

from scalelab.core.determinism import STREAMS, Seeds, derive_seeds, seed_everything


@pytest.fixture(autouse=True)
def restore_torch_settings(monkeypatch):
    """These tests flip global PyTorch switches. Put them back afterwards."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)  # restored after the test
    before = torch.are_deterministic_algorithms_enabled()
    yield  # the test runs here
    torch.use_deterministic_algorithms(before)


# --- deriving child seeds ----------------------------------------------------------


def test_child_seeds_are_pinned():
    # Golden values: if a NumPy upgrade or a code change ever alters these, every
    # past run's random numbers would silently change. This test makes that loud.
    assert derive_seeds(0) == Seeds(
        root=0,
        init=4334430513956379144,
        data=2440950710608614359,
        sampling=8226343694796210948,
    )


def test_same_seed_same_children():
    assert derive_seeds(42) == derive_seeds(42)


def test_children_all_differ():
    seeds = derive_seeds(0)
    values = [seeds.root] + [getattr(seeds, name) for name in STREAMS]
    assert len(set(values)) == len(values)


def test_different_seeds_give_different_children():
    a, b = derive_seeds(0), derive_seeds(1)
    assert all(getattr(a, name) != getattr(b, name) for name in STREAMS)


def test_children_fit_in_a_database_integer():
    seeds = derive_seeds(2**32 - 1)
    assert all(0 <= getattr(seeds, name) < 2**63 for name in STREAMS)


@pytest.mark.parametrize("bad_seed", [-1, 2**32])
def test_out_of_range_seed_raises(bad_seed):
    with pytest.raises(ValueError, match="seed must be between"):
        derive_seeds(bad_seed)


# --- seeding the global generators -----------------------------------------------------


def draw_from_global_generators() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), torch.rand(1).item()


def test_seed_everything_makes_global_generators_repeat():
    seed_everything(3)
    first = draw_from_global_generators()
    seed_everything(3)
    assert draw_from_global_generators() == first


def test_seed_everything_returns_the_child_seeds():
    assert seed_everything(5) == derive_seeds(5)


def test_streams_do_not_disturb_each_other():
    seed_everything(0)
    weights = torch.nn.Linear(4, 4).weight.detach().clone()

    seeds = seed_everything(0)
    sampler = torch.Generator().manual_seed(seeds.sampling)
    torch.rand(1000, generator=sampler)  # lots of sampling first...
    np.random.default_rng(seeds.data).random(1000)  # ...and lots of data drawing
    assert torch.equal(torch.nn.Linear(4, 4).weight, weights)  # ...initialization unchanged


def test_without_streams_one_extra_draw_changes_initialization():
    # The problem streams solve: everything sharing one generator.
    seed_everything(0)
    weights = torch.nn.Linear(4, 4).weight.detach().clone()
    seed_everything(0)
    torch.rand(1)  # one innocent-looking draw from the shared global generator
    assert not torch.equal(torch.nn.Linear(4, 4).weight, weights)


# --- deterministic mode ------------------------------------------------------------


def test_deterministic_mode_on():
    seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_deterministic_mode_off():
    seed_everything(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False


@pytest.mark.parametrize(("has_gpu", "expected"), [(False, True), (True, False)])
def test_default_is_on_for_cpu_and_off_for_gpu(monkeypatch, has_gpu, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: has_gpu)
    seed_everything(0)  # deterministic not given: use the default
    assert torch.are_deterministic_algorithms_enabled() is expected
