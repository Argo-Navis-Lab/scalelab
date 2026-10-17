"""Shared test fixtures. pytest finds this file by name and loads it automatically."""

import copy

import pytest

from scalelab.core.config import ExperimentConfig

# A known-good config, written the way a YAML file would give it to us
# (the same values as configs/experiments/hello.yaml, defaults left out).
BASE_RAW = {
    "name": "hello",
    "model": {"name": "m1", "n_layers": 4, "d_model": 128, "n_heads": 2},
    "task": {"name": "arithmetic", "max_digits": 4},
}


@pytest.fixture
def raw() -> dict:
    """A fresh copy of BASE_RAW that a test may edit freely."""
    return copy.deepcopy(BASE_RAW)


@pytest.fixture
def config(raw: dict) -> ExperimentConfig:
    """BASE_RAW validated into a real ExperimentConfig."""
    return ExperimentConfig.model_validate(raw)
