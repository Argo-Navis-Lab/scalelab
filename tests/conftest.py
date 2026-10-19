"""Shared test fixtures. pytest finds this file by name and loads it automatically."""

import copy
from typing import Any

import pytest

from scalelab.core.config import ExperimentConfig

# A known-good config, written the way a YAML file would give it to us
# (hello.yaml's model and task; every other section left to its defaults).
BASE_RAW = {
    "name": "hello",
    "model": {"name": "m1", "n_layers": 4, "d_model": 128, "n_heads": 2},
    "task": {"name": "arithmetic", "difficulties": [2, 3, 4, 5]},
}


@pytest.fixture
def raw() -> dict[str, Any]:
    """A fresh copy of BASE_RAW that a test may edit freely."""
    return copy.deepcopy(BASE_RAW)


@pytest.fixture
def config(raw: dict[str, Any]) -> ExperimentConfig:
    """BASE_RAW validated into a real ExperimentConfig."""
    return ExperimentConfig.model_validate(raw)
