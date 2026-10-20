"""The algorithm presets: each file in configs/algos/ is a valid AlgoConfig set as the
blueprint's table says, impossible settings are refused, and every name the config allows is
one the code can run."""

from pathlib import Path
from typing import Any, get_args

import pytest
import torch
import yaml
from pydantic import ValidationError

from scalelab.core.config import AlgoConfig, Estimator, Normalizer
from scalelab.rl.advantages import advantages
from scalelab.rl.losses import normalize

ALGOS = Path(__file__).resolve().parents[2] / "configs" / "algos"

# The blueprint's table (Section 6.4): estimator, clip_low, clip_high, normalizer, kl_coef,
# dynamic_sampling, overlong.
TABLE: dict[str, tuple[Any, ...]] = {
    "grpo": ("grpo", 0.2, 0.2, "seq_mean", 0.04, False, "none"),
    "dr_grpo": ("dr_grpo", 0.2, 0.2, "constant", 0.0, False, "none"),
    "rloo": ("rloo", 0.2, 0.2, "seq_mean", 0.0, False, "none"),
    "dapo": ("grpo", 0.2, 0.28, "token_mean", 0.0, True, "soft_penalty"),
}
# The fewest knobs an AlgoConfig needs.
REQUIRED = {
    "name": "test",
    "estimator": "grpo",
    "clip_low": 0.2,
    "clip_high": 0.2,
    "normalizer": "seq_mean",
}


def preset(name: str) -> AlgoConfig:
    with open(ALGOS / f"{name}.yaml") as f:
        return AlgoConfig.model_validate(yaml.safe_load(f))


# --- the presets ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TABLE))
def test_each_preset_matches_the_blueprints_table(name: str):
    algo = preset(name)
    assert algo.name == name
    assert (
        algo.estimator,
        algo.clip_low,
        algo.clip_high,
        algo.normalizer,
        algo.kl_coef,
        algo.dynamic_sampling,
        algo.overlong,
    ) == TABLE[name]


def test_every_preset_file_is_in_the_table():
    assert sorted(path.stem for path in ALGOS.glob("*.yaml")) == sorted(TABLE)


# --- the settings ---------------------------------------------------------------------------


def test_knobs_left_out_take_quiet_defaults():
    algo = AlgoConfig.model_validate(REQUIRED)
    assert (algo.kl_coef, algo.format_weight, algo.dynamic_sampling, algo.overlong) == (
        0.0,
        0.0,
        False,
        "none",
    )
    assert algo.normalizer_constant == 256


@pytest.mark.parametrize(
    "change",
    [
        {"clip_low": 1.0},
        {"clip_high": -0.1},
        {"kl_coef": -0.04},
        {"format_weight": -1.0},
        {"normalizer_constant": 0},
        {"estimator": "ppo"},
        {"normalizer": "mean"},
        {"overlong": "cut"},
        {"temperature": 1.0},
    ],
    ids=[
        "no-room-below",
        "negative-room",
        "negative-kl",
        "negative-format",
        "zero-constant",
        "unknown-estimator",
        "unknown-normalizer",
        "unknown-overlong",
        "not-an-algo-knob",
    ],
)
def test_impossible_settings_are_refused(change: dict[str, Any]):
    with pytest.raises(ValidationError):
        AlgoConfig.model_validate(REQUIRED | change)


# --- the config and the code agree ----------------------------------------------------------


@pytest.mark.parametrize("estimator", get_args(Estimator))
def test_every_estimator_the_config_allows_runs(estimator: Estimator):
    assert advantages(torch.tensor([1.0, 0.0]), torch.tensor([0, 0]), estimator).shape == (2,)


@pytest.mark.parametrize("normalizer", get_args(Normalizer))
def test_every_normalizer_the_config_allows_runs(normalizer: Normalizer):
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert normalize(torch.ones(1, 2), mask, normalizer, constant=2).isfinite()
