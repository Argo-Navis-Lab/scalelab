"""Config validation tests."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scalelab.core.config import DIFFICULTIES, ExperimentConfig, OptimConfig, load_config
from scalelab.tasks import arithmetic, countdown

REPO_ROOT = Path(__file__).resolve().parents[2]
HELLO_YAML = REPO_ROOT / "configs" / "experiments" / "hello.yaml"


def test_hello_yaml_loads_with_defaults_filled():
    config = load_config(HELLO_YAML)
    assert config.name == "hello"
    assert config.model.attn == "mha"  # default filled in
    assert config.optim == OptimConfig()  # whole missing section filled in


def test_hello_is_a_short_warm_start_on_every_arithmetic_band():
    config = load_config(HELLO_YAML)
    assert (config.warm_start.tokens, config.warm_start.batch_size) == (20_000, 16)
    assert config.task.difficulties == (2, 3, 4, 5)  # from configs/tasks/arithmetic.yaml


def test_head_dim_is_width_over_heads(config):
    assert config.model.head_dim == 64  # 128 // 2


def test_config_is_frozen(config):
    with pytest.raises(ValidationError, match="frozen"):
        config.name = "renamed"


def test_nested_config_is_frozen(config):
    with pytest.raises(ValidationError, match="frozen"):
        config.model.n_layers = 8


def test_d_model_must_be_divisible_by_n_heads(raw):
    raw["model"]["n_heads"] = 3  # 128 / 3 is not a whole number
    with pytest.raises(ValidationError, match="must be divisible by"):
        ExperimentConfig.model_validate(raw)


def test_head_dim_must_be_even(raw: dict[str, Any]):
    raw["model"]["d_model"], raw["model"]["n_heads"] = 126, 2  # 63 numbers per head
    with pytest.raises(ValidationError, match="must be even"):
        ExperimentConfig.model_validate(raw)


def test_typo_at_top_level_is_rejected(raw):
    raw["nmae"] = "oops"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig.model_validate(raw)


def test_typo_inside_a_section_is_rejected(raw):
    raw["model"]["n_layer"] = raw["model"].pop("n_layers")  # 'n_layers' misspelled
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig.model_validate(raw)


def test_runtime_fields_cannot_sneak_into_experiment_config(raw):
    raw["output_dir"] = "/content/drive/MyDrive/runs"  # belongs in RuntimeConfig
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        ("model", "n_layers", 0),
        ("model", "d_model", -128),
        ("model", "attn", "flash"),
        ("model", "context", 0),
        ("task", "name", "chess"),
        ("task", "difficulties", []),
        ("task", "difficulties", [1]),
        ("task", "difficulties", [2, 13]),
    ],
)
def test_out_of_range_values_are_rejected(raw, section, key, bad_value):
    raw[section][key] = bad_value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


# --- task difficulties ----------------------------------------------------------------------


def test_the_difficulty_limits_are_the_tasks_own():
    assert DIFFICULTIES["arithmetic"] == range(arithmetic.MIN_DIGITS, arithmetic.MAX_DIGITS + 1)
    assert DIFFICULTIES["countdown"] == range(countdown.MIN_NUMBERS, countdown.MAX_NUMBERS + 1)


def test_a_bad_difficulty_is_named_with_the_tasks_limits(raw: dict[str, Any]):
    raw["task"] = {"name": "countdown", "difficulties": [2, 3, 7]}
    message = r"countdown difficulties run from 3 to 6, got \[2, 7\]"  # \[ is a plain "["
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(raw)
