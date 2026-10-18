"""Config validation tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from scalelab.core.config import ExperimentConfig, OptimConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
HELLO_YAML = REPO_ROOT / "configs" / "experiments" / "hello.yaml"


def test_hello_yaml_loads_with_defaults_filled():
    config = load_config(HELLO_YAML)
    assert config.name == "hello"
    assert config.model.attn == "mha"  # default filled in
    assert config.optim == OptimConfig()  # whole missing section filled in


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
        ("task", "name", "chess"),
        ("task", "max_digits", 1),
        ("task", "max_digits", 13),
    ],
)
def test_out_of_range_values_are_rejected(raw, section, key, bad_value):
    raw[section][key] = bad_value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)
