"""Config composition tests: defaults lists, file values and command-line overrides."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from scalelab.core.config import compose_config, load_config, parse_override
from scalelab.core.hashing import config_hash

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
HELLO = REPO_CONFIGS / "experiments" / "hello.yaml"


@pytest.fixture
def configs(tmp_path):
    """A small configs/ folder of our own: two models, one task, one experiment."""
    root = tmp_path / "configs"
    models, tasks = root / "models", root / "tasks"
    for folder in (models, tasks, root / "experiments"):
        folder.mkdir(parents=True)
    (models / "tiny.yaml").write_text("name: tiny\nn_layers: 2\nd_model: 64\nn_heads: 1\n")
    (models / "wide.yaml").write_text("name: wide\nn_layers: 2\nd_model: 256\nn_heads: 4\n")
    (tasks / "arithmetic.yaml").write_text("name: arithmetic\nmax_digits: 4\n")
    (root / "experiments" / "exp.yaml").write_text(
        "defaults:\n  - model: tiny\n  - task: arithmetic\n\nname: exp\ntask:\n  max_digits: 6\n"
    )
    return root


# --- the real configs in this repository --------------------------------------------------


def test_hello_hash_is_pinned():
    # Golden value: composing hello from separate files must not change what it is.
    # If this changes, every past hello run becomes a different experiment. Say so loudly.
    assert config_hash(load_config(HELLO)) == "3f7ffa68d30146bd"


@pytest.mark.parametrize("name", ["m1", "m2", "m3", "m4", "m5", "m6", "m7"])
def test_every_ladder_model_is_valid_with_64_wide_heads(name):
    config = load_config(HELLO, [f"model=ladder_s/{name}"])
    assert config.model.name == name
    assert config.model.head_dim == 64


# --- how composition works ----------------------------------------------------------------


def test_defaults_fill_in_whole_sections(configs):
    composed = compose_config(configs / "experiments" / "exp.yaml")
    assert composed["model"] == {"name": "tiny", "n_layers": 2, "d_model": 64, "n_heads": 1}
    assert "defaults" not in composed


def test_the_file_overrides_its_defaults_value_by_value(configs):
    composed = compose_config(configs / "experiments" / "exp.yaml")
    assert composed["task"] == {"name": "arithmetic", "max_digits": 6}  # name kept, digits changed


def test_command_line_overrides_win_over_everything(configs):
    config = load_config(configs / "experiments" / "exp.yaml", ["task.max_digits=8"])
    assert config.task.max_digits == 8


def test_choosing_a_different_group_file(configs):
    config = load_config(configs / "experiments" / "exp.yaml", ["model=wide"])
    assert config.model.name == "wide"


def test_overrides_can_set_values_with_no_default(configs):
    config = load_config(configs / "experiments" / "exp.yaml", ["optim.lr=0.001"])
    assert config.optim.lr == 0.001


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("n=6", ("n", 6)),
        ("lr=0.001", ("lr", 0.001)),
        ("flag=true", ("flag", True)),
        ("name=m1", ("name", "m1")),
        ("model=ladder_s/m4", ("model", "ladder_s/m4")),
    ],
)
def test_override_values_are_read_as_yaml(text, expected):
    assert parse_override(text) == expected


# --- mistakes are caught ------------------------------------------------------------------


def test_typo_in_an_override_is_rejected(configs):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(configs / "experiments" / "exp.yaml", ["model.n_layer=3"])


def test_override_without_equals_is_rejected(configs):
    with pytest.raises(ValueError, match="key=value"):
        load_config(configs / "experiments" / "exp.yaml", ["nonsense"])


def test_missing_group_file_is_named(configs):
    with pytest.raises(FileNotFoundError, match="no model config 'huge'"):
        load_config(configs / "experiments" / "exp.yaml", ["model=huge"])


def test_unknown_group_in_defaults_is_rejected(configs):
    path = configs / "experiments" / "odd.yaml"
    path.write_text("defaults:\n  - optimizer: adam\nname: odd\n")
    with pytest.raises(ValueError, match="unknown config group 'optimizer'"):
        compose_config(path)


def test_experiment_must_live_under_a_configs_folder(tmp_path):
    path = tmp_path / "loose.yaml"
    path.write_text("defaults:\n  - model: tiny\nname: loose\n")
    with pytest.raises(ValueError, match="inside a folder named 'configs'"):
        compose_config(path)
