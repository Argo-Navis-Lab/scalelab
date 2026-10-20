"""Config validation tests."""

from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from scalelab.core.config import (
    DIFFICULTIES,
    ArithmeticOps,
    ExperimentConfig,
    RLConfig,
    SamplingParams,
    load_config,
)
from scalelab.core.hashing import run_id
from scalelab.tasks import arithmetic, countdown

REPO_ROOT = Path(__file__).resolve().parents[2]
HELLO_YAML = REPO_ROOT / "configs" / "experiments" / "hello.yaml"


def test_hello_yaml_loads_with_defaults_filled():
    config = load_config(HELLO_YAML)
    assert config.name == "hello"
    assert config.model.attn == "mha"  # default filled in
    assert (config.algo, config.rl) == (None, None)  # sections left out: a warm start


def test_hello_is_a_short_warm_start_on_every_arithmetic_band():
    config = load_config(HELLO_YAML)
    assert (config.warm_start.tokens, config.warm_start.batch_size) == (20_000, 16)
    assert config.task.difficulties == (2, 3, 4, 5)  # from configs/tasks/arithmetic.yaml
    assert (config.eval.every, config.eval.problems) == (1000, 20)  # graded once, at the end


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


# --- arithmetic's operations (added in m2s18) -----------------------------------------------


def test_each_choice_of_ops_is_spelled_the_way_the_task_keeps_it():
    for ops in get_args(ArithmeticOps):
        assert "".join(arithmetic.ArithmeticTask(ops).ops) == ops
    assert "".join(arithmetic.OPS) == "+-*"  # all three, which only the default (None) means


@pytest.mark.parametrize(
    "ops", ["+-*", "*+", "/", ""], ids=["all-three", "out-of-order", "division", "empty"]
)
def test_other_spellings_of_ops_are_refused(raw: dict[str, Any], ops: str):
    raw["task"]["ops"] = ops
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_only_arithmetic_has_operations_to_choose(raw: dict[str, Any]):
    raw["task"] = {"name": "countdown", "difficulties": [3], "ops": "*"}
    with pytest.raises(ValidationError, match="only arithmetic's operations can be chosen"):
        ExperimentConfig.model_validate(raw)


# --- RL's sections (added in m2s18) ---------------------------------------------------------

GRPO = {
    "name": "grpo",
    "estimator": "grpo",
    "clip_low": 0.2,
    "clip_high": 0.2,
    "normalizer": "seq_mean",
}
RL = {"from_run": "c73d1d0f4e234964-s0-2aeaf16b", "flops": 1e15}


def test_an_rl_experiment_has_both_sections(raw: dict[str, Any]):
    config = ExperimentConfig.model_validate(raw | {"algo": GRPO, "rl": RL})
    assert config.algo is not None and config.algo.estimator == "grpo"
    assert config.rl is not None and config.rl.flops == 1e15


@pytest.mark.parametrize("section", ["algo", "rl"])
def test_one_section_without_the_other_is_refused(raw: dict[str, Any], section: str):
    raw[section] = {"algo": GRPO, "rl": RL}[section]
    with pytest.raises(ValidationError, match="needs both an algo and an rl section"):
        ExperimentConfig.model_validate(raw)


def test_rl_settings_left_out_take_their_defaults():
    rl = RLConfig.model_validate(RL)
    assert (rl.prompts_per_step, rl.group_size) == (64, 8)  # the blueprint's typical step
    assert (rl.epochs, rl.micro_batch) == (1, 64)
    assert (rl.lr, rl.weight_decay, rl.grad_clip) == (1e-5, 0.0, 1.0)
    assert rl.sampling == SamplingParams()  # temperature 1 and top_p 1: the policy itself


@pytest.mark.parametrize("missing", ["from_run", "flops"])
def test_the_starting_run_and_the_budget_have_no_defaults(missing: str):
    settings = {key: value for key, value in RL.items() if key != missing}
    with pytest.raises(ValidationError, match="Field required"):
        RLConfig.model_validate(settings)


def test_from_run_takes_any_run_id(config: ExperimentConfig):
    made = run_id(config, seed=12, git_sha="0123456789abcdef0123456789abcdef01234567")
    assert RLConfig.model_validate(RL | {"from_run": made}).from_run == made


@pytest.mark.parametrize(
    "change",
    [
        {"from_run": "c73d1d0f4e234964"},
        {"from_run": "c73d1d0f4e234964-s0-2aeaf16"},
        {"flops": 0},
        {"prompts_per_step": 0},
        {"group_size": 1},
        {"epochs": 0},
        {"micro_batch": 0},
        {"lr": 0},
        {"sampling": {"temperature": 0}},
    ],
    ids=[
        "a-config-hash",
        "short-commit",
        "no-budget",
        "no-problems",
        "one-answer-each",
        "no-updates",
        "empty-micro-batch",
        "zero-lr",
        "zero-temperature",
    ],
)
def test_impossible_rl_settings_are_refused(change: dict[str, Any]):
    with pytest.raises(ValidationError):
        RLConfig.model_validate(RL | change)


# --- added in m2s24: a step limit -----------------------------------------------------------


def test_an_rl_run_has_no_step_limit_unless_asked():
    assert RLConfig.model_validate(RL).max_steps is None
    assert RLConfig.model_validate(RL | {"max_steps": 20}).max_steps == 20
    with pytest.raises(ValidationError):
        RLConfig.model_validate(RL | {"max_steps": 0})
