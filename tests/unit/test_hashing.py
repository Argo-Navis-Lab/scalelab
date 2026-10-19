"""Run identity tests."""

import re

import pytest
from pydantic import BaseModel

from scalelab.core.config import ExperimentConfig
from scalelab.core.hashing import canonical_json, config_hash, run_id

# A made-up 40-character commit SHA, the shape `git rev-parse HEAD` prints.
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


# --- canonical_json and config_hash -------------------------------------------


def test_canonical_json_sorts_keys_and_drops_whitespace():
    class Point(BaseModel):
        y: int
        x: int

    assert canonical_json(Point(y=2, x=1)) == '{"x":1,"y":2}'


def test_field_order_in_code_does_not_change_hash():
    class AB(BaseModel):
        a: int
        b: int

    class BA(BaseModel):
        b: int
        a: int

    assert config_hash(AB(a=1, b=2)) == config_hash(BA(a=1, b=2))


def test_reordered_keys_give_same_hash(raw):
    reordered = {
        "task": {"difficulties": [2, 3, 4, 5], "name": "arithmetic"},
        "model": {"n_heads": 2, "d_model": 128, "n_layers": 4, "name": "m1"},
        "name": "hello",
    }
    a = ExperimentConfig.model_validate(raw)
    b = ExperimentConfig.model_validate(reordered)
    assert config_hash(a) == config_hash(b)


def test_written_out_defaults_give_same_hash(raw):
    short = ExperimentConfig.model_validate(raw)

    raw["model"]["attn"] = "mha"  # the default, written out
    raw["optim"] = {"lr": 3e-4, "weight_decay": 0}  # 0, not 0.0: still the same value
    long = ExperimentConfig.model_validate(raw)

    assert config_hash(short) == config_hash(long)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "n_layers", 5),
        ("model", "attn", "gqa"),
        ("task", "difficulties", [2, 3]),
        ("optim", "lr", 1e-3),
        ("warm_start", "tokens", 1000),
    ],
)
def test_one_value_change_gives_different_hash(raw, section, key, value):
    before = config_hash(ExperimentConfig.model_validate(raw))
    raw.setdefault(section, {})[key] = value
    after = config_hash(ExperimentConfig.model_validate(raw))
    assert before != after


def test_config_hash_is_16_lowercase_hex(config):
    assert re.fullmatch(r"[0-9a-f]{16}", config_hash(config))


def test_infinite_values_cannot_be_hashed(raw):
    raw["optim"] = {"lr": float("inf")}
    config = ExperimentConfig.model_validate(raw)
    with pytest.raises(ValueError, match="not JSON compliant"):
        config_hash(config)


# --- run_id ---------------------------------------------------------------------


def test_run_id_format(config):
    rid = run_id(config, seed=3, git_sha=GIT_SHA)
    assert rid == f"{config_hash(config)}-s3-01234567"
    assert re.fullmatch(r"[0-9a-f]{16}-s\d+-[0-9a-f]{8}", rid)


def test_different_seeds_share_the_config_hash_prefix(config):
    ids = [run_id(config, seed=s, git_sha=GIT_SHA) for s in range(3)]
    assert len(set(ids)) == 3  # three different runs...
    assert {rid.split("-")[0] for rid in ids} == {config_hash(config)}  # ...of one experiment


@pytest.mark.parametrize(
    "sha",
    [GIT_SHA.upper(), f"  {GIT_SHA}\n", "01234567"],
    ids=["upper-case", "spaces-and-newline", "exactly-8-chars"],
)
def test_equivalent_git_shas_give_same_run_id(config, sha):
    assert run_id(config, seed=0, git_sha=sha) == run_id(config, seed=0, git_sha=GIT_SHA)


@pytest.mark.parametrize("bad_seed", [-1, -42])
def test_negative_seed_raises(config, bad_seed):
    with pytest.raises(ValueError, match="seed must be non-negative"):
        run_id(config, seed=bad_seed, git_sha=GIT_SHA)


@pytest.mark.parametrize(
    "bad_sha",
    ["", "0123456", "0123456z", "0123 4567", "not-a-sha"],
    ids=["empty", "7-chars", "non-hex-z", "inner-space", "words"],
)
def test_bad_git_sha_raises(config, bad_sha):
    with pytest.raises(ValueError, match="git_sha must be"):
        run_id(config, seed=0, git_sha=bad_sha)
