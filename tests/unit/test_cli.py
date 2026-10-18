"""Tests for the pieces of the `scalelab` command that need no real run."""

import pytest

from scalelab.cli import same_uncommitted_changes, split_overrides
from scalelab.core.envcapture import EnvInfo, GitInfo, HardwareInfo, SoftwareInfo


def test_seed_runtime_and_config_overrides_are_sorted_apart():
    seed, runtime, config = split_overrides(
        ["seed=2", "runtime.worker_name=colab-1", "model=ladder_s/m2", "optim.lr=1e-3"]
    )
    assert seed == 2
    assert runtime == {"worker_name": "colab-1"}
    assert config == ["model=ladder_s/m2", "optim.lr=1e-3"]


def test_seed_defaults_to_zero():
    assert split_overrides([])[0] == 0


@pytest.mark.parametrize("bad", ["seed=-1", "seed=abc", "seed=1.5", "seed=true", "seed=4294967296"])
def test_bad_seeds_are_rejected(bad):
    with pytest.raises(ValueError, match="seed must be a whole number"):
        split_overrides([bad])


def env_with(diff_sha256: str | None) -> EnvInfo:
    sha = "0123456789abcdef0123456789abcdef01234567"
    return EnvInfo(
        git=GitInfo(sha=sha, dirty=diff_sha256 is not None, diff_sha256=diff_sha256),
        software=SoftwareInfo(python="3.12.14", os="test-os", packages={}),
        hardware=HardwareInfo(cpu_model="Test CPU", cpu_count=4),
    )


@pytest.mark.parametrize(
    ("then", "now", "same"),
    [
        (None, None, True),  # clean both times
        ("a" * 64, "a" * 64, True),  # the same uncommitted changes
        ("a" * 64, "b" * 64, False),  # different uncommitted changes
        (None, "b" * 64, False),  # clean then, edited now
    ],
)
def test_resuming_needs_the_same_uncommitted_changes(then, now, same):
    assert same_uncommitted_changes(env_with(then).model_dump_json(), env_with(now)) is same
