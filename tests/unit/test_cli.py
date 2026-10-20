"""Tests for the pieces of the `scalelab` command that need no real run."""

import pytest

from scalelab.cli import duration, same_uncommitted_changes, speed, split_overrides
from scalelab.core.envcapture import EnvInfo, GitInfo, HardwareInfo, SoftwareInfo


def test_seed_runtime_and_config_overrides_are_sorted_apart():
    seed, runtime, config = split_overrides(
        ["seed=2", "runtime.worker_name=colab-1", "model=ladder_s/m2", "warm_start.lr=1e-3"]
    )
    assert seed == 2
    assert runtime == {"worker_name": "colab-1"}
    assert config == ["model=ladder_s/m2", "warm_start.lr=1e-3"]


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


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (40, "40 s"),
        (100, "2 min"),
        (12 * 60, "12 min"),
        (2 * 3600, "2.0 h"),
        (3.24 * 3600, "3.2 h"),
    ],
)
def test_durations_read_the_way_people_say_them(seconds: float, text: str):
    assert duration(seconds) == text


def test_speed_and_time_left():
    # 10,000 tokens in 2 s is 5,000 a second; 50,000 of 100,000 are left: 10 s to go.
    assert speed(10_000, 2.0, 50_000, 100_000) == "5,000 tokens/s, about 10 s to go"


def test_once_the_budget_is_spent_nothing_is_left():
    # The last step reads a little past the budget (it counts as "at least"), not before it.
    assert speed(1_000, 1.0, 100_500, 100_000) == "1,000 tokens/s, about 0 s to go"
