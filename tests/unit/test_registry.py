"""Run registry tests."""

import hashlib
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, insert, inspect, select
from sqlalchemy.exc import IntegrityError

from scalelab.core.config import ExperimentConfig
from scalelab.core.envcapture import (
    EnvInfo,
    GitInfo,
    GpuInfo,
    HardwareInfo,
    SoftwareInfo,
    capture_env,
)
from scalelab.core.hashing import canonical_json, config_hash
from scalelab.core.registry import Registry, RunExistsError, experiments, runs


@pytest.fixture
def env() -> EnvInfo:
    """A made-up but valid environment, so these tests never depend on your machine."""
    return EnvInfo(
        git=GitInfo(sha="0123456789abcdef0123456789abcdef01234567", dirty=False),
        software=SoftwareInfo(python="3.12.14", os="test-os", packages={"scalelab": "0.1.0"}),
        hardware=HardwareInfo(cpu_model="Test CPU", cpu_count=4),
    )


@pytest.fixture
def registry(tmp_path) -> Registry:
    """A brand-new, empty registry file in a temporary folder."""
    return Registry.sqlite(tmp_path / "registry.db")


def count_rows(registry: Registry, table) -> int:
    with registry.engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar()


# --- creating and reopening --------------------------------------------------------


def test_creates_file_and_tables(tmp_path):
    path = tmp_path / "sub" / "registry.db"  # the folder doesn't exist yet either
    registry = Registry.sqlite(path)
    assert path.is_file()
    assert sorted(inspect(registry.engine).get_table_names()) == ["experiments", "runs"]


def test_rows_survive_reopening_the_file(tmp_path, config, env):
    path = tmp_path / "registry.db"
    run_id = Registry.sqlite(path).add_run(config, seed=0, env=env)
    reopened = Registry.sqlite(path)  # like a new Colab session finding the file
    assert reopened.get_run(run_id) is not None


# --- add_run -----------------------------------------------------------------------


def test_add_run_records_a_complete_row(registry, config, env):
    before = datetime.now(UTC)
    run_id = registry.add_run(config, seed=7, env=env)
    run = registry.get_run(run_id)

    assert run.run_id == f"{config_hash(config)}-s7-01234567"
    assert run.config_hash == config_hash(config)
    assert run.seed == 7
    assert run.git_sha == env.git.sha
    assert run.git_dirty is False
    assert run.status == "running"
    assert run.executor == "local"
    assert run.hardware == "cpu"
    assert before <= run.started_at <= datetime.now(UTC)
    assert run.ended_at is None
    assert EnvInfo.model_validate_json(run.env_json) == env


def test_your_real_environment_round_trips(registry, config):
    env = capture_env()
    run_id = registry.add_run(config, seed=0, env=env)
    assert EnvInfo.model_validate_json(registry.get_run(run_id).env_json) == env


def test_stored_config_is_exactly_what_was_hashed(registry, config, env):
    registry.add_run(config, seed=0, env=env)
    stored = registry.experiment_json(config_hash(config))
    assert stored == canonical_json(config)
    assert hashlib.sha256(stored.encode("utf-8")).hexdigest()[:16] == config_hash(config)


def test_seeds_share_one_experiment_row(registry, config, env):
    for seed in range(3):
        registry.add_run(config, seed=seed, env=env)
    assert count_rows(registry, experiments) == 1
    assert [run.seed for run in registry.list_runs(config_hash(config))] == [0, 1, 2]


def test_list_runs_filters_by_experiment(registry, config, env, raw):
    raw["model"]["n_layers"] = 6
    other = ExperimentConfig.model_validate(raw)
    registry.add_run(config, seed=0, env=env)
    registry.add_run(other, seed=0, env=env)
    assert len(registry.list_runs()) == 2
    assert len(registry.list_runs(config_hash(config))) == 1
    assert count_rows(registry, experiments) == 2


def test_same_run_twice_is_refused(registry, config, env):
    registry.add_run(config, seed=0, env=env)
    with pytest.raises(RunExistsError, match="already in the registry"):
        registry.add_run(config, seed=0, env=env)
    assert count_rows(registry, runs) == 1


def test_new_commit_means_new_run(registry, config, env):
    first = registry.add_run(config, seed=0, env=env)
    git = GitInfo(sha="fedcba98" * 5, dirty=False)  # a different commit
    second = registry.add_run(config, seed=0, env=env.model_copy(update={"git": git}))
    assert first != second
    assert count_rows(registry, experiments) == 1  # same experiment, different code


def test_dirty_tree_is_recorded(registry, config, env):
    git = GitInfo(sha=env.git.sha, dirty=True, diff_sha256="ab" * 32)
    run_id = registry.add_run(config, seed=0, env=env.model_copy(update={"git": git}))
    assert registry.get_run(run_id).git_dirty is True


def test_hardware_label_names_the_gpus(registry, config, env):
    t4 = GpuInfo(name="Tesla T4", memory_mib=15360, driver="550.54.15")
    hardware = HardwareInfo(cpu_model="Xeon", cpu_count=4, gpus=(t4, t4))
    run_id = registry.add_run(config, seed=0, env=env.model_copy(update={"hardware": hardware}))
    assert registry.get_run(run_id).hardware == "2x Tesla T4"


# --- finish_run --------------------------------------------------------------------


@pytest.mark.parametrize("status", ["done", "failed"])
def test_finish_run_sets_status_and_end_time(registry, config, env, status):
    run_id = registry.add_run(config, seed=0, env=env)
    registry.finish_run(run_id, status)
    run = registry.get_run(run_id)
    assert run.status == status
    assert run.started_at <= run.ended_at


def test_finishing_twice_is_refused(registry, config, env):
    run_id = registry.add_run(config, seed=0, env=env)
    registry.finish_run(run_id, "done")
    with pytest.raises(ValueError, match="no running run"):
        registry.finish_run(run_id, "failed")
    assert registry.get_run(run_id).status == "done"  # the first answer stands


def test_finishing_an_unknown_run_is_refused(registry):
    with pytest.raises(ValueError, match="no running run"):
        registry.finish_run("0000000000000000-s0-00000000", "done")


def test_finish_status_must_be_done_or_failed(registry, config, env):
    run_id = registry.add_run(config, seed=0, env=env)
    with pytest.raises(ValueError, match="only finish as"):
        registry.finish_run(run_id, "exploded")


# --- reading -----------------------------------------------------------------------


def test_unknown_run_and_experiment_give_none(registry):
    assert registry.get_run("0000000000000000-s0-00000000") is None
    assert registry.experiment_json("0000000000000000") is None


def test_foreign_keys_are_enforced(registry, env):
    orphan = {
        "run_id": "x",
        "config_hash": "0000000000000000",  # no such experiment
        "seed": 0,
        "git_sha": env.git.sha,
        "git_dirty": False,
        "status": "running",
        "executor": "local",
        "hardware": "cpu",
        "started_at": datetime.now(UTC),
        "env_json": "{}",
    }
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        with registry.engine.begin() as conn:
            conn.execute(insert(runs).values(**orphan))
