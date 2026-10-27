"""The registry's work queue (m4s5): sweeps, their jobs, and the workers' leases."""

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from scalelab.core.registry import Job, LeaseLostError, Registry, SweepExistsError

SHA = "a" * 40  # the commit the worker runs at


def job(sweep_id: str, number: int, **fields: object) -> Job:
    """Job `number` of a sweep: 60 s on the t4, unless `fields` say otherwise."""
    values: dict[str, object] = {
        "job_id": f"{sweep_id}-{number}",
        "sweep_id": sweep_id,
        "number": number,
        "run_id": f"run-{number}",
        "experiment": "configs/experiments/rl.yaml",
        "overrides": ("algo=grpo", f"seed={number}"),
        "hardware": "t4",
        "seconds": 60.0,
    }
    return Job.model_validate(values | fields)


def submit(registry: Registry, sweep_id: str, count: int, sha: str = SHA, **fields: object) -> None:
    jobs = [job(sweep_id, number, **fields) for number in range(1, count + 1)]
    registry.enqueue(sweep_id, "{}", sha, jobs)


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry.sqlite(tmp_path / "registry.db")


def test_a_sweep_and_its_jobs_are_recorded_together(registry: Registry) -> None:
    submit(registry, "s", 3)
    sweep = registry.get_sweep("s")
    assert sweep is not None
    assert (sweep.git_sha, sweep.created_at.tzinfo) == (SHA, UTC)
    jobs = registry.list_jobs("s")
    assert jobs == [job("s", 1), job("s", 2), job("s", 3)]  # overrides back out of JSON
    assert registry.get_sweep("t") is None
    assert registry.list_jobs("t") == []


def test_a_sweep_is_recorded_once(registry: Registry) -> None:
    submit(registry, "s", 2)
    with pytest.raises(SweepExistsError, match="sweep s is already"):
        submit(registry, "s", 3)
    assert len(registry.list_jobs("s")) == 2


def test_a_failed_enqueue_records_nothing(registry: Registry) -> None:
    with pytest.raises(IntegrityError):  # the second job's id is the first one's
        registry.enqueue("s", "{}", SHA, [job("s", 1), job("s", 1)])
    assert registry.get_sweep("s") is None


def test_leases_go_in_sweep_order_then_job_order(registry: Registry) -> None:
    submit(registry, "first", 2)
    submit(registry, "second", 1)
    leased = [registry.lease("w", SHA, "t4", 3600, ttl_s=600) for _ in range(4)]
    ids = [None if found is None else found.job_id for found in leased]
    assert ids == ["first-1", "first-2", "second-1", None]
    first = leased[0]
    assert first is not None
    assert (first.state, first.lease_owner, first.attempts) == ("leased", "w", 1)
    assert first.lease_expires_at is not None
    left = first.lease_expires_at - datetime.now(UTC)
    assert timedelta(seconds=590) < left <= timedelta(seconds=600)


def test_a_worker_only_gets_jobs_it_can_run(registry: Registry) -> None:
    submit(registry, "old", 1, sha="b" * 40)  # submitted at another commit
    submit(registry, "a100", 1, hardware="a100-40g")
    submit(registry, "finished", 1, state="done")
    submit(registry, "long", 1, seconds=7200.0)
    submit(registry, "fits", 1)
    found = registry.lease("w", SHA, "t4", 3600, ttl_s=600)
    assert found is not None
    assert found.job_id == "fits-1"
    assert registry.lease("w", SHA, "t4", 3600, ttl_s=600) is None
    found = registry.lease("w", SHA, "t4", math.inf, ttl_s=600)
    assert found is not None
    assert found.job_id == "long-1"
    assert registry.lease("w", SHA, "t4", math.inf, ttl_s=600) is None


def test_an_expired_lease_goes_to_the_next_worker_first(registry: Registry) -> None:
    submit(registry, "s", 2)
    lost = registry.lease("w1", SHA, "t4", 3600, ttl_s=0)  # expires at once
    assert lost is not None
    again = registry.lease("w2", SHA, "t4", 3600, ttl_s=600)
    assert again is not None
    assert (again.job_id, again.lease_owner, again.attempts) == (lost.job_id, "w2", 2)
    with pytest.raises(LeaseLostError, match="w1 no longer holds job s-1"):
        registry.heartbeat(lost.job_id, "w1", ttl_s=600)  # w1 woke up too late


def test_a_heartbeat_keeps_the_lease_and_marks_the_job_running(registry: Registry) -> None:
    submit(registry, "s", 1)
    leased = registry.lease("w", SHA, "t4", 3600, ttl_s=0)
    assert leased is not None
    registry.heartbeat(leased.job_id, "w", ttl_s=600)  # late, but nobody took it meanwhile
    [running] = registry.list_jobs("s")
    assert (running.state, running.lease_owner, running.attempts) == ("running", "w", 1)
    assert running.lease_expires_at is not None
    assert running.lease_expires_at > datetime.now(UTC) + timedelta(seconds=590)
    assert registry.lease("w2", SHA, "t4", 3600, ttl_s=600) is None
    with pytest.raises(LeaseLostError):
        registry.heartbeat(leased.job_id, "w2", ttl_s=600)


def test_a_finished_job_is_never_leased_again(registry: Registry) -> None:
    submit(registry, "s", 2)
    for state in ("done", "failed"):
        leased = registry.lease("w", SHA, "t4", 3600, ttl_s=0)
        assert leased is not None
        registry.complete(leased.job_id, "w", state)
    found = [(j.state, j.lease_owner, j.lease_expires_at) for j in registry.list_jobs("s")]
    assert found == [("done", "w", None), ("failed", "w", None)]
    assert registry.lease("w", SHA, "t4", 3600, ttl_s=600) is None
    with pytest.raises(LeaseLostError):
        registry.complete("s-1", "w", "done")
    with pytest.raises(ValueError, match="not 'queued'"):
        registry.complete("s-1", "w", "queued")


def test_a_released_job_is_queued_again_for_anyone(registry: Registry) -> None:
    submit(registry, "s", 1)
    assert registry.lease("w1", SHA, "t4", math.inf, 600) is not None
    registry.release("s-1", "w1")  # m4s6: its run stopped, a checkpoint saved
    (released,) = registry.list_jobs("s")
    assert (released.state, released.lease_owner, released.lease_expires_at) == (
        "queued",
        None,
        None,
    )
    assert released.attempts == 1
    with pytest.raises(LeaseLostError):
        registry.release("s-1", "w1")  # no longer its own
    again = registry.lease("w2", SHA, "t4", math.inf, 600)
    assert again is not None
    assert (again.job_id, again.lease_owner, again.attempts) == ("s-1", "w2", 2)
