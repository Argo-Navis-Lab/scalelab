"""`scalelab worker` (m4s6): jobs leased, run in-process, heartbeated, finished, given back."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import yaml

from scalelab import cli
from scalelab.cli import main
from scalelab.core.envcapture import EnvInfo, capture_env
from scalelab.core.registry import Job, Registry

REPO = Path(__file__).resolve().parents[2]
HELLO = REPO / "configs" / "experiments" / "hello.yaml"


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fresh SCALELAB_HOME, run from the repo's folder."""
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(REPO)
    return tmp_path / "home"


@pytest.fixture
def clean(monkeypatch: pytest.MonkeyPatch) -> EnvInfo:
    """This checkout's environment, as if it had no uncommitted changes."""
    env = capture_env()
    git = env.git.model_copy(update={"dirty": False, "diff_sha256": None})
    env = env.model_copy(update={"git": git})
    monkeypatch.setattr(cli, "capture_env", lambda: env)
    return env


def submit(tmp_path: Path, env: EnvInfo, seeds: list[int]) -> str:
    """Submit a sweep of hello warm starts, one job per seed, priced on the CPU; its id."""
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump({"name": "hello", "experiment": str(HELLO), "seeds": seeds}))
    assert main(["sweep", "submit", str(path), "--hardware", "cpu"]) == 0
    return f"hello-{env.git.sha[:8]}"


def jobs(home: Path, sweep_id: str) -> list[Job]:
    return Registry.sqlite(home / "registry.db").list_jobs(sweep_id)


class FakeRun:
    """Stands in for cli.run: keeps each job's command, answers with the next outcome."""

    def __init__(self, *outcomes: int | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[str]] = []

    def __call__(
        self,
        experiment: Path,
        overrides: Sequence[str],
        on_step: Callable[[object], None] | None = None,
    ) -> int:
        self.calls.append([str(experiment), *overrides])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_a_worker_runs_the_jobs_in_order(
    home: Path,
    tmp_path: Path,
    clean: EnvInfo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sweep_id = submit(tmp_path, clean, [0, 1, 2])
    fake = FakeRun(0, 0, 0)
    monkeypatch.setattr(cli, "run", fake)
    assert main(["worker"]) == 0
    assert fake.calls == [[str(HELLO), f"seed={seed}"] for seed in (0, 1, 2)]
    finished = jobs(home, sweep_id)
    assert [job.state for job in finished] == ["done", "done", "done"]
    assert [(job.attempts, job.lease_expires_at) for job in finished] == [(1, None)] * 3
    assert len({job.lease_owner for job in finished}) == 1  # one worker ran them all
    out = capsys.readouterr().out
    assert f"job:           {sweep_id}-3 done\n" in out
    assert "worker:        nothing left to lease in the 12.0 h left (jobs run: 3 done)\n" in out


def test_each_outcome_finishes_its_job(
    home: Path,
    tmp_path: Path,
    clean: EnvInfo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sweep_id = submit(tmp_path, clean, [0, 1, 2])
    halted, cannot = cli.EXIT_HALTED, cli.UsageError("the warm start isn't done")
    monkeypatch.setattr(cli, "run", FakeRun(halted, cannot, 0))
    assert main(["worker"]) == 0  # the worker carries on after a failed job
    assert [job.state for job in jobs(home, sweep_id)] == ["failed", "failed", "done"]
    out, err = capsys.readouterr()
    assert "(jobs run: 2 failed, 1 done)\n" in out
    assert "scalelab: error: the warm start isn't done\n" in err


def test_a_stopped_run_gives_its_job_back_and_stops_the_worker(
    home: Path, tmp_path: Path, clean: EnvInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep_id = submit(tmp_path, clean, [0, 1])
    fake = FakeRun(cli.EXIT_INTERRUPTED)
    monkeypatch.setattr(cli, "run", fake)
    assert main(["worker"]) == 130
    first, second = jobs(home, sweep_id)
    assert (first.state, first.lease_owner, first.lease_expires_at) == ("queued", None, None)
    assert (first.attempts, second.state, second.attempts) == (1, "queued", 0)
    assert len(fake.calls) == 1


def test_a_crash_fails_its_job_and_stops_the_worker(
    home: Path, tmp_path: Path, clean: EnvInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep_id = submit(tmp_path, clean, [0, 1])
    monkeypatch.setattr(cli, "run", FakeRun(RuntimeError("CUDA out of memory")))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        main(["worker"])
    assert [job.state for job in jobs(home, sweep_id)] == ["failed", "queued"]


@pytest.mark.parametrize(("every", "beats"), [(60.0, 1), (0.0, 4)])
def test_a_running_job_heartbeats_every_heartbeat_s(
    home: Path,
    tmp_path: Path,
    clean: EnvInfo,
    monkeypatch: pytest.MonkeyPatch,
    every: float,
    beats: int,
) -> None:
    sweep_id = submit(tmp_path, clean, [0])
    heartbeat = Registry.heartbeat
    calls: list[str] = []
    states: list[str] = []

    def counted(registry: Registry, job_id: str, worker: str, ttl_s: float) -> None:
        calls.append(job_id)
        heartbeat(registry, job_id, worker, ttl_s)

    def run(
        experiment: Path, overrides: Sequence[str], on_step: Callable[[object], None] | None = None
    ) -> int:
        assert on_step is not None
        states.append(jobs(home, sweep_id)[0].state)
        for _ in range(3):  # three training steps
            on_step(None)
        return 0

    monkeypatch.setattr(Registry, "heartbeat", counted)
    monkeypatch.setattr(cli, "HEARTBEAT_S", every)
    monkeypatch.setattr(cli, "run", run)
    assert main(["worker"]) == 0
    assert states == ["running"]  # the first heartbeat comes before the run starts
    assert len(calls) == beats  # that one, then one a step when every step is due


def test_only_the_first_job_may_outlast_the_time_left(
    home: Path,
    tmp_path: Path,
    clean: EnvInfo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sweep_id = submit(tmp_path, clean, [0, 1])
    monkeypatch.setattr(cli, "run", FakeRun(0))
    assert main(["worker", "--session-hours", "1e-9"]) == 0  # no job fits 3.6 µs
    assert [job.state for job in jobs(home, sweep_id)] == ["done", "queued"]
    assert "in the 0.0 h left (jobs run: 1 done)\n" in capsys.readouterr().out


def test_a_local_registry_starts_from_home_and_is_copied_back(
    home: Path, tmp_path: Path, clean: EnvInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep_id = submit(tmp_path, clean, [0])
    live = tmp_path / "vm" / "registry.db"
    live.parent.mkdir()
    Registry.sqlite(live)  # a stale copy from earlier in the session, without the sweep
    fake = FakeRun(0)
    monkeypatch.setattr(cli, "run", fake)
    assert main(["worker", f"runtime.local_registry={live}"]) == 0
    assert fake.calls == [[str(HELLO), "seed=0", f"runtime.local_registry={live}"]]
    assert [job.state for job in jobs(home, sweep_id)] == ["done"]  # home has it too


def test_a_worker_refuses_what_it_cannot_do(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["worker", "algo=rloo"]) == 2  # a config change would change the run ids
    assert "a worker takes runtime settings only" in capsys.readouterr().err
    env = capture_env()
    dirty = env.model_copy(update={"git": env.git.model_copy(update={"dirty": True})})
    monkeypatch.setattr(cli, "capture_env", lambda: dirty)
    assert main(["worker"]) == 2
    assert "commit your changes first" in capsys.readouterr().err
    assert not (home / "registry.db").exists()


def test_a_worker_runs_real_jobs_in_process(
    home: Path, tmp_path: Path, clean: EnvInfo, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep_id = submit(tmp_path, clean, [0, 1])
    assert main(["worker"]) == 0
    finished = jobs(home, sweep_id)
    assert [job.state for job in finished] == ["done", "done"]
    runs = {run.run_id: run.status for run in Registry.sqlite(home / "registry.db").list_runs()}
    assert [runs[job.run_id] for job in finished] == ["done", "done"]
    assert capsys.readouterr().out.count("\nstarted\n") == 2
    assert main(["worker"]) == 0  # nothing left: the second worker runs nothing
    assert "(jobs run: none)\n" in capsys.readouterr().out
