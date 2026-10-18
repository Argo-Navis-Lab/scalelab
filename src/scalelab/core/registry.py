"""Run registry: one database row per experiment and per run."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    insert,
    inspect,
    select,
    update,
)

from scalelab.core import hashing
from scalelab.core.config import ExperimentConfig, StrictModel
from scalelab.core.envcapture import EnvInfo

RunStatus = Literal["running", "done", "failed"]

# --- the schema ------------------------------------------------------------------

metadata = MetaData()

experiments = Table(
    "experiments",
    metadata,
    Column("config_hash", String(16), primary_key=True),
    Column("name", String, nullable=False),
    Column("config_json", Text, nullable=False),  # exactly the text that was hashed
    Column("created_at", DateTime(timezone=True), nullable=False),
)

runs = Table(
    "runs",
    metadata,
    Column("run_id", String, primary_key=True),
    Column(
        "config_hash",
        String(16),
        ForeignKey("experiments.config_hash"),
        nullable=False,
        index=True,
    ),
    Column("seed", Integer, nullable=False),
    Column("git_sha", String(40), nullable=False),
    Column("git_dirty", Boolean, nullable=False),
    Column("status", String, nullable=False),
    Column("executor", String, nullable=False),
    Column("hardware", String, nullable=False),
    Column("deterministic", Boolean, nullable=False, default=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("ended_at", DateTime(timezone=True)),  # empty until the run finishes
    Column("env_json", Text, nullable=False),
)


class RunRecord(StrictModel):
    """One row of the runs table, as a typed Python object."""

    run_id: str
    config_hash: str
    seed: int
    git_sha: str
    git_dirty: bool
    status: RunStatus
    executor: str
    hardware: str
    deterministic: bool
    started_at: datetime
    ended_at: datetime | None
    env_json: str


class OutdatedRegistryError(Exception):
    """The registry file was created by an older ScaleLab whose tables had fewer columns."""


class RunExistsError(Exception):
    """This exact run (same config, seed and code version) is already in the registry."""


# --- helpers ---------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite hands datetimes back without a timezone. We only ever store UTC, so say so."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _hardware_label(env: EnvInfo) -> str:
    gpus = env.hardware.gpus
    if not gpus:
        return "cpu"
    return gpus[0].name if len(gpus) == 1 else f"{len(gpus)}x {gpus[0].name}"


def _to_record(row) -> RunRecord:
    """Turn a database row into a RunRecord, restoring the UTC timezone on timestamps."""
    data = dict(row._mapping)
    data["started_at"] = _as_utc(data["started_at"])
    data["ended_at"] = _as_utc(data["ended_at"])
    return RunRecord.model_validate(data)


def _check_schema(engine) -> None:
    """create_all never changes an existing table, so catch out-of-date tables right away."""
    inspector = inspect(engine)
    for table in metadata.sorted_tables:
        found = {column["name"] for column in inspector.get_columns(table.name)}
        missing = [column.name for column in table.columns if column.name not in found]
        if missing:
            raise OutdatedRegistryError(
                f"the registry's {table.name!r} table is missing the columns {missing}: "
                f"it was created by an older version of ScaleLab ({engine.url.database})"
            )


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    """SQLite ignores foreign keys unless each new connection switches them on."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# --- the registry ------------------------------------------------------------------


class Registry:
    """Reads and writes experiment and run rows. Talks SQL only through SQLAlchemy Core."""

    def __init__(self, url: str) -> None:
        self.engine = create_engine(url)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
        metadata.create_all(self.engine)  # creates missing tables; leaves existing ones alone
        _check_schema(self.engine)

    @classmethod
    def sqlite(cls, path: str | Path) -> "Registry":
        """Open (or create) a registry stored in one SQLite file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(f"sqlite:///{path}")

    def add_run(
        self,
        config: ExperimentConfig,
        seed: int,
        env: EnvInfo,
        executor: str = "local",
        deterministic: bool = False,
    ) -> str:
        """Record a new run as 'running' (and its experiment, if new). Returns the run_id."""
        run_id = hashing.run_id(config, seed, env.git.sha)
        config_hash = hashing.config_hash(config)
        now = _now()
        with self.engine.begin() as conn:  # one transaction: all of it is saved, or none
            if conn.execute(select(runs.c.run_id).where(runs.c.run_id == run_id)).first():
                raise RunExistsError(f"run {run_id} is already in the registry")
            # Fine for one worker at a time (Section 4.6, phase 1).
            known = conn.execute(
                select(experiments.c.config_hash).where(experiments.c.config_hash == config_hash)
            ).first()
            if not known:
                conn.execute(
                    insert(experiments).values(
                        config_hash=config_hash,
                        name=config.name,
                        config_json=hashing.canonical_json(config),
                        created_at=now,
                    )
                )
            conn.execute(
                insert(runs).values(
                    run_id=run_id,
                    config_hash=config_hash,
                    seed=seed,
                    git_sha=env.git.sha,
                    git_dirty=env.git.dirty,
                    status="running",
                    executor=executor,
                    hardware=_hardware_label(env),
                    deterministic=deterministic,
                    started_at=now,
                    ended_at=None,
                    env_json=env.model_dump_json(),
                )
            )
        return run_id

    def finish_run(self, run_id: str, status: RunStatus) -> None:
        """Mark a running run as 'done' or 'failed' and stamp its end time."""
        if status not in ("done", "failed"):
            raise ValueError(f"a run can only finish as 'done' or 'failed', not {status!r}")
        with self.engine.begin() as conn:
            result = conn.execute(
                update(runs)
                .where(runs.c.run_id == run_id, runs.c.status == "running")
                .values(status=status, ended_at=_now())
            )
            if result.rowcount == 0:
                raise ValueError(f"no running run with id {run_id}")

    def reopen_run(self, run_id: str) -> None:
        """Mark an interrupted or failed run as running again, because it is being resumed."""
        with self.engine.begin() as conn:
            result = conn.execute(
                update(runs)
                .where(runs.c.run_id == run_id, runs.c.status != "done")
                .values(status="running", ended_at=None)
            )
            if result.rowcount == 0:
                raise ValueError(f"no unfinished run with id {run_id}")

    def get_run(self, run_id: str) -> RunRecord | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(runs).where(runs.c.run_id == run_id)).first()
        return None if row is None else _to_record(row)

    def list_runs(self, config_hash: str | None = None) -> list[RunRecord]:
        """All runs, or only the runs (seeds) of one experiment, oldest first."""
        query = select(runs).order_by(runs.c.started_at, runs.c.run_id)
        if config_hash is not None:
            query = query.where(runs.c.config_hash == config_hash)
        with self.engine.connect() as conn:
            return [_to_record(row) for row in conn.execute(query)]

    def experiment_json(self, config_hash: str) -> str | None:
        """The canonical JSON stored for an experiment, or None if it isn't registered."""
        with self.engine.connect() as conn:
            return conn.execute(
                select(experiments.c.config_json).where(experiments.c.config_hash == config_hash)
            ).scalar()
