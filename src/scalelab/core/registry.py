"""Run registry: one database row per experiment and per run, one per eval result and one per
anomaly event."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    RowMapping,
    String,
    Table,
    Text,
    create_engine,
    delete,
    event,
    insert,
    inspect,
    select,
    update,
)
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.pool import ConnectionPoolEntry

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

evals = Table(
    "evals",
    metadata,
    Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
    Column("step", Integer, primary_key=True),
    Column("task", String, primary_key=True),  # the eval set, such as "arithmetic-3"
    Column("metric", String, primary_key=True),  # such as "greedy_pass@1"
    Column("value", Float, nullable=False),
    Column("n", Integer, nullable=False),  # how many problems were graded
    Column("stderr", Float, nullable=False),
)

anomalies = Table(  # m2s35
    "anomalies",
    metadata,
    Column("run_id", String, ForeignKey("runs.run_id"), primary_key=True),
    Column("step", Integer, primary_key=True),  # where the finding started
    Column("detector", String, primary_key=True),  # "spike", "divergence" or "collapse"
    Column("metric", String, primary_key=True),  # what it watched, such as "optim/grad_norm"
    Column("severity", String, nullable=False),  # "WARN", "SNAPSHOT" or "HALT"
    Column("signature_json", Text, nullable=False),  # the numbers it judged by, as JSON
    Column("label", String),  # what really happened, once known (M5's fault library)
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


class EvalRecord(StrictModel):
    """One row of the evals table: one metric of one run on one eval set, at one step."""

    run_id: str
    step: int
    task: str
    metric: str
    value: float
    n: int
    stderr: float


class AnomalyRecord(StrictModel):
    """One row of the anomalies table: one event of a run's online detectors."""

    run_id: str
    step: int
    detector: str
    metric: str
    severity: str
    signature_json: str
    label: str | None = None


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


def _to_record(row: RowMapping) -> RunRecord:
    """Turn a database row into a RunRecord, restoring the UTC timezone on timestamps."""
    data = dict(row)
    data["started_at"] = _as_utc(data["started_at"])
    data["ended_at"] = _as_utc(data["ended_at"])
    return RunRecord.model_validate(data)


def _check_schema(engine: Engine) -> None:
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


def _enable_sqlite_foreign_keys(
    dbapi_connection: DBAPIConnection, _connection_record: ConnectionPoolEntry
) -> None:
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

    def record_eval(self, record: EvalRecord) -> None:
        """Store one eval result, replacing any stored for the same run, step, eval set and
        metric: a resumed run redoes the steps after its last checkpoint, gradings included."""
        same = (
            (evals.c.run_id == record.run_id)
            & (evals.c.step == record.step)
            & (evals.c.task == record.task)
            & (evals.c.metric == record.metric)
        )
        with self.engine.begin() as conn:  # one transaction: never both rows, never neither
            conn.execute(delete(evals).where(same))
            conn.execute(insert(evals).values(**record.model_dump()))

    def list_evals(self, run_id: str) -> list[EvalRecord]:
        """A run's eval results, in order of step, then eval set, then metric."""
        query = (
            select(evals)
            .where(evals.c.run_id == run_id)
            .order_by(evals.c.step, evals.c.task, evals.c.metric)
        )
        with self.engine.connect() as conn:
            return [EvalRecord.model_validate(dict(row)) for row in conn.execute(query).mappings()]

    def record_anomaly(self, record: AnomalyRecord) -> None:
        """Store one anomaly event, replacing any stored for the same run, step, detector and
        metric: a resumed run redoes the steps after its last checkpoint, detectors included."""
        same = (
            (anomalies.c.run_id == record.run_id)
            & (anomalies.c.step == record.step)
            & (anomalies.c.detector == record.detector)
            & (anomalies.c.metric == record.metric)
        )
        with self.engine.begin() as conn:
            conn.execute(delete(anomalies).where(same))
            conn.execute(insert(anomalies).values(**record.model_dump()))

    def list_anomalies(self, run_id: str) -> list[AnomalyRecord]:
        """A run's anomaly events, in order of step, then detector, then metric."""
        query = (
            select(anomalies)
            .where(anomalies.c.run_id == run_id)
            .order_by(anomalies.c.step, anomalies.c.detector, anomalies.c.metric)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings()
            return [AnomalyRecord.model_validate(dict(row)) for row in rows]

    def get_run(self, run_id: str) -> RunRecord | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(runs).where(runs.c.run_id == run_id)).mappings().first()
        return None if row is None else _to_record(row)

    def list_runs(self, config_hash: str | None = None) -> list[RunRecord]:
        """All runs, or only the runs (seeds) of one experiment, oldest first."""
        query = select(runs).order_by(runs.c.started_at, runs.c.run_id)
        if config_hash is not None:
            query = query.where(runs.c.config_hash == config_hash)
        with self.engine.connect() as conn:
            return [_to_record(row) for row in conn.execute(query).mappings()]

    def experiment_json(self, config_hash: str) -> str | None:
        """The canonical JSON stored for an experiment, or None if it isn't registered."""
        with self.engine.connect() as conn:
            return conn.execute(
                select(experiments.c.config_json).where(experiments.c.config_hash == config_hash)
            ).scalar()
