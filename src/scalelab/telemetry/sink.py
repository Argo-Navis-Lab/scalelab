"""The metrics sink: every number the collectors measure, as rows of (run_id, step,
wall_time, metric, layer, value), kept in memory and written to Parquet files under
<home>/metrics/<run_id>/."""

import io
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scalelab.core.storage import atomic_write_bytes
from scalelab.telemetry.hooks import CheckpointEvent, Hook

SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("step", pa.int64()),
        ("wall_time", pa.float64()),  # when it was recorded, in seconds since 1970
        ("metric", pa.string()),
        ("layer", pa.int64()),  # the layer a per-layer metric is about; null for the others
        ("value", pa.float64()),
    ]
)


@dataclass(frozen=True)
class MetricRow:
    """One measurement."""

    run_id: str
    step: int
    wall_time: float
    metric: str
    layer: int | None
    value: float


def metrics_folder(home: Path, run_id: str) -> Path:
    return home / "metrics" / run_id


class MetricsSink(Hook):
    """One run's metric rows, kept in memory and written to Parquet, a new part file each
    time: once `every_seconds` have passed since the last write (checked as rows arrive),
    and at every checkpoint, so that whatever a checkpoint keeps, its metrics are on disk
    too. A resumed run carries on numbering its parts."""

    def __init__(
        self,
        home: Path,
        run_id: str,
        every_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_id = run_id
        self.folder = metrics_folder(home, run_id)
        self.every_seconds = every_seconds
        self.clock = clock
        self.rows: list[MetricRow] = []
        self.parts = len(list(self.folder.glob("part-*.parquet")))  # by any earlier session
        self.last_write = clock()
        self.latest: dict[str, float] = {}  # m2s35: the newest step's metrics with no layer
        self.latest_step = 0

    def record(self, step: int, metric: str, value: float, layer: int | None = None) -> None:
        """Keep one measurement, and write the rows kept so far if it is time to."""
        self.rows.append(MetricRow(self.run_id, step, time.time(), metric, layer, float(value)))
        if layer is None:  # m2s35: kept for the detectors too
            if step != self.latest_step:
                self.latest, self.latest_step = {}, step
            self.latest[metric] = float(value)
        if self.clock() - self.last_write >= self.every_seconds:
            self.flush()

    def values(self, step: int) -> dict[str, float]:
        """The metrics with no layer recorded at `step`, if it is the newest step recorded
        (what the detectors judge, m2s35); none otherwise. Unlike the rows, a write keeps them."""
        return dict(self.latest) if step == self.latest_step else {}

    def flush(self) -> None:
        """Write the rows kept so far as the next part file (no file if there are none)."""
        self.last_write = self.clock()
        if not self.rows:
            return
        columns = {name: [getattr(row, name) for row in self.rows] for name in SCHEMA.names}
        buffer = io.BytesIO()
        pq.write_table(pa.table(columns, schema=SCHEMA), buffer)
        atomic_write_bytes(self.folder / f"part-{self.parts:06d}.parquet", buffer.getvalue())
        self.parts += 1
        self.rows = []

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        self.flush()


def read_metrics(home: Path, run_id: str) -> list[MetricRow]:
    """Every row a run's sink wrote, in step order. A resumed run redoes the steps after its
    last checkpoint, and may have written some of them before it stopped: for each (step,
    metric, layer) the newest row is the one kept."""
    newest: dict[tuple[int, str, int | None], MetricRow] = {}
    for part in sorted(metrics_folder(home, run_id).glob("part-*.parquet")):
        for record in pq.read_table(part).to_pylist():
            row = MetricRow(**record)
            newest[row.step, row.metric, row.layer] = row
    return sorted(newest.values(), key=lambda row: row.step)
