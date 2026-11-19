"""A dataset of finished runs: their registry rows, gradings and metrics, copied into a new
folder that works as a home of its own, with nothing in it that names the machine they ran on
or the person who ran them."""

import hashlib
import re
import shutil
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scalelab.analysis.datacard import write_card
from scalelab.core.registry import Registry
from scalelab.core.storage import atomic_write_json, registry_path
from scalelab.rl.gradings import gradings_folder
from scalelab.telemetry.sink import metrics_folder

MANIFEST = "manifest.json"

# The start of an absolute path in the places runs happen: macOS and Linux homes, Colab,
# vast.ai, temporary folders, Windows drives. Relative paths (configs/...) are fine.
PATH_PATTERN = re.compile(
    r"(?<![\w.])(/Users/|/home/|/root/|/content/|/workspace/|/tmp/|/private/|/var/|[A-Za-z]:\\)"
)


class ExportError(Exception):
    """The export can't go ahead, or found something it must not ship."""


@dataclass(frozen=True)
class Exported:
    runs: int
    files: int
    size: int  # bytes, the manifest left out


def machine_names() -> tuple[str, ...]:
    """What names this machine and its user: the host name and the home folder."""
    names = {socket.gethostname(), str(Path.home())}
    return tuple(sorted(name for name in names if len(name) >= 4 and name != "localhost"))


def leaks(text: str, names: Sequence[str]) -> list[str]:
    """Every absolute path start and machine name in `text`, in order found."""
    found = [match.group(0) for match in PATH_PATTERN.finditer(text)]
    return found + [name for name in names if name in text]


def file_leaks(path: Path, names: Sequence[str]) -> list[str]:
    """leaks() in a file: a Parquet file's text columns, any other file's bytes as text
    (compressed Parquet pages would hide a path from a byte search)."""
    if path.suffix != ".parquet":
        return leaks(path.read_bytes().decode("utf-8", errors="replace"), names)
    table = pq.read_table(path)
    found: list[str] = []
    for column in table.column_names:
        values = table.column(column)
        if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
            for value in values.unique().to_pylist():
                found += leaks(str(value), names)
    return found


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_runs(home: Path, registry: Registry, run_ids: Sequence[str], out: Path) -> Exported:
    """Copy runs from `registry` (home's, or a copy of it) and their files under `home` into
    `out`, a new or empty folder: registry.db, gradings/, metrics/, runs.csv, runs.parquet,
    README.md and a manifest of every file's sha256. Then search every file for leaks(); finding
    one deletes `out` again and raises ExportError."""
    if out.exists() and any(out.iterdir()):
        raise ExportError(f"{out} is not empty: export into a new folder")
    out.mkdir(parents=True, exist_ok=True)
    try:
        target = Registry.sqlite(registry_path(out))
        registry.export_runs(run_ids, target)
        for run_id in run_ids:
            for folder in (gradings_folder, metrics_folder):
                if folder(home, run_id).is_dir():
                    shutil.copytree(folder(home, run_id), folder(out, run_id))
        write_card(
            out, target
        )  # m4s18: runs.csv and README.md (m4s20: + runs.parquet), searched like the rest
        target.engine.dispose()  # closes the file before it is searched and hashed
        files = sorted(path for path in out.rglob("*") if path.is_file())
        names = machine_names()
        found = [
            f"{path.relative_to(out)}: {leak}" for path in files for leak in file_leaks(path, names)
        ]
        if found:
            raise ExportError("machine details in the export:\n  " + "\n  ".join(found[:20]))
    except BaseException:
        shutil.rmtree(out)  # half an export is worse than none
        raise
    manifest = {
        "runs": sorted(run_ids),
        "files": {path.relative_to(out).as_posix(): sha256(path) for path in files},
    }
    atomic_write_json(out / MANIFEST, manifest)
    return Exported(runs=len(run_ids), files=len(files), size=sum(p.stat().st_size for p in files))
