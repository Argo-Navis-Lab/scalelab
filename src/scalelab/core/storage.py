"""Storage: where ScaleLab keeps files, and how it writes them safely."""

import json
import os
from pathlib import Path
from typing import Any

HOME_ENV_VAR = "SCALELAB_HOME"


def scalelab_home(default: str | Path = "runs") -> Path:
    """The root folder for everything a run writes.

    $SCALELAB_HOME wins if set (on Colab, a Google Drive folder); otherwise `default`.
    Always returned as an absolute path, so what we record is never ambiguous.
    """
    root = os.environ.get(HOME_ENV_VAR) or default
    return Path(root).expanduser().resolve()


def registry_path(home: Path | None = None) -> Path:
    """Where the SQLite registry lives: <home>/registry.db."""
    return (home or scalelab_home()) / "registry.db"


def _fsync_dir(directory: Path) -> None:
    """Make a rename inside `directory` survive a power cut, where the filesystem allows it."""
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # some filesystems can't sync a folder; the file's own data is already safe


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write `data` to `path` so that `path` always holds either the old or the new
    contents, never half of the new ones: write <path>.tmp, flush, fsync, rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()  # Python's buffer -> the operating system
            os.fsync(f.fileno())  # the operating system's cache -> the disk
        os.replace(tmp, path)  # the switch: one step, old file or new file, nothing between
    except BaseException:
        tmp.unlink(missing_ok=True)  # leave no half-written leftovers behind
        raise
    _fsync_dir(path.parent)
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, data: Any) -> Path:
    """Pretty, sorted JSON (easy to read and to diff), written atomically."""
    return atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")
