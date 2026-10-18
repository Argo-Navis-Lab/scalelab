"""Content-addressed artifact store: every file is named by the SHA-256 of its own bytes."""

import hashlib
import re
from pathlib import Path

from scalelab.core.storage import atomic_write_bytes, scalelab_home

_SHA256 = re.compile(r"[0-9a-f]{64}")


class CorruptArtifactError(Exception):
    """A stored file's bytes no longer match the hash it is named after."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_path(sha256: str, home: Path | None = None) -> Path:
    """<home>/artifacts/<first 2 characters>/<full hash>."""
    if not _SHA256.fullmatch(sha256):
        raise ValueError(f"not a SHA-256 hex digest: {sha256!r}")
    return (home or scalelab_home()) / "artifacts" / sha256[:2] / sha256


def put_artifact(data: bytes, home: Path | None = None) -> str:
    """Store `data` and return its hash. Storing the same bytes again costs nothing."""
    sha256 = sha256_hex(data)
    path = artifact_path(sha256, home)
    if not path.exists():
        atomic_write_bytes(path, data)
    return sha256


def get_artifact(sha256: str, home: Path | None = None) -> bytes:
    """Read an artifact back, checking that its bytes still match its name."""
    data = artifact_path(sha256, home).read_bytes()
    if sha256_hex(data) != sha256:
        raise CorruptArtifactError(f"artifact {sha256} is corrupted: its contents changed")
    return data
