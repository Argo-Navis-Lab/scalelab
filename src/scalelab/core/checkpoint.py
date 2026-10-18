"""Checkpoints: save a run's complete state, and load it back exactly."""

import json
import random
import signal
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load as tensors_from_bytes
from safetensors.torch import save as tensors_to_bytes

from scalelab.core.artifacts import artifact_path, get_artifact, put_artifact
from scalelab.core.storage import atomic_write_json, scalelab_home

FORMAT_VERSION = 1  # bump this if the manifest layout ever changes


# --- nested state <-> JSON plus a flat bag of tensors ----------------------------------


def _encode(obj: Any, tensors: dict[str, torch.Tensor]) -> Any:
    """Make `obj` JSON-safe. Each tensor moves into `tensors`, leaving a small marker behind."""
    if isinstance(obj, torch.Tensor):
        key = str(len(tensors))
        tensors[key] = obj.detach().cpu().clone().contiguous()  # a private copy, frozen now
        return {"__tensor__": key}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):  # a single NumPy number, like np.int64(3)
        return obj.item()
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(item, tensors) for item in obj]}
    if isinstance(obj, list):
        return [_encode(item, tensors) for item in obj]
    if isinstance(obj, dict):
        if all(isinstance(key, str) for key in obj):
            return {key: _encode(value, tensors) for key, value in obj.items()}
        # JSON keys must be text, so dicts with other keys (like ints) become a list of pairs.
        pairs = [[_encode(key, tensors), _encode(value, tensors)] for key, value in obj.items()]
        return {"__dict__": pairs}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"can't put a {type(obj).__name__} in a checkpoint")


def _decode(obj: Any, tensors: dict[str, torch.Tensor]) -> Any:
    """Undo `_encode`: rebuild tuples, int-keyed dicts, arrays and tensors."""
    if isinstance(obj, list):
        return [_decode(item, tensors) for item in obj]
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            return tensors[obj["__tensor__"]]
        if "__ndarray__" in obj:
            return np.array(obj["__ndarray__"], dtype=obj["dtype"])
        if "__tuple__" in obj:
            return tuple(_decode(item, tensors) for item in obj["__tuple__"])
        if "__dict__" in obj:
            pairs = obj["__dict__"]
            return {_decode(key, tensors): _decode(value, tensors) for key, value in pairs}
        return {key: _decode(value, tensors) for key, value in obj.items()}
    return obj


# --- random generators -----------------------------------------------------------------


def capture_rng() -> dict[str, Any]:
    """The exact position of every global random generator."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(legacy=False),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    """Put every global random generator back where `capture_rng` found it."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# --- saving and loading ------------------------------------------------------------------


def save_checkpoint(
    directory: str | Path, step: int, state: dict[str, Any], home: Path | None = None
) -> Path:
    """Save `state` as of `step`. Returns the path of the new manifest.

    The tensors go into the artifact store as one safetensors file. Everything else goes
    into a small JSON manifest that names that file by its hash.
    """
    tensors: dict[str, torch.Tensor] = {}
    tree = _encode(state, tensors)
    blob = put_artifact(tensors_to_bytes(tensors), home)  # first the tensors...
    manifest = {
        "format": FORMAT_VERSION,
        "step": step,
        "saved_at": datetime.now(UTC).isoformat(),
        "tensors_sha256": blob,
        "state": tree,
    }
    # ...then the manifest, so a manifest never points at tensors that aren't there yet.
    return atomic_write_json(Path(directory) / f"step-{step:09d}.json", manifest)


def load_checkpoint(manifest_path: str | Path, home: Path | None = None) -> tuple[int, Any]:
    """Load a checkpoint. Returns (step, state), with every tensor back on the CPU."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["format"] != FORMAT_VERSION:
        raise ValueError(f"unknown checkpoint format {manifest['format']} in {manifest_path}")
    tensors = tensors_from_bytes(get_artifact(manifest["tensors_sha256"], home))
    return manifest["step"], _decode(manifest["state"], tensors)


def latest_checkpoint(directory: str | Path) -> Path | None:
    """The manifest with the highest step in `directory`, or None if there isn't one."""
    manifests = sorted(Path(directory).glob("step-*.json"))  # zero-padded, so text order works
    return manifests[-1] if manifests else None


# --- keeping only the newest checkpoints --------------------------------------------------


def _tensors_of(manifest: Path) -> str:
    return json.loads(manifest.read_text())["tensors_sha256"]


def prune_checkpoints(directory: str | Path, keep: int, home: Path | None = None) -> None:
    """Delete all but the newest `keep` checkpoints in `directory`, then delete their tensor
    files, unless some other checkpoint under `home` still uses the very same file."""
    if keep < 1:
        raise ValueError(f"keep must be at least 1, got {keep}")
    home = (home or scalelab_home()).resolve()
    directory = Path(directory).resolve()
    if not directory.is_relative_to(home):
        raise ValueError(f"checkpoint folder {directory} must be inside {home} to clean up safely")
    old = sorted(directory.glob("step-*.json"))[:-keep]
    if not old:
        return
    candidates = {_tensors_of(manifest) for manifest in old}
    for manifest in old:  # manifests first: nothing may ever point at a deleted file
        manifest.unlink()
    # One worker at a time for now (Section 4.6, phase 1), so nobody saves while we look.
    still_used = {_tensors_of(manifest) for manifest in home.rglob("step-*.json")}
    for sha256 in candidates - still_used:
        artifact_path(sha256, home).unlink(missing_ok=True)


# --- when to save ------------------------------------------------------------------------


class CheckpointPolicy:
    """When to save, and how many checkpoints to keep.

    A save is due every `every_steps` steps or once `every_seconds` have passed since the
    last save, whichever comes first, and at once after a stop has been requested.
    Use a fresh policy for every run.
    """

    def __init__(
        self,
        every_steps: int | None = None,
        every_seconds: float | None = 15 * 60,
        keep: int = 2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if keep < 1:
            raise ValueError(f"keep must be at least 1, got {keep}")
        self.every_steps = every_steps
        self.every_seconds = every_seconds
        self.keep = keep
        self.clock = clock
        self.last_save = clock()
        self.stop_requested = False

    def due(self, step: int) -> bool:
        if self.stop_requested:
            return True
        if self.every_steps is not None and step % self.every_steps == 0:
            return True
        if self.every_seconds is not None and self.clock() - self.last_save >= self.every_seconds:
            return True
        return False

    def saved(self) -> None:
        """Call right after each save, to restart the clock."""
        self.last_save = self.clock()

    def request_stop(self, *_signal_details: Any) -> None:
        """Save at the end of the current step, then stop. Also works as a signal handler."""
        self.stop_requested = True

    def stop_on_signals(self) -> None:
        """Make SIGTERM and Ctrl+C (SIGINT) mean "save and stop" instead of "die now"."""
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
