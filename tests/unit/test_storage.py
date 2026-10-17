"""Storage tests: the SCALELAB_HOME root and atomic writes."""

import json
import os
from pathlib import Path

import pytest

from scalelab.core.storage import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    registry_path,
    scalelab_home,
)

# --- where files live ----------------------------------------------------------------


def test_home_defaults_to_runs_in_the_current_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("SCALELAB_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert scalelab_home() == tmp_path.resolve() / "runs"


def test_home_follows_the_environment_variable(monkeypatch, tmp_path):
    drive = tmp_path / "drive" / "MyDrive" / "scalelab"  # stands in for a Colab Drive folder
    monkeypatch.setenv("SCALELAB_HOME", str(drive))
    assert scalelab_home() == drive.resolve()


def test_home_expands_tilde(monkeypatch):
    monkeypatch.setenv("SCALELAB_HOME", "~/scalelab_home")
    assert scalelab_home() == Path.home().resolve() / "scalelab_home"


def test_empty_environment_variable_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("SCALELAB_HOME", "")
    monkeypatch.chdir(tmp_path)
    assert scalelab_home() == tmp_path.resolve() / "runs"


def test_home_is_always_absolute(monkeypatch, tmp_path):
    monkeypatch.delenv("SCALELAB_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert scalelab_home(default="results").is_absolute()


def test_registry_lives_in_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path))
    assert registry_path() == tmp_path.resolve() / "registry.db"
    assert registry_path(tmp_path / "other") == tmp_path / "other" / "registry.db"


# --- atomic writes ------------------------------------------------------------------


def test_write_creates_folders_and_file(tmp_path):
    path = tmp_path / "a" / "b" / "data.bin"
    assert atomic_write_bytes(path, b"\x00\x01\x02") == path
    assert path.read_bytes() == b"\x00\x01\x02"


def test_write_replaces_an_existing_file(tmp_path):
    path = tmp_path / "note.txt"
    atomic_write_text(path, "old")
    atomic_write_text(path, "new")
    assert path.read_text() == "new"


def test_no_tmp_file_is_left_behind(tmp_path):
    atomic_write_text(tmp_path / "note.txt", "hello")
    assert [p.name for p in tmp_path.iterdir()] == ["note.txt"]


def test_text_is_utf8(tmp_path):
    path = atomic_write_text(tmp_path / "note.txt", "π ≈ 3.14159")
    assert path.read_bytes().decode("utf-8") == "π ≈ 3.14159"


def test_json_is_sorted_readable_and_ends_with_a_newline(tmp_path):
    path = atomic_write_json(tmp_path / "manifest.json", {"step": 10, "loss": 0.5})
    text = path.read_text()
    assert text == '{\n  "loss": 0.5,\n  "step": 10\n}\n'
    assert json.loads(text) == {"loss": 0.5, "step": 10}


def test_fsync_happens_before_rename(monkeypatch, tmp_path):
    calls = []
    real_fsync, real_replace = os.fsync, os.replace

    def spy_fsync(fd):
        calls.append("fsync")
        real_fsync(fd)

    def spy_replace(src, dst):
        calls.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    atomic_write_text(tmp_path / "note.txt", "hello")
    assert calls[:2] == ["fsync", "replace"]  # data safely on disk first, then the switch


@pytest.mark.parametrize("failing_step", ["fsync", "replace"])
def test_a_failed_write_never_damages_the_old_file(monkeypatch, tmp_path, failing_step):
    path = atomic_write_text(tmp_path / "checkpoint.json", "the good old contents")

    def crash(*args):
        raise OSError(f"simulated crash during {failing_step}")

    monkeypatch.setattr(os, failing_step, crash)
    with pytest.raises(OSError, match="simulated crash"):
        atomic_write_text(path, "the new contents")

    assert path.read_text() == "the good old contents"
    assert [p.name for p in tmp_path.iterdir()] == ["checkpoint.json"]  # no .tmp left


def test_a_stale_tmp_from_a_killed_process_is_harmless(tmp_path):
    path = tmp_path / "checkpoint.json"
    (tmp_path / "checkpoint.json.tmp").write_text('{"half": ')  # what a hard kill leaves
    atomic_write_json(path, {"step": 5})
    assert json.loads(path.read_text()) == {"step": 5}
    assert [p.name for p in tmp_path.iterdir()] == ["checkpoint.json"]
