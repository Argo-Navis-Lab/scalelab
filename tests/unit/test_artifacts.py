"""Content-addressed artifact store tests."""

import hashlib

import pytest

from scalelab.core.artifacts import (
    CorruptArtifactError,
    artifact_path,
    get_artifact,
    put_artifact,
)


def test_put_then_get_gives_the_same_bytes(tmp_path):
    sha256 = put_artifact(b"hello weights", tmp_path)
    assert get_artifact(sha256, tmp_path) == b"hello weights"


def test_name_is_the_sha256_of_the_contents(tmp_path):
    sha256 = put_artifact(b"hello weights", tmp_path)
    assert sha256 == hashlib.sha256(b"hello weights").hexdigest()
    assert artifact_path(sha256, tmp_path) == tmp_path / "artifacts" / sha256[:2] / sha256


def test_same_bytes_are_stored_once(tmp_path):
    first = put_artifact(b"same", tmp_path)
    second = put_artifact(b"same", tmp_path)
    assert first == second
    assert len([p for p in (tmp_path / "artifacts").rglob("*") if p.is_file()]) == 1


def test_different_bytes_get_different_names(tmp_path):
    assert put_artifact(b"one", tmp_path) != put_artifact(b"two", tmp_path)


def test_corruption_is_detected(tmp_path):
    sha256 = put_artifact(b"good bytes", tmp_path)
    artifact_path(sha256, tmp_path).write_bytes(b"bad bytes")  # e.g. a damaged sync
    with pytest.raises(CorruptArtifactError, match="corrupted"):
        get_artifact(sha256, tmp_path)


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_artifact(hashlib.sha256(b"never stored").hexdigest(), tmp_path)


@pytest.mark.parametrize("bad", ["", "abc", "A" * 64, "../../etc/passwd"])
def test_names_must_be_sha256_hex(tmp_path, bad):
    with pytest.raises(ValueError, match="not a SHA-256"):
        artifact_path(bad, tmp_path)


def test_default_home_is_scalelab_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path))
    sha256 = put_artifact(b"x")
    assert (tmp_path.resolve() / "artifacts" / sha256[:2] / sha256).is_file()
