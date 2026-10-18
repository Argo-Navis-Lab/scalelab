"""Environment capture tests."""

import hashlib
import platform
import re
import subprocess
from pathlib import Path

import pytest

from scalelab.core.envcapture import (
    EnvInfo,
    capture_env,
    capture_git,
    parse_nvidia_smi,
    repo_root,
)
from scalelab.core.hashing import run_id

REPO_ROOT = Path(__file__).resolve().parents[2]


def git(repo: Path, *args: str) -> str:
    """Run a git command inside `repo` and return what it prints."""
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A tiny, brand-new git repo with one commit, in a temporary folder."""
    git(tmp_path, "init", "--quiet")
    # A throwaway identity for this repo only, so `commit` works anywhere (even on CI).
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "commit.gpgsign", "false")

    (tmp_path / "train.py").write_text("lr = 0.1\n")
    (tmp_path / "uv.lock").write_text("pretend lockfile\n")
    (tmp_path / ".gitignore").write_text("runs/\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "--quiet", "-m", "first commit")
    return tmp_path


# --- git -----------------------------------------------------------------------


def test_clean_repo(git_repo):
    info = capture_git(git_repo)
    assert info.sha == git(git_repo, "rev-parse", "HEAD")
    assert re.fullmatch(r"[0-9a-f]{40}", info.sha)
    assert info.dirty is False
    assert info.diff_sha256 is None


def test_edited_file_makes_repo_dirty(git_repo):
    (git_repo / "train.py").write_text("lr = 0.2\n")
    info = capture_git(git_repo)
    assert info.dirty is True
    assert info.diff_sha256
    assert re.fullmatch(r"[0-9a-f]{64}", info.diff_sha256)


def test_new_untracked_file_makes_repo_dirty(git_repo):
    (git_repo / "notes.py").write_text("print('hi')\n")
    assert capture_git(git_repo).dirty is True


def test_ignored_files_do_not_make_repo_dirty(git_repo):
    (git_repo / "runs").mkdir()
    (git_repo / "runs" / "metrics.parquet").write_bytes(b"\x00\x01")
    assert capture_git(git_repo).dirty is False


def test_diff_hash_identifies_the_change(git_repo):
    def diff_hash_after_writing(text: str) -> str | None:
        (git_repo / "train.py").write_text(text)
        return capture_git(git_repo).diff_sha256

    first = diff_hash_after_writing("lr = 0.2\n")
    other = diff_hash_after_writing("lr = 0.3\n")
    again = diff_hash_after_writing("lr = 0.2\n")
    assert first != other  # different edits, different hash
    assert first == again  # the same edit, the same hash


def test_diff_hash_sees_inside_untracked_files(git_repo):
    (git_repo / "notes.py").write_text("x = 1\n")
    first = capture_git(git_repo).diff_sha256
    (git_repo / "notes.py").write_text("x = 2\n")
    assert capture_git(git_repo).diff_sha256 != first


def test_works_from_a_subfolder(git_repo):
    sub = git_repo / "src" / "deep"
    sub.mkdir(parents=True)
    assert capture_git(sub).sha == capture_git(git_repo).sha


def test_outside_a_git_repo_raises(tmp_path):
    with pytest.raises(RuntimeError, match="git rev-parse --show-toplevel failed"):
        capture_git(tmp_path)


def test_default_repo_is_this_checkout():
    assert repo_root() == Path(git(REPO_ROOT, "rev-parse", "--show-toplevel"))
    assert capture_git().sha == git(REPO_ROOT, "rev-parse", "HEAD")


def test_captured_sha_works_as_run_id_input(config):
    sha = capture_git().sha
    assert run_id(config, seed=0, git_sha=sha).endswith(sha[:8])


# --- software and hardware ---------------------------------------------------------


def test_lockfile_hash_matches_file(git_repo):
    expected = hashlib.sha256((git_repo / "uv.lock").read_bytes()).hexdigest()
    assert capture_env(git_repo).software.uv_lock_sha256 == expected


def test_missing_lockfile_gives_none(git_repo):
    (git_repo / "uv.lock").unlink()
    assert capture_env(git_repo).software.uv_lock_sha256 is None


def test_parse_nvidia_smi_two_gpus():
    text = "Tesla T4, 15360, 550.54.15\nTesla T4, 15360, 550.54.15\n"
    gpus = parse_nvidia_smi(text)
    assert len(gpus) == 2
    assert gpus[0].name == "Tesla T4"
    assert gpus[0].memory_mib == 15360
    assert gpus[0].driver == "550.54.15"


def test_parse_nvidia_smi_unknown_memory():
    (gpu,) = parse_nvidia_smi("NVIDIA L4, [N/A], 550.54.15")
    assert gpu.memory_mib is None


def test_parse_nvidia_smi_no_gpus():
    assert parse_nvidia_smi("") == ()


def test_capture_env_on_this_machine():
    env = capture_env()
    assert env.software.python == platform.python_version()
    assert env.software.packages["scalelab"] is not None
    assert env.software.packages["pydantic"] is not None
    assert env.hardware.cpu_count >= 1
    assert env.hardware.cpu_model != ""


def test_env_survives_a_json_round_trip():
    env = capture_env()
    assert EnvInfo.model_validate_json(env.model_dump_json()) == env
