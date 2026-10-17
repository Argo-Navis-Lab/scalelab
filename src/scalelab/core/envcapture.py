"""Environment capture: the code, libraries and hardware behind a run."""

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from scalelab.core.config import StrictModel

# Where this file lives. In an editable install that is inside your git checkout.
_PACKAGE_DIR = Path(__file__).resolve().parent

# Packages whose versions we record by name. The uv.lock hash pins everything else.
KEY_PACKAGES = ("scalelab", "pydantic", "pyyaml", "numpy", "torch")


class GitInfo(StrictModel):
    sha: str  # full 40-character commit SHA of HEAD
    dirty: bool  # True if anything differs from that commit (edits or new files)
    diff_sha256: str | None = None  # hash of the uncommitted changes; None when clean


class GpuInfo(StrictModel):
    name: str
    memory_mib: int | None
    driver: str


class SoftwareInfo(StrictModel):
    python: str
    os: str
    packages: dict[str, str | None]  # None means "not installed"
    cuda: str | None = None  # CUDA version torch was built with
    cudnn: int | None = None
    uv_lock_sha256: str | None = None


class HardwareInfo(StrictModel):
    cpu_model: str
    cpu_count: int
    gpus: tuple[GpuInfo, ...] = ()


class EnvInfo(StrictModel):
    """Everything we record about where a run happened. Stored, never hashed into run_id."""

    git: GitInfo
    software: SoftwareInfo
    hardware: HardwareInfo


# --- git -----------------------------------------------------------------------


def _git(repo: Path, *args: str) -> bytes:
    """Run one git command in `repo` and return its raw output. Fails loudly."""
    try:
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("git is not installed or not on PATH") from None
    except subprocess.CalledProcessError as err:
        message = err.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {message}") from None
    return result.stdout


def repo_root(start: Path | None = None) -> Path:
    """The top folder of the git checkout containing `start` (default: this package)."""
    out = _git(start or _PACKAGE_DIR, "rev-parse", "--show-toplevel")
    return Path(out.decode().strip())


def _diff_sha256(root: Path) -> str:
    """Hash of everything that differs from HEAD: edited tracked files plus new untracked files."""
    h = hashlib.sha256(_git(root, "diff", "HEAD", "--binary"))
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    for name in sorted(untracked.split(b"\0")):
        if name:
            h.update(name + b"\0" + (root / name.decode()).read_bytes())
    return h.hexdigest()


def capture_git(repo: Path | None = None) -> GitInfo:
    """Commit SHA, dirty flag, and a hash of any uncommitted changes."""
    root = repo_root(repo)
    sha = _git(root, "rev-parse", "HEAD").decode().strip()
    dirty = bool(_git(root, "status", "--porcelain").strip())
    return GitInfo(sha=sha, dirty=dirty, diff_sha256=_diff_sha256(root) if dirty else None)


# --- software --------------------------------------------------------------------


def _version(package: str) -> str | None:
    """Installed version of a package, or None if it isn't installed."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_software(root: Path) -> SoftwareInfo:
    packages = {name: _version(name) for name in KEY_PACKAGES}
    cuda = cudnn = None
    if packages["torch"] is not None:
        import torch  # imported only here: torch is big and not installed yet in M0

        cuda = torch.version.cuda
        cudnn = torch.backends.cudnn.version()
    return SoftwareInfo(
        python=platform.python_version(),
        os=platform.platform(),
        packages=packages,
        cuda=cuda,
        cudnn=cudnn,
        uv_lock_sha256=_file_sha256(root / "uv.lock"),
    )


# --- hardware --------------------------------------------------------------------


def _run(*cmd: str) -> str | None:
    """Run an optional helper program. Returns its output, or None if it is missing or fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _cpu_model() -> str:
    if platform.system() == "Darwin":  # macOS
        name = _run("sysctl", "-n", "machdep.cpu.brand_string")
        if name:
            return name
    cpuinfo = Path("/proc/cpuinfo")  # Linux: Colab, Kaggle, GitHub Actions
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def parse_nvidia_smi(text: str) -> tuple[GpuInfo, ...]:
    """Turn `nvidia-smi --query-gpu=name,memory.total,driver_version` CSV lines into GpuInfo."""
    gpus = []
    for line in text.strip().splitlines():
        name, memory, driver = (part.strip() for part in line.split(","))
        memory_mib = int(memory) if memory.isdigit() else None
        gpus.append(GpuInfo(name=name, memory_mib=memory_mib, driver=driver))
    return tuple(gpus)


def capture_hardware() -> HardwareInfo:
    smi = _run(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    return HardwareInfo(
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count() or 1,
        gpus=parse_nvidia_smi(smi) if smi else (),
    )


# --- everything ------------------------------------------------------------------


def capture_env(repo: Path | None = None) -> EnvInfo:
    """Snapshot the code version, software and hardware. Call once at the start of a run."""
    root = repo_root(repo)
    return EnvInfo(
        git=capture_git(root),
        software=capture_software(root),
        hardware=capture_hardware(),
    )


if __name__ == "__main__":
    print(capture_env().model_dump_json(indent=2))
