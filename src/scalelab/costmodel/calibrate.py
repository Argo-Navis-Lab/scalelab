"""Calibration: the efficiencies that make roofline timing match a device's benchmarks, fitted
on log time, and saved with the versions the benchmarks ran on."""

import statistics
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple, get_args

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from scalelab.core.config import StrictModel
from scalelab.core.envcapture import EnvInfo
from scalelab.core.storage import atomic_write_text
from scalelab.costmodel.ops import Op
from scalelab.costmodel.roofline import Efficiency, Group, group, time_ops
from scalelab.costmodel.specs import HardwareSpec
from scalelab.telemetry.detectors import MAD_TO_STD, median_and_mad

GROUPS: tuple[Group, ...] = get_args(Group)
START = Efficiency(compute=0.5, memory=0.5, launch_s=1e-6)  # where the fit begins
FLOOR = 1e-4  # the smallest efficiency the fit may reach
LAUNCH_RANGE = (1e-9, 1e-2)  # seconds: a launch costs between a nanosecond and 10 ms
# soft-L1: an error in log time up to about this size counts squared, a bigger one only
# linearly, so a few odd measurements can't drag every efficiency toward them.
LOSS_SCALE = 0.1


class Sample(NamedTuple):
    """A benchmark: ops run back to back (one op, or a whole step's), and how long they took."""

    suite: str
    ops: Sequence[Op]
    seconds: float


class Errors(StrictModel):
    """How far predictions land from measurements, each as predicted / measured - 1."""

    samples: int
    mape: float  # the mean size of the error
    bias: float  # the mean error: above 0, the predictions run slow
    worst: float  # the error of the largest size
    # m3s25: their median, and a standard deviation estimated from their MAD: a few wild runs
    # barely move either. The throughput detector's band; None in files saved before m3s25
    median: float | None = None
    spread: float | None = None


def errors_between(predicted: Sequence[float], measured: Sequence[float]) -> Errors:
    gaps = [p / m - 1 for p, m in zip(predicted, measured, strict=True)]
    median, mad = median_and_mad(gaps)
    return Errors(
        samples=len(gaps),
        mape=statistics.fmean(abs(gap) for gap in gaps),
        bias=statistics.fmean(gaps),
        worst=max(gaps, key=abs),
        median=median,
        spread=MAD_TO_STD * mad,
    )


class Calibration(StrictModel):
    """A device's fitted efficiencies, what they were fitted on, and how well they fit."""

    hardware: str
    versions: dict[str, str]  # what the benchmarks ran on (env_versions): a change means refit
    efficiencies: dict[Group, Efficiency]
    errors: dict[str, Errors]  # by suite: the fit against that suite's own samples


def env_versions(env: EnvInfo) -> dict[str, str]:
    """What a calibration depends on: torch, the CUDA it was built with, and the device."""
    gpus = ", ".join(gpu.name for gpu in env.hardware.gpus)
    return {
        "torch": env.software.packages.get("torch") or "none",
        "cuda": env.software.cuda or "none",
        "device": gpus or env.hardware.cpu_model,
    }


def fit(samples: Sequence[Sample], hardware: HardwareSpec) -> dict[Group, Efficiency]:
    """The efficiencies of every group the samples use, fitted to all samples at once (a whole
    step's ops span groups): least squares on log time with a soft-L1 loss."""
    ops = [op for sample in samples for op in sample.ops]
    owner = np.array([i for i, sample in enumerate(samples) for _ in sample.ops])
    names: list[Group] = [group(op) for op in ops]
    groups: list[Group] = [name for name in GROUPS if name in names]
    which = np.array([groups.index(name) for name in names])
    compute = np.array([op.flops / hardware.peak_flops[op.dtype] for op in ops])  # s at peak
    memory = np.array([op.bytes_read + op.bytes_written for op in ops]) / hardware.memory_bw
    measured = np.log([sample.seconds for sample in samples])

    # A row a group: log η_c, log η_m, log t_launch. η_c stays 1 where nothing does FLOPs.
    free = np.ones((len(groups), 3), dtype=bool)
    free[:, 0] = np.bincount(which, weights=compute, minlength=len(groups)) > 0
    logs = np.log([[START.compute if f else 1.0, START.memory, START.launch_s] for f in free[:, 0]])
    low = np.broadcast_to(np.log([FLOOR, FLOOR, LAUNCH_RANGE[0]]), free.shape)[free]
    high = np.broadcast_to(np.log([1.0, 1.0, LAUNCH_RANGE[1]]), free.shape)[free]

    def residuals(x: NDArray[np.float64]) -> NDArray[np.float64]:
        trial = logs.copy()
        trial[free] = x
        eta = np.exp(trial)[which]  # each op's group's row
        seconds = np.maximum(compute / eta[:, 0], memory / eta[:, 1]) + eta[:, 2]
        return np.log(np.bincount(owner, weights=seconds, minlength=len(samples))) - measured

    result = least_squares(
        residuals, logs[free], bounds=(low, high), loss="soft_l1", f_scale=LOSS_SCALE
    )
    # A term no sample was ever limited by can't be fitted: it stays ideal, the roofline's bound.
    logs[free] = np.where(np.any(result.jac != 0, axis=0), result.x, 0.0)
    return {
        name: Efficiency(compute=c, memory=m, launch_s=t)
        for name, (c, m, t) in zip(groups, np.exp(logs).tolist(), strict=True)
    }


def calibrate(
    samples: Sequence[Sample], hardware: HardwareSpec, versions: Mapping[str, str]
) -> Calibration:
    """Fits a device's efficiencies, then scores the fit suite by suite, as predictions will
    be timed."""
    efficiencies = fit(samples, hardware)
    suites: dict[str, tuple[list[float], list[float]]] = {}
    for sample in samples:
        predicted, measured = suites.setdefault(sample.suite, ([], []))
        predicted.append(time_ops(sample.ops, hardware, efficiencies).seconds)
        measured.append(sample.seconds)
    return Calibration(
        hardware=hardware.name,
        versions=dict(versions),
        efficiencies=efficiencies,
        errors={suite: errors_between(*pair) for suite, pair in suites.items()},
    )


class StaleCalibration(UserWarning):
    """The software or the device changed since the benchmarks ran: bench and fit again."""


def save_calibration(path: Path, calibration: Calibration) -> None:
    atomic_write_text(path, calibration.model_dump_json(indent=2) + "\n")


def load_calibration(path: Path, current: Mapping[str, str] | None = None) -> Calibration:
    """A saved calibration. Given the versions in use now (env_versions), warns about each one
    that differs from what the benchmarks ran on: the efficiencies may no longer hold."""
    calibration = Calibration.model_validate_json(path.read_text())
    for key, now in (current or {}).items():
        then = calibration.versions.get(key)
        if now != then:
            warnings.warn(
                f"{path.name} was fitted on {key} {then}, this is {now}: recalibrate",
                StaleCalibration,
                stacklevel=2,
            )
    return calibration
