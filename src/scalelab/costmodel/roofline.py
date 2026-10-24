"""Roofline timing: how long each op takes on a device, and which limit it runs into.

An op takes the longer of its arithmetic time (its FLOPs at the device's peak rate) and its
memory time (its bytes at the device's bandwidth), plus a fixed cost to launch it. Real
kernels reach only a fraction of either peak: benchmarks fit those fractions later, and
until then every op is ideal, so each prediction is a lower bound.
"""

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import Field

from scalelab.core.config import StrictModel
from scalelab.costmodel.ops import Op, OpClass, op_graph
from scalelab.costmodel.specs import Dtype, HardwareSpec, ModelSpec, WorkloadSpec

# compute: arithmetic sets the op's time. memory: moving its bytes does. launch: the fixed
# cost of starting it is bigger than both (tiny ops).
Bound = Literal["compute", "memory", "launch"]
DTYPE: Dtype = "fp32"  # the op graph counts 4-byte values (VALUE_BYTES); fp16 joins at m3s15


class Efficiency(StrictModel):
    """How close one class of op gets to the datasheet: the fractions of peak FLOP/s and of
    bandwidth it reaches, and a fixed cost per op. Calibration fits these (m3s7)."""

    compute: float = Field(default=1.0, gt=0, le=1)  # η_c
    memory: float = Field(default=1.0, gt=0, le=1)  # η_m
    launch_s: float = Field(default=0.0, ge=0)  # t_launch, in seconds


IDEAL = Efficiency()  # the datasheet's peaks, reached exactly, and launches for free


class OpTime(StrictModel):
    """One op's time on a device, term by term."""

    op: Op
    compute_s: float  # its FLOPs at the FLOP/s it reaches
    memory_s: float  # its bytes at the bandwidth it reaches
    launch_s: float

    @property
    def seconds(self) -> float:
        """Arithmetic and memory traffic overlap, so only the slower counts; launching adds."""
        return max(self.compute_s, self.memory_s) + self.launch_s

    @property
    def bound(self) -> Bound:
        """The largest term: what would have to get faster for this op to."""
        terms: dict[Bound, float] = {
            "compute": self.compute_s,
            "memory": self.memory_s,
            "launch": self.launch_s,
        }
        return max(terms, key=lambda bound: terms[bound])  # a tie goes to the first


def op_time(op: Op, hardware: HardwareSpec, efficiency: Efficiency = IDEAL) -> OpTime:
    """t_op = max(F / (P η_c), M / (B η_m)) + t_launch."""
    return OpTime(
        op=op,
        compute_s=op.flops / (hardware.peak_flops[DTYPE] * efficiency.compute),
        memory_s=(op.bytes_read + op.bytes_written) / (hardware.memory_bw * efficiency.memory),
        launch_s=efficiency.launch_s,
    )


class ClassTime(StrictModel):
    """One op class's share of a prediction."""

    ops: int
    seconds: float
    bound: Bound  # the bound of the ops that take most of these seconds


class Prediction(StrictModel):
    """A workload's predicted time on one device, op by op, and what limits it."""

    hardware: str
    times: tuple[OpTime, ...]  # in run order

    @property
    def seconds(self) -> float:
        return sum(t.seconds for t in self.times)

    @property
    def by_bound(self) -> dict[Bound, float]:
        """The seconds spent in ops of each bound."""
        return _by_bound(self.times)

    @property
    def bottleneck(self) -> Bound:
        """The bound that holds most of the time: the label a report leads with."""
        return _bottleneck(self.times)

    @property
    def by_class(self) -> dict[OpClass, ClassTime]:
        """Each op class's ops, seconds and bound, classes in the order they first run."""
        classes: dict[OpClass, list[OpTime]] = {}
        for t in self.times:
            classes.setdefault(t.op.op_class, []).append(t)
        return {
            op_class: ClassTime(
                ops=len(times), seconds=sum(t.seconds for t in times), bound=_bottleneck(times)
            )
            for op_class, times in classes.items()
        }


def _by_bound(times: Iterable[OpTime]) -> dict[Bound, float]:
    seconds: dict[Bound, float] = {"compute": 0.0, "memory": 0.0, "launch": 0.0}
    for t in times:
        seconds[t.bound] += t.seconds
    return seconds


def _bottleneck(times: Iterable[OpTime]) -> Bound:
    seconds = _by_bound(times)
    return max(seconds, key=lambda bound: seconds[bound])


def time_ops(
    ops: Iterable[Op],
    hardware: HardwareSpec,
    efficiencies: Mapping[OpClass, Efficiency] | None = None,
) -> Prediction:
    """Times every op with its class's efficiency (all ideal when there are none yet)."""
    times = tuple(
        op_time(op, hardware, IDEAL if efficiencies is None else efficiencies[op.op_class])
        for op in ops
    )
    return Prediction(hardware=hardware.name, times=times)


def predict(
    model: ModelSpec,
    workload: WorkloadSpec,
    hardware: HardwareSpec,
    efficiencies: Mapping[OpClass, Efficiency] | None = None,
) -> Prediction:
    """A workload's op graph, timed on a device."""
    return time_ops(op_graph(model, workload), hardware, efficiencies)
