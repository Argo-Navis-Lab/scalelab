"""The collectives benchmark: all-reduce, all-gather and reduce-scatter among several processes
on this machine, over gloo (CPU tensors) or NCCL (one GPU a process)."""

import socket
import tempfile
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from pydantic import Field, TypeAdapter
from torch.multiprocessing.spawn import start_processes

from scalelab.bench.timing import Timing, measure
from scalelab.core.config import StrictModel
from scalelab.core.storage import atomic_write_bytes
from scalelab.costmodel.comms import COLLECTIVES, Collective

Backend = Literal["gloo", "nccl"]
SIZES = tuple(1024 * 4**k for k in range(1, 9))  # 4 KB to 64 MB: latency-bound to bandwidth-bound
WARMUP = 10  # untimed calls first: the first ones also open the connections between ranks
REPEATS = 15  # one call a sample (see _rank), so more samples than measure's 7


class CommMeasurement(StrictModel):
    """One collective's time at one message size and world size, as rank 0 saw it."""

    collective: Collective
    world: int = Field(gt=1)  # processes taking part
    nbytes: int = Field(gt=0)  # all-reduce's buffer, all-gather's output, reduce-scatter's input
    timing: Timing

    @property
    def seconds(self) -> float:
        return self.timing.median


RESULTS = TypeAdapter(list[CommMeasurement])


def collective_call(
    collective: Collective, nbytes: int, world: int, device: torch.device
) -> tuple[Callable[[], object], int]:
    """One collective over float32 tensors on `device`, ready to call, and the bytes it moves
    (`nbytes` rounded down to a multiple of 4 × world). Zeros: summing them stays finite."""
    piece = nbytes // (4 * world)  # each rank's share, in values
    whole = torch.zeros(piece * world, device=device)
    part = torch.zeros(piece, device=device)
    if collective == "all_reduce":
        call = partial(dist.all_reduce, whole)
    elif collective == "all_gather":
        call = partial(dist.all_gather_single, whole, part)
    else:
        call = partial(dist.reduce_scatter_single, part, whole)
    return call, 4 * piece * world


def _rank(
    rank: int,
    world: int,
    backend: Backend,
    port: int,
    sizes: Sequence[int],
    collectives: Sequence[Collective],
    repeats: int,
    out: Path,
) -> None:
    """One process of the benchmark: joins the group, times every collective, and rank 0 saves
    what it measured to `out` for the parent."""
    device = torch.device("cpu")
    if backend == "nccl":
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend, init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    try:
        results: list[CommMeasurement] = []
        for collective in collectives:
            for size in sizes:
                call, nbytes = collective_call(collective, size, world, device)
                # min_sample_s 0: one call a sample. Every rank must make the same calls, and
                # measure's doubling decides from each rank's own clock.
                timing = measure(call, device, warmup=WARMUP, repeats=repeats, min_sample_s=0)
                results.append(
                    CommMeasurement(
                        collective=collective, world=world, nbytes=nbytes, timing=timing
                    )
                )
        if rank == 0:
            atomic_write_bytes(out, RESULTS.dump_json(results))
    finally:
        dist.destroy_process_group()


def free_port() -> int:
    """A TCP port nobody is listening on, for the processes to meet at."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def bench_collectives(
    world: int,
    backend: Backend = "gloo",
    *,
    sizes: Sequence[int] = SIZES,
    collectives: Sequence[Collective] = COLLECTIVES,
    repeats: int = REPEATS,
) -> list[CommMeasurement]:
    """Starts `world` fresh processes (spawned: each imports torch anew), times each collective
    at each size among them, and returns rank 0's timings."""
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "results.json"
        args = (world, backend, free_port(), tuple(sizes), tuple(collectives), repeats, out)
        start_processes(_rank, args=args, nprocs=world, start_method="spawn")
        return RESULTS.validate_json(out.read_bytes())
