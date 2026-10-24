"""The communication model: the alpha-beta formulas by hand, the link fit, and the costs of each
kind of parallelism."""

import pytest
import torch

from scalelab.bench.comms import SIZES, collective_call
from scalelab.costmodel.comms import (
    COLLECTIVES,
    Collective,
    Link,
    collective_time,
    data_parallel_time,
    fit_link,
    pipeline_bubble,
    tensor_parallel_time,
    weight_sync_time,
)

LINK = Link(alpha=1e-5, beta=1e9)  # 10 us a message, 1 GB/s
MB = 10**6


def test_the_formulas_by_hand() -> None:
    # 4 devices: all-reduce sends 2 · 3/4 of the bytes in 2 · 3 messages; the others half that
    assert collective_time("all_reduce", 8 * MB, 4, LINK) == pytest.approx(1.5 * 8e-3 + 6e-5)
    assert collective_time("all_gather", 8 * MB, 4, LINK) == pytest.approx(0.75 * 8e-3 + 3e-5)
    assert collective_time("reduce_scatter", 8 * MB, 4, LINK) == pytest.approx(6e-3 + 3e-5)


@pytest.mark.parametrize("world", [2, 3, 8])
def test_an_all_reduce_is_a_reduce_scatter_then_an_all_gather(world: int) -> None:
    halves = [collective_time(c, 5 * MB, world, LINK) for c in ("reduce_scatter", "all_gather")]
    assert collective_time("all_reduce", 5 * MB, world, LINK) == pytest.approx(sum(halves))


def test_one_device_alone_sends_nothing() -> None:
    assert all(collective_time(c, 5 * MB, 1, LINK) == 0 for c in COLLECTIVES)


def test_a_large_message_is_bandwidth_bound_and_a_small_one_latency_bound() -> None:
    latency = 2 * (8 - 1) * LINK.alpha  # 8 devices: 14 messages in a row
    assert latency / collective_time("all_reduce", 1000 * MB, 8, LINK) < 1e-4
    assert latency / collective_time("all_reduce", 8, 8, LINK) > 0.999


@pytest.mark.parametrize("collective", COLLECTIVES)
@pytest.mark.parametrize("world", [2, 4])
def test_the_fit_recovers_the_link_that_made_the_times(collective: Collective, world: int) -> None:
    points = [(n, collective_time(collective, n, world, LINK)) for n in SIZES]
    link = fit_link(collective, world, points)
    assert link.alpha == pytest.approx(LINK.alpha) and link.beta == pytest.approx(LINK.beta)


def test_the_fit_weighs_errors_relative_to_the_time() -> None:
    # every time 10% too slow, small or large: the fitted link's times are all 10% slower
    points = [(n, 1.1 * collective_time("all_reduce", n, 2, LINK)) for n in SIZES]
    link = fit_link("all_reduce", 2, points)
    for n, seconds in points:
        assert collective_time("all_reduce", n, 2, link) == pytest.approx(seconds)


def test_times_that_dont_grow_with_the_size_have_no_bandwidth() -> None:
    with pytest.raises(ValueError, match="don't grow with the message size"):
        # faster the larger they get: the best non-negative slope is 0
        fit_link("all_reduce", 2, [(n, 1e-3 / (k + 1)) for k, n in enumerate(SIZES)])


def test_data_parallel_hides_its_overlap_behind_the_backward_pass() -> None:
    whole = collective_time("all_reduce", 40 * MB, 2, LINK)
    assert data_parallel_time(40 * MB, 2, LINK) == whole
    assert data_parallel_time(40 * MB, 2, LINK, overlap=0.75) == pytest.approx(whole / 4)
    assert data_parallel_time(40 * MB, 2, LINK, overlap=1.0) == 0


def test_tensor_parallel_all_reduces_four_times_a_layer_when_training() -> None:
    one = collective_time("all_reduce", MB, 2, LINK)
    assert tensor_parallel_time(6, MB, 2, LINK) == pytest.approx(24 * one)
    assert tensor_parallel_time(6, MB, 2, LINK, backward=False) == pytest.approx(12 * one)


def test_the_pipeline_bubble() -> None:
    assert pipeline_bubble(1, 8) == 0  # one stage never waits
    assert pipeline_bubble(4, 1) == pytest.approx(3 / 4)
    assert pipeline_bubble(4, 13) == pytest.approx(3 / 16)


def test_weight_sync_is_free_when_the_weights_are_shared() -> None:
    assert weight_sync_time(4 * MB, LINK, shared=False) == pytest.approx(4e-3)
    assert weight_sync_time(4 * MB, LINK, shared=True) == 0


@pytest.mark.parametrize("collective", COLLECTIVES)
def test_a_collective_moves_a_whole_number_of_values_a_rank(collective: Collective) -> None:
    # building the call needs no process group: nothing runs until it is called
    _, nbytes = collective_call(collective, 4096, 3, torch.device("cpu"))
    assert nbytes == 4092  # 341 values a rank, 3 ranks, 4 bytes a value
    _, nbytes = collective_call(collective, 4096, 4, torch.device("cpu"))
    assert nbytes == 4096
