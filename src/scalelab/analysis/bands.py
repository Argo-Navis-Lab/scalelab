"""The difficulty bands the M4 RL task keeps, by a rule fixed before the warm starts ran:
where the smallest model already solves some problems (RL learns only from the rewards it
gets) and the largest still leaves room above it (for the larger models the prediction is
about)."""

from collections.abc import Mapping

FLOOR = 0.05  # the smallest model's pass rate at least this: some reward to learn from
CEILING = 0.50  # the largest model's at most this: room above it


def pick_bands(smallest: Mapping[int, float], largest: Mapping[int, float]) -> list[int]:
    """The bands where `smallest` (a pass rate for each band) is at least FLOOR and `largest`
    at most CEILING, in increasing order. A band only one of them was graded on is left out."""
    return sorted(
        band
        for band in smallest.keys() & largest.keys()
        if smallest[band] >= FLOOR and largest[band] <= CEILING
    )
