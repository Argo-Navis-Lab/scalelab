"""Sweep files (configs/sweeps/*.yaml): one experiment, run once for every combination of a
choice from each axis and a seed. A sweep only lists runs: nothing here runs or records one."""

import itertools
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator

from scalelab.core.config import StrictModel


class Sweep(StrictModel):
    """A sweep file. `axes` maps each axis's name to its choices, each a list of key=value
    overrides (a choice of one override can be written alone); a job takes one choice from
    every axis, in the file's order."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    experiment: Path  # read from the folder scalelab runs in, as `scalelab run`'s argument
    overrides: tuple[str, ...] = ()  # every job's, before its choices
    axes: dict[str, tuple[tuple[str, ...], ...]] = {}
    seeds: tuple[int, ...] = (0,)

    @field_validator("axes", mode="before")
    @classmethod
    def lists_of_overrides(cls, axes: Any) -> Any:
        """A choice written as one override is a list of one."""
        if not isinstance(axes, dict):
            return axes  # pydantic's own check says what is wrong
        return {
            name: [[choice] if isinstance(choice, str) else choice for choice in choices]
            for name, choices in axes.items()
        }

    @model_validator(mode="after")
    def check_choices(self) -> "Sweep":
        if not self.seeds or len(set(self.seeds)) < len(self.seeds):
            raise ValueError(f"seeds must be distinct, and at least one: {list(self.seeds)}")
        for name, choices in self.axes.items():
            if not choices:
                raise ValueError(f"axis {name} has no choices")
        return self


def load_sweep(path: str | Path) -> Sweep:
    """Read and check a sweep file."""
    return Sweep.model_validate(yaml.safe_load(Path(path).read_text()))


def expand(sweep: Sweep) -> list[tuple[str, ...]]:
    """Every job's overrides, as `scalelab run` takes them: the sweep's own, one choice from
    each axis, then seed=N. Seeds outermost: every combination's seed 0 comes before any seed
    1, so a sweep cut short still covers every combination."""
    return [
        (*sweep.overrides, *itertools.chain.from_iterable(choices), f"seed={seed}")
        for seed in sweep.seeds
        for choices in itertools.product(*sweep.axes.values())
    ]
