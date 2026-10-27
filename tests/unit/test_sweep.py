"""Sweep files (m4s4): what they may say, and the jobs they expand to."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from scalelab.cli import split_overrides
from scalelab.core.sweep import Sweep, expand, load_sweep

SWEEPS = Path(__file__).resolve().parents[2] / "configs" / "sweeps"


def sweep(**fields: object) -> Sweep:
    return Sweep.model_validate({"name": "s", "experiment": "e.yaml", **fields})


def test_jobs_take_one_choice_from_each_axis_with_seeds_outermost() -> None:
    axes = {"model": ["m=1", "m=2"], "algo": [["x=1", "y=1"], "x=2"]}
    jobs = expand(sweep(overrides=["a=1"], axes=axes, seeds=[0, 3]))
    assert jobs[:4] == [
        ("a=1", "m=1", "x=1", "y=1", "seed=0"),
        ("a=1", "m=1", "x=2", "seed=0"),
        ("a=1", "m=2", "x=1", "y=1", "seed=0"),
        ("a=1", "m=2", "x=2", "seed=0"),
    ]
    assert jobs[4:] == [(*job[:-1], "seed=3") for job in jobs[:4]]


def test_without_axes_a_sweep_is_one_job_a_seed() -> None:
    assert expand(sweep(seeds=[2, 1])) == [("seed=2",), ("seed=1",)]
    assert expand(sweep()) == [("seed=0",)]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"seeds": [1, 1]}, "seeds must be distinct"),
        ({"seeds": []}, "seeds must be distinct"),
        ({"axes": {"algo": []}}, "axis algo has no choices"),
        ({"name": "My Sweep"}, "should match pattern"),
        ({"seed": [0]}, "Extra inputs are not permitted"),
    ],
)
def test_a_sweep_says_what_is_wrong(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        sweep(**fields)


def test_a_sweep_file_is_yaml(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("name: s\nexperiment: e.yaml\naxes:\n  algo: [algo=grpo, [algo=rloo, b=2]]\n")
    assert load_sweep(path).axes == {"algo": (("algo=grpo",), ("algo=rloo", "b=2"))}


def test_the_warm_starts_follow_the_m4_plan() -> None:
    # fp16 everywhere, the padding cut from m3 up (m4s2 judged), seed 0 only
    jobs = [split_overrides(job) for job in expand(load_sweep(SWEEPS / "warmstarts.yaml"))]
    models = [
        next(text for text in overrides if text.startswith("model=")) for *_, overrides in jobs
    ]
    assert models == [f"model=ladder_s/m{size}" for size in range(1, 6)]
    assert [seed for seed, _, _ in jobs] == [0] * 5
    assert [runtime.get("precision") for _, runtime, _ in jobs] == ["fp16"] * 5
    assert [runtime.get("padding_cut", False) for _, runtime, _ in jobs] == [False] * 2 + [True] * 3
