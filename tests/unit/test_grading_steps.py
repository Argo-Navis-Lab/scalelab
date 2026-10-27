"""m4s3: when an RL run is graded: step 0, every eval.every steps, the geometric steps of
rl.gradings_per_decade, and the last."""

from pathlib import Path

from scalelab.core.config import ExperimentConfig, load_config
from scalelab.rl.experiment import geometric_steps, graded, grading_steps

RL = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "rl.yaml"


def rl_config(*overrides: str) -> ExperimentConfig:
    return load_config(RL, list(overrides))


def test_four_a_decade_are_evenly_spaced_in_log_steps() -> None:
    steps = geometric_steps(4, 1000)
    assert steps == [1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000]
    for decade in (1, 10, 100):  # 4 in each tenfold stretch: [1, 10), [10, 100), [100, 1000)
        assert len([step for step in steps if decade <= step < 10 * decade]) == 4


def test_crowded_steps_are_kept_once_and_none_go_past_the_last() -> None:
    assert geometric_steps(10, 10) == [1, 2, 3, 4, 5, 6, 8, 10]  # 1.26 rounds to 1 again...
    assert geometric_steps(4, 999) == geometric_steps(4, 1000)[:-1]
    assert geometric_steps(4, 0) == []
    assert geometric_steps(0, 1000) == []  # 0 a decade: none


def test_a_run_is_graded_at_eval_every_the_geometric_steps_and_the_ends() -> None:
    config = rl_config("eval.every=50")
    assert graded(config, 50) and graded(config, 56) and not graded(config, 55)
    assert grading_steps(config, 120) == [0, 1, 2, 3, 6, 10, 18, 32, 50, 56, 100, 120]
    assert grading_steps(config, 1) == [0, 1]  # the last step is graded once
    off = rl_config("eval.every=0", "rl.gradings_per_decade=0")
    assert grading_steps(off, 120) == [0, 120]  # the ends only


def test_rl_runs_get_four_a_decade_by_default() -> None:
    config = rl_config()
    assert config.rl is not None and config.rl.gradings_per_decade == 4
    assert config.eval.every == 1000  # rl.yaml keeps EvalConfig's: the two add up
