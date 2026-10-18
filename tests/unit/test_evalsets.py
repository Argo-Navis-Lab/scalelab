"""Frozen eval set tests: the files match their hashes, and the code still makes them exactly."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from scalelab.core.determinism import MAX_SEED, derive_seeds
from scalelab.tasks.arithmetic import ArithmeticProblem, ArithmeticTask
from scalelab.tasks.countdown import CountdownProblem
from scalelab.tasks.evalsets import (
    ARITHMETIC_BANDS,
    COUNTDOWN_BANDS,
    EVAL_DIR,
    EVAL_ENTROPY,
    EVAL_SETS,
    EVAL_SIZE,
    CorruptEvalSetError,
    eval_rng,
    eval_set_name,
    make_eval_set,
    read_eval_set,
    write_all,
)

# --- the frozen files ----------------------------------------------------------------------


def test_every_band_has_a_recorded_hash():
    arithmetic = {eval_set_name("arithmetic", d) for d in ARITHMETIC_BANDS}
    countdown = {eval_set_name("countdown", d) for d in COUNTDOWN_BANDS}
    assert set(EVAL_SETS) == arithmetic | countdown


def test_the_code_still_makes_exactly_the_committed_files(tmp_path: Path):
    # Remake every set from scratch: the same hashes, and the same bytes as in git.
    assert write_all(tmp_path) == EVAL_SETS
    for name in EVAL_SETS:
        made = (tmp_path / f"{name}.jsonl").read_bytes()
        assert made == (EVAL_DIR / f"{name}.jsonl").read_bytes(), name


@pytest.mark.parametrize("difficulty", ARITHMETIC_BANDS)
def test_arithmetic_sets(difficulty: int):
    problems = read_eval_set(eval_set_name("arithmetic", difficulty), ArithmeticProblem)
    assert len(problems) == EVAL_SIZE
    assert all(len(str(p.a)) == len(str(p.b)) == difficulty for p in problems)
    assert len({p.question for p in problems}) == EVAL_SIZE  # no problem appears twice


@pytest.mark.parametrize("difficulty", COUNTDOWN_BANDS)
def test_countdown_sets(difficulty: int):
    problems = read_eval_set(eval_set_name("countdown", difficulty), CountdownProblem)
    assert len(problems) == EVAL_SIZE
    assert all(len(p.numbers) == difficulty for p in problems)
    assert len({p.question for p in problems}) == EVAL_SIZE


# --- mistakes are caught -------------------------------------------------------------------


def test_a_changed_file_is_refused(tmp_path: Path):
    data = (EVAL_DIR / "arithmetic-3.jsonl").read_bytes()
    tampered = data.replace(b"1", b"2", 1)  # change a single digit
    assert tampered != data
    (tmp_path / "arithmetic-3.jsonl").write_bytes(tampered)
    with pytest.raises(CorruptEvalSetError, match="recorded sha256"):
        read_eval_set("arithmetic-3", ArithmeticProblem, tmp_path)


def test_unknown_names_are_refused():
    with pytest.raises(ValueError, match="no eval set named 'arithmetic-9'"):
        read_eval_set("arithmetic-9", ArithmeticProblem)


def test_reading_a_file_as_the_wrong_kind_of_problem_fails():
    with pytest.raises(ValidationError):
        read_eval_set("arithmetic-3", CountdownProblem)


def test_a_band_too_small_for_the_set_is_refused():
    # 2-digit addition has only 90 * 90 = 8100 different problems.
    with pytest.raises(ValueError, match="fewer than 10000 different problems"):
        make_eval_set(ArithmeticTask("+"), 2, n=10_000)


# --- why the sets are made this way --------------------------------------------------------


def test_raw_draws_repeat_in_small_bands():
    raw = ArithmeticTask().sample(eval_rng("arithmetic", 2), EVAL_SIZE, 2)
    assert len({p.question for p in raw}) < EVAL_SIZE  # so make_eval_set skips repeats


def test_eval_streams_are_not_run_streams():
    assert EVAL_ENTROPY > MAX_SEED  # no run seed can be the eval sets' seed
    eval_draws = eval_rng("arithmetic", 3).integers(2**63, size=4)
    for seed in range(100):
        run_draws = np.random.default_rng(derive_seeds(seed).data).integers(2**63, size=4)
        assert not np.array_equal(eval_draws, run_draws)


def test_every_task_and_band_has_its_own_stream():
    firsts = {
        int(eval_rng(t, d).integers(2**63)) for t in ("arithmetic", "countdown") for d in (3, 4)
    }
    assert len(firsts) == 4


def test_1000_problems_pin_a_pass_rate_within_1_6_points():
    worst = math.sqrt(0.5 * 0.5 / EVAL_SIZE)  # the standard error is largest at a 50% pass rate
    assert worst < 0.016
