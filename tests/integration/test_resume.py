"""The resume contract: a run interrupted and resumed on CPU is bit-identical to one that wasn't."""

from scalelab.core.checkpoint import latest_checkpoint
from scalelab.core.dummy_job import run_dummy_job

STEPS = 50


def test_uninterrupted_run_trains(tmp_path):
    losses = run_dummy_job(seed=0, steps=STEPS, checkpoint_dir=tmp_path / "ckpt", home=tmp_path)
    assert len(losses) == STEPS
    assert losses[-1] < losses[0]
    assert latest_checkpoint(tmp_path / "ckpt") is None  # nothing asked it to save


def test_stop_and_resume_is_bit_identical(tmp_path):
    uninterrupted = run_dummy_job(0, STEPS, tmp_path / "a", home=tmp_path)

    before = run_dummy_job(0, STEPS, tmp_path / "b", home=tmp_path, stop_after=20)
    assert latest_checkpoint(tmp_path / "b").name == "step-000000020.json"
    after = run_dummy_job(0, STEPS, tmp_path / "b", home=tmp_path)  # a "new session"

    assert len(before) == 20 and len(after) == 30
    assert before + after == uninterrupted  # exactly equal, every step


def test_interrupted_twice_is_still_bit_identical(tmp_path):
    uninterrupted = run_dummy_job(0, STEPS, tmp_path / "a", home=tmp_path)
    first = run_dummy_job(0, STEPS, tmp_path / "b", home=tmp_path, stop_after=7)
    second = run_dummy_job(0, STEPS, tmp_path / "b", home=tmp_path, stop_after=33)
    third = run_dummy_job(0, STEPS, tmp_path / "b", home=tmp_path)
    assert first + second + third == uninterrupted
