"""Checkpoint save/load tests."""

import json
import random

import numpy as np
import pytest
import torch

from scalelab.core.artifacts import CorruptArtifactError, artifact_path
from scalelab.core.checkpoint import (
    capture_rng,
    latest_checkpoint,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)


def round_trip(state, tmp_path, step=1):
    manifest = save_checkpoint(tmp_path / "ckpt", step, state, home=tmp_path)
    return load_checkpoint(manifest, home=tmp_path)


# --- every kind of value comes back exactly ------------------------------------------


def test_plain_values_round_trip(tmp_path):
    state = {"step": 7, "lr": 3e-4, "name": "m1", "done": False, "nothing": None, "xs": [1, 2]}
    assert round_trip(state, tmp_path) == (1, state)


def test_tuples_stay_tuples(tmp_path):
    _, loaded = round_trip({"betas": (0.9, 0.999)}, tmp_path)
    assert loaded["betas"] == (0.9, 0.999)
    assert isinstance(loaded["betas"], tuple)


def test_int_keys_stay_ints(tmp_path):
    _, loaded = round_trip({"state": {0: "first", 1: "second"}}, tmp_path)
    assert loaded["state"] == {0: "first", 1: "second"}


def test_numpy_arrays_round_trip(tmp_path):
    array = np.arange(5, dtype=np.uint32)
    _, loaded = round_trip({"key": array}, tmp_path)
    assert loaded["key"].dtype == np.uint32
    assert np.array_equal(loaded["key"], array)


def test_tensors_round_trip_bit_for_bit(tmp_path):
    tensors = {
        "weights": torch.randn(3, 4),
        "step": torch.tensor(12.0),  # a single number, like AdamW's step counter
        "rng": torch.get_rng_state(),  # bytes
    }
    _, loaded = round_trip(tensors, tmp_path)
    for name, tensor in tensors.items():
        assert loaded[name].dtype == tensor.dtype
        assert torch.equal(loaded[name], tensor)


def test_saved_tensors_are_a_snapshot(tmp_path):
    weights = torch.zeros(3)
    manifest = save_checkpoint(tmp_path / "ckpt", 1, {"w": weights}, home=tmp_path)
    weights += 1  # training carries on after the save...
    _, loaded = load_checkpoint(manifest, home=tmp_path)
    assert torch.equal(loaded["w"], torch.zeros(3))  # ...but the checkpoint kept the old values


def test_tensors_shared_between_entries_can_be_saved(tmp_path):
    shared = torch.ones(2)  # like tied input and output embeddings later on
    _, loaded = round_trip({"embed": shared, "unembed": shared}, tmp_path)
    assert torch.equal(loaded["embed"], loaded["unembed"])


def test_unsupported_values_are_refused(tmp_path):
    with pytest.raises(TypeError, match="can't put a object in a checkpoint"):
        save_checkpoint(tmp_path / "ckpt", 1, {"thing": object()}, home=tmp_path)


# --- real training state ------------------------------------------------------------


def test_model_and_optimizer_round_trip(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model(torch.ones(1, 4)).sum().backward()
    optimizer.step()  # now AdamW has moment estimates worth saving

    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    _, loaded = round_trip(state, tmp_path)

    fresh_model = torch.nn.Linear(4, 2)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-2)
    fresh_model.load_state_dict(loaded["model"])
    fresh_optimizer.load_state_dict(loaded["optimizer"])
    assert torch.equal(fresh_model.weight, model.weight)
    pairs = zip(optimizer.state.values(), fresh_optimizer.state.values(), strict=True)
    for before, after in pairs:
        assert torch.equal(before["exp_avg"], after["exp_avg"])
        assert torch.equal(before["exp_avg_sq"], after["exp_avg_sq"])


def draw_everything(data_rng: np.random.Generator) -> tuple[float, ...]:
    return (random.random(), np.random.rand(), torch.rand(1).item(), data_rng.random())


def test_random_generators_resume_exactly(tmp_path):
    data_rng = np.random.default_rng(123)
    state = {"rng": capture_rng(), "data_rng": data_rng.bit_generator.state}
    expected = draw_everything(data_rng)  # what comes next, the first time round

    _, loaded = round_trip(state, tmp_path)
    draw_everything(np.random.default_rng(0))  # wander off somewhere else
    restore_rng(loaded["rng"])
    data_rng.bit_generator.state = loaded["data_rng"]
    assert draw_everything(data_rng) == expected


# --- files on disk ---------------------------------------------------------------------


def test_manifest_is_readable_json(tmp_path):
    manifest = save_checkpoint(tmp_path / "ckpt", 42, {"w": torch.ones(2)}, home=tmp_path)
    assert manifest.name == "step-000000042.json"
    data = json.loads(manifest.read_text())
    assert data["format"] == 1
    assert data["step"] == 42
    assert data["state"] == {"w": {"__tensor__": "0"}}
    assert artifact_path(data["tensors_sha256"], tmp_path).is_file()  # tensors in the store


def test_identical_state_stores_its_tensors_once(tmp_path):
    state = {"w": torch.arange(10.0)}
    first = save_checkpoint(tmp_path / "ckpt", 1, state, home=tmp_path)
    second = save_checkpoint(tmp_path / "ckpt", 2, state, home=tmp_path)
    blob = json.loads(first.read_text())["tensors_sha256"]
    assert json.loads(second.read_text())["tensors_sha256"] == blob
    assert len([p for p in (tmp_path / "artifacts").rglob("*") if p.is_file()]) == 1


def test_latest_checkpoint_is_the_highest_step(tmp_path):
    for step in (5, 100, 20):
        save_checkpoint(tmp_path / "ckpt", step, {}, home=tmp_path)
    # Zero-padding matters: as plain text, "100" would sort before "20" and "5".
    assert latest_checkpoint(tmp_path / "ckpt") == tmp_path / "ckpt" / "step-000000100.json"


def test_latest_checkpoint_ignores_half_written_files(tmp_path):
    save_checkpoint(tmp_path / "ckpt", 5, {}, home=tmp_path)
    (tmp_path / "ckpt" / "step-000000010.json.tmp").write_text('{"half')
    assert latest_checkpoint(tmp_path / "ckpt") == tmp_path / "ckpt" / "step-000000005.json"


def test_no_checkpoint_yet(tmp_path):
    assert latest_checkpoint(tmp_path / "ckpt") is None  # the folder doesn't even exist


def test_corrupted_tensors_are_detected(tmp_path):
    manifest = save_checkpoint(tmp_path / "ckpt", 1, {"w": torch.ones(2)}, home=tmp_path)
    blob = json.loads(manifest.read_text())["tensors_sha256"]
    artifact_path(blob, tmp_path).write_bytes(b"garbage")
    with pytest.raises(CorruptArtifactError):
        load_checkpoint(manifest, home=tmp_path)


def test_unknown_format_is_refused(tmp_path):
    manifest = save_checkpoint(tmp_path / "ckpt", 1, {}, home=tmp_path)
    data = json.loads(manifest.read_text())
    data["format"] = 99
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown checkpoint format 99"):
        load_checkpoint(manifest, home=tmp_path)
