"""Finished runs, found by run_id: the model a run ended with, ready to grade or to start RL
from."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from scalelab.core.checkpoint import latest_checkpoint, load_checkpoint
from scalelab.core.config import ModelConfig, TaskConfig
from scalelab.core.registry import Registry
from scalelab.models.transformer import Transformer


@dataclass(frozen=True)
class FinishedRun:
    """What a finished run ended with."""

    model: Transformer  # with the weights of the run's last checkpoint, on the CPU
    task: TaskConfig  # what it was trained on
    step: int  # the step of that checkpoint: the run's last step


def load_finished_run(registry: Registry, home: Path, run_id: str) -> FinishedRun:
    """The model run `run_id` ended with, rebuilt from the config the registry recorded for
    it and from its last checkpoint in <home>/checkpoints/<run_id>, where `scalelab run`
    saves them.

    Raises ValueError, saying why, if the registry has no such run, if the run isn't done
    (an unfinished run's last checkpoint isn't its final model), or if `home` holds no
    checkpoint for it.
    """
    run = registry.get_run(run_id)
    if run is None:
        raise ValueError(f"the registry has no run {run_id}")
    if run.status != "done":
        raise ValueError(f"run {run_id} is {run.status}, not done")
    folder = home / "checkpoints" / run_id
    latest = latest_checkpoint(folder)
    if latest is None:
        raise ValueError(f"run {run_id} has no checkpoint in {folder}")

    config_json = registry.experiment_json(run.config_hash)
    assert config_json is not None  # every run row points at an experiment row (a foreign key)
    # Only the model and task sections are read, so a run recorded before the config gained
    # newer sections (RL's, from m2s18) still loads.
    recorded = json.loads(config_json)
    step, state = load_checkpoint(latest, home)
    with torch.random.fork_rng():  # building a model draws random weights: undo those draws
        model = Transformer(ModelConfig.model_validate(recorded["model"]))
    model.load_state_dict(state["model"])
    return FinishedRun(model=model, task=TaskConfig.model_validate(recorded["task"]), step=step)
