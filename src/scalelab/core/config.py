"""Typed experiment configuration, composed from YAML files plus command-line overrides."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Shared base: unknown fields are errors, and configs can't be edited after creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    name: str
    n_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    attn: Literal["mha", "gqa", "mla", "swa"] = "mha"
    context: int = Field(default=256, gt=0)  # the longest sequence the model reads, in tokens
    qk_norm: bool = False  # normalize attention's queries and keys (off by default)

    @model_validator(mode="after")
    def check_heads(self) -> "ModelConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for rotary embeddings")
        return self

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        """The SwiGLU MLP's hidden width: 8/3 of d_model, rounded up to a multiple of 32
        (whole-number maths: dividing by 96 = 3 * 32 and rounding up)."""
        return (8 * self.d_model + 95) // 96 * 32


# The difficulty levels each task can make: arithmetic's digits, Countdown's numbers. They
# repeat the tasks' own limits, so a bad level is caught when the config is read (a test
# keeps the two in step).
DIFFICULTIES = {"arithmetic": range(2, 13), "countdown": range(3, 7)}


class TaskConfig(StrictModel):
    name: Literal["arithmetic", "countdown"]
    difficulties: tuple[int, ...] = Field(min_length=1)  # each problem's level, drawn evenly

    @model_validator(mode="after")
    def check_difficulties(self) -> "TaskConfig":
        allowed = DIFFICULTIES[self.name]
        wrong = [d for d in self.difficulties if d not in allowed]
        if wrong:
            raise ValueError(
                f"{self.name} difficulties run from {allowed.start} to {allowed.stop - 1}, "
                f"got {wrong}"
            )
        return self


class OptimConfig(StrictModel):
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)


class WarmStartConfig(StrictModel):
    """Supervised training on worked solutions, before RL: how much, and how fast."""

    tokens: int = Field(default=100_000_000, gt=0)  # the budget: positions read, padding excluded
    batch_size: int = Field(default=64, gt=0)  # sequences per step
    lr: float = Field(default=1e-3, gt=0)  # the peak learning rate
    warmup: float = Field(default=0.02, ge=0, lt=1)  # share of the budget spent ramping lr up
    decay_to: float = Field(default=0.1, ge=0, le=1)  # the final lr, as a fraction of lr
    weight_decay: float = Field(default=0.1, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)  # the largest gradient norm a step may use


class ExperimentConfig(StrictModel):
    """Everything that defines WHAT the experiment is. This gets hashed."""

    name: str
    model: ModelConfig
    task: TaskConfig
    optim: OptimConfig = Field(default_factory=OptimConfig)
    warm_start: WarmStartConfig = Field(default_factory=WarmStartConfig)


class RuntimeConfig(StrictModel):
    """Everything about WHERE and HOW it runs. Never hashed."""

    output_dir: str = "runs"
    worker_name: str = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    deterministic: bool | None = None  # None: on for CPU runs, off for GPU runs


# --- composing a config from several files --------------------------------------------

# A `defaults:` entry like `- model: ladder_s/m1` loads configs/models/ladder_s/m1.yaml
# into the `model` section. These are the groups that can be chosen that way.
GROUP_FOLDERS = {"model": "models", "task": "tasks"}


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping (key: value lines)")
    return data


def _configs_folder(path: Path) -> Path:
    """The `configs/` folder that `path` lives in; group files are looked up from there."""
    for folder in path.resolve().parents:
        if folder.name == "configs":
            return folder
    raise ValueError(f"{path} must be inside a folder named 'configs'")


def _deep_merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """Combine two nested dicts. Where both have a value, `top` wins, section by section."""
    merged = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set data["model"]["n_layers"] = value, given "model.n_layers"."""
    *sections, last = dotted_key.split(".")
    node = data
    for section in sections:
        node = node.setdefault(section, {})
        if not isinstance(node, dict):
            raise ValueError(f"can't set {dotted_key!r}: {section!r} is not a section")
    node[last] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Split "key=value" and read the value as YAML, so "6" becomes 6 and "true" True."""
    key, equals, value = text.partition("=")
    if not equals or not key:
        raise ValueError(f"overrides look like key=value, got {text!r}")
    return key, yaml.safe_load(value)


def compose_config(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Build the raw config dict, in order: defaults, then the file itself, then overrides.

    An override whose key is a group name picks a different file for that group, as in
    model=ladder_s/m4. Any other override sets one value, as in optim.lr=1e-3.
    """
    path = Path(path)
    raw = _read_yaml(path)
    choices: dict[str, str] = {}
    for entry in raw.pop("defaults", []):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"each defaults entry looks like `- group: choice`, got {entry!r}")
        ((group, choice),) = entry.items()
        choices[group] = choice

    values = []
    for text in overrides:
        key, value = parse_override(text)
        if key in GROUP_FOLDERS:
            choices[key] = value
        else:
            values.append((key, value))

    composed: dict[str, Any] = {}
    for group, choice in choices.items():
        if group not in GROUP_FOLDERS:
            raise ValueError(f"unknown config group {group!r}; known: {sorted(GROUP_FOLDERS)}")
        group_file = _configs_folder(path) / GROUP_FOLDERS[group] / f"{choice}.yaml"
        if not group_file.is_file():
            raise FileNotFoundError(f"no {group} config {choice!r}: {group_file} doesn't exist")
        composed[group] = _read_yaml(group_file)

    composed = _deep_merge(composed, raw)
    for key, value in values:
        _set_dotted(composed, key, value)
    return composed


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Compose a YAML config (plus any key=value overrides) and validate it."""
    return ExperimentConfig.model_validate(compose_config(path, overrides))
