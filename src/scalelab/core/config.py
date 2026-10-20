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

# What arithmetic can be narrowed to: some of its operations, written in the order "+-*" (the
# order tasks/arithmetic.py keeps them in), so each choice has one spelling. All three is the
# default, None.
ArithmeticOps = Literal["+", "-", "*", "+-", "+*", "-*"]


class TaskConfig(StrictModel):
    name: Literal["arithmetic", "countdown"]
    difficulties: tuple[int, ...] = Field(min_length=1)  # each problem's level, drawn evenly
    ops: ArithmeticOps | None = None  # arithmetic only: None mixes all three operations

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

    @model_validator(mode="after")
    def check_ops(self) -> "TaskConfig":
        if self.ops is not None and self.name != "arithmetic":
            raise ValueError(f"only arithmetic's operations can be chosen, not {self.name}'s")
        return self


class WarmStartConfig(StrictModel):
    """Supervised training on worked solutions, before RL: how much, and how fast."""

    tokens: int = Field(default=100_000_000, gt=0)  # the budget: positions read, padding excluded
    batch_size: int = Field(default=64, gt=0)  # sequences per step
    lr: float = Field(default=1e-3, gt=0)  # the peak learning rate
    warmup: float = Field(default=0.02, ge=0, lt=1)  # share of the budget spent ramping lr up
    decay_to: float = Field(default=0.1, ge=0, le=1)  # the final lr, as a fraction of lr
    weight_decay: float = Field(default=0.1, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)  # the largest gradient norm a step may use


class EvalConfig(StrictModel):
    """How a run is graded: greedy pass@1 on the frozen eval sets of its training levels."""

    every: int = Field(default=1000, ge=0)  # steps between gradings; 0 means only at the end
    problems: int = Field(default=1000, gt=0)  # the first this many of each eval set


class SamplingParams(StrictModel):
    """How a rollout picks each next token: drawn at random from the model's probabilities,
    made sharper or flatter by the temperature, and cut down to the likeliest tokens by
    top_p. (Greedy decoding, always the likeliest token, is grading's job and needs no
    settings.)"""

    temperature: float = Field(default=1.0, gt=0)  # below 1 sharper, above 1 flatter
    # Draw only from the fewest likeliest tokens holding at least this much probability; 1
    # keeps every token. Below 1, the draws no longer come from the policy itself.
    top_p: float = Field(default=1.0, gt=0, le=1)


# The names RL's knobs take. The code that does the work (rl/advantages.py, rl/losses.py)
# uses these same types, so a config can't name something the code can't do.
Estimator = Literal["grpo", "dr_grpo", "rloo"]
Normalizer = Literal["seq_mean", "token_mean", "constant"]
Overlong = Literal["none", "truncate_mask", "soft_penalty"]


class AlgoConfig(StrictModel):
    """One RL algorithm, as a setting of the unified loss's knobs. GRPO, Dr. GRPO, RLOO and
    the DAPO-style preset are files in configs/algos/ that set them differently."""

    name: str
    estimator: Estimator  # how a group's rewards become advantages
    clip_low: float = Field(ge=0, lt=1)  # the ratio's range is [1 - clip_low, 1 + clip_high]
    clip_high: float = Field(ge=0)
    normalizer: Normalizer  # how per-token values are averaged into the loss
    # The constant normalizer divides by rows times this: the longest answer allowed.
    normalizer_constant: int = Field(default=256, gt=0)
    kl_coef: float = Field(default=0.0, ge=0)  # beta: the weight of the KL to the reference
    format_weight: float = Field(default=0.0, ge=0)  # the weight of the reward's format part
    dynamic_sampling: bool = False  # drop groups whose rewards are all equal
    overlong: Overlong = "none"  # what happens to answers that run too long
    # The soft penalty's zone: the last this many tokens before an answer fills the context.
    overlong_buffer: int = Field(default=16, gt=0)


# What a run_id looks like (core/hashing.py makes them): config hash, seed, commit.
RUN_ID = r"^[0-9a-f]{16}-s[0-9]+-[0-9a-f]{8}$"


class RLConfig(StrictModel):
    """RL after a warm start: the finished run it starts from, the compute it may spend, how
    each step's answers are made, and how the model is updated from them. (How the loss is
    worked out is the algo section's job.)"""

    # The run_id of a finished warm start of the same model: RL's first weights, and the
    # reference its KL penalty measures against.
    from_run: str = Field(pattern=RUN_ID)
    flops: float = Field(gt=0)  # the budget: sampling, log-probs and training, counted by A
    prompts_per_step: int = Field(default=64, gt=0)  # problems drawn each step
    group_size: int = Field(default=8, ge=2)  # answers to each problem, compared to each other
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    epochs: int = Field(default=1, gt=0)  # updates made from each step's answers
    micro_batch: int = Field(default=64, gt=0)  # answers per backward pass, to fit in memory
    lr: float = Field(default=1e-5, gt=0)  # constant: no warmup, no decay
    weight_decay: float = Field(default=0.0, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)  # the largest gradient norm an update may use


class ExperimentConfig(StrictModel):
    """Everything that defines WHAT the experiment is. This gets hashed.

    An experiment is a warm start, or RL after one. RL experiments have an algo and an rl
    section, and warm starts have neither."""

    name: str
    model: ModelConfig
    task: TaskConfig
    # RL experiments carry this section too, with its defaults, but don't use it: their
    # weights come from rl.from_run.
    warm_start: WarmStartConfig = Field(default_factory=WarmStartConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    algo: AlgoConfig | None = None  # chosen like a model: algo=grpo picks configs/algos/grpo.yaml
    rl: RLConfig | None = None

    @model_validator(mode="after")
    def check_rl(self) -> "ExperimentConfig":
        if (self.algo is None) != (self.rl is None):
            raise ValueError("an RL experiment needs both an algo and an rl section")
        return self


class RuntimeConfig(StrictModel):
    """Everything about WHERE and HOW it runs. Never hashed."""

    output_dir: str = "runs"
    worker_name: str = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    deterministic: bool | None = None  # None: on for CPU runs, off for GPU runs
    # Keep the live registry here, on the machine's own disk, and copy it to <home> after
    # every change. For Colab, where <home> is on Google Drive: SQLite over the Drive mount
    # is unreliable. None keeps the one registry at <home>/registry.db.
    local_registry: str | None = None
    # Also save a checkpoint every this many steps. None: only every 15 minutes, when asked
    # to stop, and at the end.
    checkpoint_every: int | None = Field(default=None, gt=0)


# --- composing a config from several files --------------------------------------------

# A `defaults:` entry like `- model: ladder_s/m1` loads configs/models/ladder_s/m1.yaml
# into the `model` section. These are the groups that can be chosen that way.
GROUP_FOLDERS = {"model": "models", "task": "tasks", "algo": "algos"}


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
    """Split "key=value" and read the value as YAML, so "6" becomes 6 and "true" True. A
    value YAML can't read stays text, such as "*" (which YAML takes for the start of an
    alias)."""
    key, equals, value = text.partition("=")
    if not equals or not key:
        raise ValueError(f"overrides look like key=value, got {text!r}")
    try:
        return key, yaml.safe_load(value)
    except yaml.YAMLError:
        return key, value


def compose_config(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Build the raw config dict, in order: defaults, then the file itself, then overrides.

    An override whose key is a group name picks a different file for that group, as in
    model=ladder_s/m4. Any other override sets one value, as in warm_start.lr=3e-4.
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
