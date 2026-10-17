"""Typed experiment configuration"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictModel(BaseModel):
    """Shared base: unknown fields are errors, and configs can't be edited after creation."""

    model_config = ConfigDict(extra="forbid", frozen = True)

class ModelConfig(StrictModel):
    name: str
    n_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    attn: Literal["mha", "gqa", "mla", "swa"] = "mha"
    
    @model_validator(mode="after")
    def check_heads_divide_width(self) -> "ModelConfig":
        if self.d_model % self.n_heads !=0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads({self.n_heads})"
            )
        return self
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

class TaskConfig(StrictModel):
    name: Literal["arithmetic", "countdown"]
    max_digits: int = Field(default= 4, ge=2, le=12)

class OptimConfig(StrictModel):
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)

class ExperimentConfig(StrictModel):
    """Everything that defines WHAT the experiment is."""
    
    name: str
    model: ModelConfig
    task: TaskConfig
    optim: OptimConfig = Field(default_factory=OptimConfig)

class RuntimeConfig(StrictModel):
    """Everything about WHERE and HOW it runs. Never hashed."""
    
    output_dir: str = "runs"
    worker_name: str = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"