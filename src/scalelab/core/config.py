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