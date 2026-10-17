"""Typed experiment configuration"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictModel(BaseModel):
    """Shared base: unknown fields are errors, and configs can't be edited after creation."""

    model_config = ConfigDict(extra="forbid", frozen = True)