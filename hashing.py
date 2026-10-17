"""Run identity: config -> canonical JSON -> config_hash -> run_id (blueprint Section 4.1)."""

import hashlib
import json
import re

from pydantic import BaseModel

_HEX = re.compile(r"[0-9a-f]+")


def canonical_json(config: BaseModel) -> str:
    """One exact text form per config: defaults filled, keys sorted, no whitespace."""
    data = config.model_dump(mode="json")
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def config_hash(config: BaseModel) -> str:
    """First 16 hex characters of the SHA-256 of the canonical JSON."""
    digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    return digest[:16]


def run_id(config: BaseModel, seed: int, git_sha: str) -> str:
    """Identify one run: which experiment, which seed, which code version."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    sha = git_sha.strip().lower()
    if len(sha) < 8 or not _HEX.fullmatch(sha):
        raise ValueError(f"git_sha must be at least 8 hex characters, got {git_sha!r}")
    return f"{config_hash(config)}-s{seed}-{sha[:8]}"