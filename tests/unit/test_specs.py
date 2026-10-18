"""Cost model spec tests: parameter counts from the shape alone, exact to the last parameter."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from scalelab.core.config import ModelConfig, load_config
from scalelab.costmodel.specs import ModelSpec, ParamCounts
from scalelab.models.transformer import Transformer

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"


def ladder(name: str, qk_norm: bool = False) -> ModelConfig:
    overrides = [f"model=ladder_s/{name}", f"model.qk_norm={str(qk_norm).lower()}"]
    return load_config(HELLO, overrides).model


def m1_spec(**changes: object) -> ModelSpec:
    """m1's spec, with some fields changed."""
    return ModelSpec.model_validate(ModelSpec.from_config(ladder("m1")).model_dump() | changes)


# --- by hand -------------------------------------------------------------------------------


def test_m1_worked_out_by_hand():
    counts = ModelSpec.from_config(ladder("m1")).param_counts()
    assert counts == ParamCounts(
        embedding=64 * 128,  # 64 tokens x d_model
        attention=4 * (4 * 128 * 128),  # 4 layers x (q, k, v, o)
        mlp=4 * (3 * 128 * 352),  # 4 layers x (gate, up, down)
        norms=(2 * 4 + 1) * 128,  # two per layer, plus the final one
        head=128 * 64,
    )
    assert counts.non_embedding == 803_968
    assert counts.total == 820_352


def test_the_spec_of_a_track_s_model():
    spec = ModelSpec.from_config(ladder("m4"))
    assert (spec.n_layers, spec.d_model, spec.n_heads, spec.head_dim) == (10, 512, 8, 64)
    assert spec.n_kv_heads == spec.n_heads  # MHA: every head has its own keys and values
    assert (spec.ffn, spec.d_ff, spec.vocab, spec.tied_embeddings) == ("swiglu", 1376, 64, False)


# --- exactly what the code builds ----------------------------------------------------------


@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qk-norm"])
@pytest.mark.parametrize("name", ["m1", "m2", "m3", "m4", "m5", "m6", "m7"])
def test_matches_the_real_model_exactly(name: str, qk_norm: bool):
    config = ladder(name, qk_norm)
    counts = ModelSpec.from_config(config).param_counts()
    with torch.device("meta"):
        model = Transformer(config)
    assert counts.non_embedding == model.num_parameters()
    assert counts.total == model.num_parameters(embeddings=True)


def test_each_part_matches_the_real_models_own_layers():
    config = ladder("m1", qk_norm=True)
    parts = {"embedding": 0, "attention": 0, "mlp": 0, "norms": 0, "head": 0}
    for name, parameter in Transformer(config).named_parameters():
        if name.startswith("embed."):
            part = "embedding"
        elif name.startswith("head."):
            part = "head"
        elif ".attn." in name:  # includes the QK-norm weights inside attention
            part = "attention"
        elif ".mlp." in name:
            part = "mlp"
        else:
            part = "norms"  # attn_norm, mlp_norm and the final norm
        parts[part] += parameter.numel()
    assert ModelSpec.from_config(config).param_counts() == ParamCounts(**parts)


# --- shapes the code can't build yet ----------------------------------------------------------


def test_gqa_shares_keys_and_values_between_heads():
    # 4 query heads of 32 sharing 1 key/value head: q and o are 128 x 128, k and v 128 x 32.
    spec = m1_spec(attn="gqa", n_heads=4, head_dim=32, n_kv_heads=1)
    assert spec.param_counts().attention == 4 * (2 * 128 * 128 + 2 * 128 * 32)


def test_a_gelu_mlp_has_two_matrices():
    assert m1_spec(ffn="gelu").param_counts().mlp == 4 * (2 * 128 * 352)


def test_tied_embeddings_have_no_separate_head():
    counts = m1_spec(tied_embeddings=True).param_counts()
    assert counts.head == 0
    assert counts.total == 820_352 - 128 * 64


def test_mla_arrives_later():
    with pytest.raises(NotImplementedError, match="M6"):
        m1_spec(attn="mla").param_counts()


def test_kv_heads_must_divide_the_heads():
    with pytest.raises(ValidationError, match="multiple of n_kv_heads"):
        m1_spec(n_kv_heads=3)


# --- independence ---------------------------------------------------------------------------


def test_the_cost_model_never_imports_pytorch():
    # A fresh Python process, so nothing else has imported torch first.
    code = "import sys, scalelab.costmodel.specs; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
