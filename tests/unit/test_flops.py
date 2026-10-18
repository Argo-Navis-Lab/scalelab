"""FLOP count tests: by hand, against PyTorch's own counter, the 6N rule, and decoding."""

import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
import torch
from pydantic import ValidationError
from torch.utils.flop_counter import FlopCounterMode

from scalelab.core.config import ModelConfig, load_config
from scalelab.costmodel.ops import FlopCounts, count_flops
from scalelab.costmodel.specs import ModelSpec, WorkloadSpec
from scalelab.models.transformer import Transformer

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
LADDER = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]


def ladder(name: str) -> ModelConfig:
    return load_config(HELLO, [f"model=ladder_s/{name}"]).model


def spec(name: str = "m1") -> ModelSpec:
    return ModelSpec.from_config(ladder(name))


def prefill(batch: int, tokens: int) -> WorkloadSpec:
    return WorkloadSpec(phase="prefill", batch=batch, tokens=tokens)


def decode(batch: int, past: int) -> WorkloadSpec:
    return WorkloadSpec(phase="decode", batch=batch, tokens=1, past=past)


def train(batch: int, tokens: int, activation_ckpt: bool = False) -> WorkloadSpec:
    return WorkloadSpec(phase="train", batch=batch, tokens=tokens, activation_ckpt=activation_ckpt)


def pytorch_matmul_flops(config: ModelConfig, batch: int, tokens: int, backward: bool) -> int:
    """PyTorch's own count of the real model's matrix multiplies (aten.mm), on the meta
    device, where only shapes exist, so even m7 takes no time or memory. Attention's kernel
    isn't a plain matrix multiply, so this is exactly the dense part."""
    with torch.device("meta"):
        model = Transformer(config)
        token_ids = torch.zeros(batch, tokens, dtype=torch.long)
    with FlopCounterMode(display=False) as counter:
        logits = model(token_ids)
        if backward:
            logits.sum().backward()
    return counter.get_flop_counts()["Global"].get(torch.ops.aten.mm, 0)


# --- by hand -------------------------------------------------------------------------------


def test_m1_reading_two_prompts_of_ten_tokens_by_hand():
    tokens, pairs = 2 * 10, 2 * (10 * 11 // 2)  # 20 tokens; each sequence 1 + 2 + ... + 10 pairs
    assert count_flops(spec(), prefill(2, 10)) == FlopCounts(
        projections=4 * 2 * tokens * 128 * (2 * 128 + 2 * 128),  # 4 layers: q, o and k, v
        attention=4 * 4 * pairs * 128,  # scores and mixing, 2 FLOPs each per dimension
        mlp=4 * 2 * tokens * 3 * 128 * 352,  # gate, up and down
        logits=2 * tokens * 128 * 64,
    )


# --- PyTorch agrees, to the last FLOP ---------------------------------------------------------


@pytest.mark.parametrize("phase", ["prefill", "train"])
@pytest.mark.parametrize("name", LADDER)
def test_pytorch_counts_the_same_matrix_multiplies(name: str, phase: Literal["prefill", "train"]):
    workload = WorkloadSpec(phase=phase, batch=2, tokens=256)
    measured = pytorch_matmul_flops(ladder(name), 2, 256, backward=phase == "train")
    assert count_flops(spec(name), workload).dense == measured


# --- training -------------------------------------------------------------------------------


def test_training_costs_three_forward_passes():
    forward, training = count_flops(spec(), prefill(4, 64)), count_flops(spec(), train(4, 64))
    assert training == FlopCounts(**{k: 3 * v for k, v in forward.model_dump().items()})


def test_activation_checkpointing_costs_one_more_forward_pass():
    forward = count_flops(spec(), prefill(4, 64))
    checkpointed = count_flops(spec(), train(4, 64, activation_ckpt=True))
    assert checkpointed == FlopCounts(**{k: 4 * v for k, v in forward.model_dump().items()})


@pytest.mark.parametrize("name", LADDER)
def test_the_6n_rule(name: str):
    # Section 11's golden test: without attention, training costs about 6N FLOPs per token.
    n_params = spec(name).param_counts().non_embedding
    per_token = count_flops(spec(name), train(8, 256)).dense / (8 * 256)
    assert abs(per_token / (6 * n_params) - 1) < 0.05


# --- decoding -------------------------------------------------------------------------------


def test_reading_at_once_or_one_token_at_a_time_costs_the_same():
    at_once = count_flops(spec(), prefill(2, 10))
    steps = [count_flops(spec(), prefill(2, 1))]
    steps += [count_flops(spec(), decode(2, past)) for past in range(1, 10)]
    for part in ("projections", "attention", "mlp", "logits"):
        assert sum(getattr(step, part) for step in steps) == getattr(at_once, part), part


def test_a_generated_tokens_attention_grows_with_the_context():
    early, late = count_flops(spec(), decode(8, 0)), count_flops(spec(), decode(8, 255))
    assert late.attention == 256 * early.attention  # it sees 256 keys instead of 1
    assert late.dense == early.dense  # everything else costs the same per token


def test_gqa_saves_only_on_key_and_value_projections():
    mha = ModelSpec.model_validate(spec().model_dump() | {"n_heads": 4, "head_dim": 32})
    gqa = ModelSpec.model_validate(mha.model_dump() | {"attn": "gqa", "n_kv_heads": 1})
    a, b = count_flops(mha, prefill(2, 10)), count_flops(gqa, prefill(2, 10))
    assert b.projections < a.projections
    assert (b.attention, b.mlp, b.logits) == (a.attention, a.mlp, a.logits)


# --- mistakes are caught --------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"phase": "decode", "batch": 1, "tokens": 2},
        {"phase": "prefill", "batch": 1, "tokens": 5, "past": 3},
        {"phase": "prefill", "batch": 1, "tokens": 5, "activation_ckpt": True},
    ],
    ids=["decode-two-tokens", "prefill-with-a-past", "checkpointing-without-training"],
)
def test_impossible_workloads_are_refused(fields: dict[str, object]):
    with pytest.raises(ValidationError):
        WorkloadSpec.model_validate(fields)


@pytest.mark.parametrize("attn", ["mla", "swa"])
def test_mla_and_sliding_window_arrive_later(attn: str):
    with pytest.raises(NotImplementedError, match="M6"):
        count_flops(ModelSpec.model_validate(spec().model_dump() | {"attn": attn}), prefill(1, 4))


def test_flop_counting_never_imports_pytorch():
    code = "import sys, scalelab.costmodel.ops; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
