"""Memory model tests: exact KV cache bytes, 16 bytes per parameter, activations, what fits."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scalelab.core.config import ModelConfig, load_config
from scalelab.costmodel.memory import (
    RESERVED_FRACTION,
    generation_memory,
    kv_cache_bytes,
    max_generation_batch,
    max_generation_context,
    training_memory,
)
from scalelab.costmodel.specs import HARDWARE, HardwareSpec, ModelSpec
from scalelab.models.attention.mha import MultiHeadAttention
from scalelab.models.transformer import Transformer

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
LADDER = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]


def ladder(name: str) -> ModelConfig:
    return load_config(HELLO, [f"model=ladder_s/{name}"]).model


def spec(name: str = "m1") -> ModelSpec:
    return ModelSpec.from_config(ladder(name))


def usable(hardware: HardwareSpec) -> int:
    return int(hardware.memory_bytes * (1 - RESERVED_FRACTION))


def saved_for_backward(config: ModelConfig, batch: int, seq: int) -> int:
    """Bytes the real model's forward pass keeps for the backward pass, in bfloat16. Each
    piece of memory counts once (several operations can keep the same tensor), and the
    weights don't count: they're parameters, not activations."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = Transformer(config).to(torch.bfloat16)
    weights = {p.untyped_storage().data_ptr() for p in model.parameters()}
    kept: dict[int, int] = {}

    def pack(tensor: torch.Tensor) -> torch.Tensor:  # called for every tensor kept
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in weights:
            kept[storage.data_ptr()] = storage.nbytes()
        return tensor

    tokens = torch.randint(0, 24, (batch, seq), generator=torch.Generator().manual_seed(0))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        model(tokens)
    return sum(kept.values())


# --- the KV cache: exact ----------------------------------------------------------------------


def test_kv_cache_bytes_by_hand():
    # A key and a value x 4 layers x 2 heads x 64 numbers x 2 bytes = 2048 bytes per token.
    assert kv_cache_bytes(spec(), batch=3, context=256, bytes_per_element=2) == 2048 * 256 * 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["float32", "float16"])
def test_kv_cache_bytes_equal_the_real_cache(dtype: torch.dtype):
    cache = Transformer(ladder("m1")).to(dtype).new_cache(3)
    real = sum(layer.k.nbytes + layer.v.nbytes for layer in cache.layers)
    assert kv_cache_bytes(spec(), 3, 256, dtype.itemsize) == real


@pytest.mark.parametrize("name", LADDER)
def test_kv_cache_bytes_equal_cache_spec(name: str):
    # The contract with the attention modules (Section 8.2): A and the code can't drift apart.
    config = ladder(name)
    with torch.device("meta"):
        attention = MultiHeadAttention(config.d_model, config.n_heads)
    per_token = config.n_layers * attention.cache_spec().bytes_per_token(2)
    assert kv_cache_bytes(spec(name), batch=1, context=1, bytes_per_element=2) == per_token


def test_mla_caches_arrive_later():
    mla = ModelSpec.model_validate(spec().model_dump() | {"attn": "mla"})
    with pytest.raises(NotImplementedError, match="M6"):
        kv_cache_bytes(mla, 1, 1, 2)


# --- training ------------------------------------------------------------------------------


def test_training_keeps_16_bytes_per_parameter():
    memory = training_memory(spec(), batch=4, seq_len=256)
    params = 820_352  # m1, embeddings included
    assert (memory.weights, memory.gradients, memory.optimizer) == (
        2 * params,
        2 * params,
        12 * params,
    )
    assert memory.weights + memory.gradients + memory.optimizer == 16 * params


def test_the_activation_estimate_by_hand():
    # 4 layers x 34 bytes x (4 x 256 tokens) x d_model 128
    assert training_memory(spec(), batch=4, seq_len=256).activations == 4 * 34 * 1024 * 128


def test_the_real_model_keeps_activations_in_proportion_to_the_tokens():
    # Twice the tokens, exactly twice the memory: SDPA never keeps a seq x seq attention table.
    assert saved_for_backward(ladder("m1"), 4, 256) == 2 * saved_for_backward(ladder("m1"), 4, 128)


def test_the_real_model_keeps_about_one_and_a_half_times_the_estimate():
    # The cost model's first measured prediction error, recorded rather than hidden: the
    # published 34 bytes is for GPT-style layers, and ours keep SwiGLU's wider intermediates
    # and RMSNorm's float32 copies. M3's calibration will fit the constant.
    estimate = training_memory(spec(), batch=4, seq_len=256).activations
    assert 1.4 < saved_for_backward(ladder("m1"), 4, 256) / estimate < 1.7


# --- generation and what fits ---------------------------------------------------------------


def test_generation_needs_the_weights_and_the_cache():
    memory = generation_memory(spec(), batch=8, context=256)
    assert memory.weights == 2 * 820_352  # float16 weights
    assert memory.kv_cache == kv_cache_bytes(spec(), 8, 256, 2)
    assert memory.gradients == memory.optimizer == memory.activations == 0


@pytest.mark.parametrize("hardware", list(HARDWARE))
def test_the_largest_batch_fits_and_one_more_does_not(hardware: str):
    device, model = HARDWARE[hardware], spec("m4")
    most = max_generation_batch(model, device, context=256)
    assert generation_memory(model, most, 256).total <= usable(device)
    assert generation_memory(model, most + 1, 256).total > usable(device)


@pytest.mark.parametrize("hardware", list(HARDWARE))
def test_the_longest_context_fits_and_one_more_does_not(hardware: str):
    device, model = HARDWARE[hardware], spec("m4")
    longest = max_generation_context(model, device, batch=512)
    assert generation_memory(model, 512, longest).total <= usable(device)
    assert generation_memory(model, 512, longest + 1).total > usable(device)


def test_m7_on_a_t4_can_sample_a_whole_rl_step():
    # Section 10.2's typical RL step: 64 prompts x 8 samples = 512 sequences at once.
    assert max_generation_batch(spec("m7"), HARDWARE["t4"], context=256) >= 512


def test_a_model_that_does_not_fit_at_all_gets_zero():
    tiny = HardwareSpec(name="tiny", memory_bytes=10**6, supports_bf16=False)  # 1 MB
    assert max_generation_batch(spec("m4"), tiny, context=256) == 0
    assert max_generation_context(spec("m4"), tiny, batch=1) == 0


def test_the_hardware_profiles():
    assert HARDWARE["t4"].memory_bytes == 16 * 10**9 and not HARDWARE["t4"].supports_bf16
    assert HARDWARE["l4"].memory_bytes == 24 * 10**9 and HARDWARE["l4"].supports_bf16
    assert HARDWARE["a100-40g"].memory_bytes == 40 * 10**9 and HARDWARE["a100-40g"].supports_bf16


def test_the_memory_model_never_imports_pytorch():
    code = "import sys, scalelab.costmodel.memory; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
