"""The memory model: how many bytes a workload needs on the device (Section 5.4)."""

from scalelab.core.config import StrictModel
from scalelab.costmodel.specs import HardwareSpec, ModelSpec

# Mixed-precision Adam keeps, per parameter: 16-bit weights (2 bytes) and gradients (2), plus
# float32 master weights (4) and Adam's two running averages (4 + 4).
WEIGHT_BYTES, GRADIENT_BYTES, OPTIMIZER_BYTES = 2, 2, 12

# Korthikanti et al. (2022): a layer saves about 34 bytes per token per d_model for the
# backward pass, in 16-bit, once attention runs in a memory-efficient kernel (as SDPA does).
ACTIVATION_BYTES = 34

# Kept free for the CUDA context, PyTorch's allocator and temporary tensors. An assumption
# until M3 measures it.
RESERVED_FRACTION = 0.10


class MemoryEstimate(StrictModel):
    """Bytes on the device, by what they hold."""

    weights: int
    gradients: int
    optimizer: int  # Adam's master weights and running averages
    activations: int  # what the forward pass saves for the backward pass
    kv_cache: int  # keys and values of every token so far, when generating

    @property
    def total(self) -> int:
        return self.weights + self.gradients + self.optimizer + self.activations + self.kv_cache


def kv_cache_bytes(model: ModelSpec, batch: int, context: int, bytes_per_element: int) -> int:
    """M_KV = 2 * layers * n_kv_heads * head_dim * context * batch * bytes: a key and a value
    per token, per layer. The tests hold it equal to the attention modules' cache_spec()."""
    if model.attn == "mla":
        raise NotImplementedError("MLA caches a compressed latent instead; it arrives in M6")
    per_token = 2 * model.n_layers * model.n_kv_heads * model.head_dim * bytes_per_element
    return per_token * context * batch


def training_memory(model: ModelSpec, batch: int, seq_len: int) -> MemoryEstimate:
    """Training with mixed-precision Adam: 16 bytes per parameter, plus activations."""
    params = model.param_counts().total
    tokens = batch * seq_len
    return MemoryEstimate(
        weights=WEIGHT_BYTES * params,
        gradients=GRADIENT_BYTES * params,
        optimizer=OPTIMIZER_BYTES * params,
        activations=model.n_layers * ACTIVATION_BYTES * tokens * model.d_model,
        kv_cache=0,
    )


def generation_memory(
    model: ModelSpec, batch: int, context: int, bytes_per_element: int = 2
) -> MemoryEstimate:
    """Generating: the weights, and a KV cache for `context` tokens of `batch` sequences."""
    return MemoryEstimate(
        weights=bytes_per_element * model.param_counts().total,
        gradients=0,
        optimizer=0,
        activations=0,  # one token at a time: tiny next to the cache
        kv_cache=kv_cache_bytes(model, batch, context, bytes_per_element),
    )


def max_generation_batch(
    model: ModelSpec, hardware: HardwareSpec, context: int, bytes_per_element: int = 2
) -> int:
    """The most sequences of `context` tokens that can be generated at once, often the
    single most useful number for planning RL sampling. 0 if not even one fits."""
    free = _usable(hardware) - generation_memory(model, 1, 0, bytes_per_element).weights
    return max(0, free // kv_cache_bytes(model, 1, context, bytes_per_element))


def max_generation_context(
    model: ModelSpec, hardware: HardwareSpec, batch: int, bytes_per_element: int = 2
) -> int:
    """The longest context whose KV cache fits for `batch` sequences at once."""
    free = _usable(hardware) - generation_memory(model, 1, 0, bytes_per_element).weights
    return max(0, free // kv_cache_bytes(model, batch, 1, bytes_per_element))


def _usable(hardware: HardwareSpec) -> int:
    return int(hardware.memory_bytes * (1 - RESERVED_FRACTION))
