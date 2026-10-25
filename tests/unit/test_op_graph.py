"""Op graph tests: the FLOPs are count_flops', bytes by hand, and decoding's intensity."""

from pathlib import Path

import pytest

from scalelab.core.config import load_config
from scalelab.costmodel.ops import Op, count_flops, op_graph
from scalelab.costmodel.specs import ModelSpec, Precision, WorkloadSpec

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
LADDER = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
WORKLOADS = {
    "prefill": WorkloadSpec(phase="prefill", batch=2, tokens=10),
    "decode-first": WorkloadSpec(phase="decode", batch=8, tokens=1),
    "decode-late": WorkloadSpec(phase="decode", batch=8, tokens=1, past=255),
    "train": WorkloadSpec(phase="train", batch=2, tokens=10),
    "train-checkpointed": WorkloadSpec(phase="train", batch=2, tokens=10, activation_ckpt=True),
}
BLOCK = [  # one block's ops, as the model runs them
    "attn_norm", "attn.q", "attn.q_rope", "attn.k", "attn.k_rope", "attn.v", "attn.sdpa",
    "attn.o", "attn_residual", "mlp_norm", "mlp.gate", "mlp.up", "mlp.act", "mlp.down",
    "mlp_residual",
]  # fmt: skip


def spec(name: str = "m1") -> ModelSpec:
    return ModelSpec.from_config(load_config(HELLO, [f"model=ladder_s/{name}"]).model)


def flops(ops: list[Op], *classes: str) -> int:
    return sum(op.flops for op in ops if not classes or op.op_class in classes)


def fp16(workload: WorkloadSpec) -> WorkloadSpec:
    return workload.model_copy(update={"precision": "fp16"})


# --- the FLOPs are count_flops', to the last one --------------------------------------------


@pytest.mark.parametrize("workload", WORKLOADS.values(), ids=WORKLOADS.keys())
@pytest.mark.parametrize("name", LADDER)
@pytest.mark.parametrize("precision", ["fp32", "fp16"])  # m3s18: casts do no FLOPs
def test_the_flops_are_count_flops(name: str, workload: WorkloadSpec, precision: Precision):
    ops = op_graph(spec(name), workload.model_copy(update={"precision": precision}))
    counts = count_flops(spec(name), workload)
    assert flops(ops) == counts.total
    assert flops(ops, "attention") == counts.attention


@pytest.mark.parametrize(
    "change",
    [
        {"ffn": "gelu"},
        {"qk_norm": True},
        {"attn": "gqa", "n_heads": 4, "head_dim": 32, "n_kv_heads": 1},
    ],
    ids=["gelu", "qk-norm", "gqa"],
)
@pytest.mark.parametrize("workload", WORKLOADS.values(), ids=WORKLOADS.keys())
def test_other_shapes_agree_too(change: dict[str, object], workload: WorkloadSpec):
    model = ModelSpec.model_validate(spec().model_dump() | change)
    counts, ops = count_flops(model, workload), op_graph(model, workload)
    assert flops(ops) == counts.total
    assert flops(ops, "attention") == counts.attention


# --- by hand --------------------------------------------------------------------------------


def test_m1_reading_two_prompts_of_ten_tokens_by_hand():
    ops = op_graph(spec(), WORKLOADS["prefill"])  # 20 new tokens: the rows of every matmul
    assert [op.name for op in ops] == ["embed", *(BLOCK * 4), "norm", "head"]
    assert [op.layer for op in ops] == [0, *(layer for layer in range(1, 5) for _ in BLOCK), 5, 5]
    block = {op.name: op for op in ops if op.layer == 1}
    assert block["attn.q"] == Op(
        name="attn.q",
        layer=1,
        op_class="gemm",
        flops=2 * 20 * 128 * 128,
        bytes_read=4 * (20 * 128 + 128 * 128),  # the 20 inputs and the weight matrix
        bytes_written=4 * 20 * 128,
    )
    sdpa, norm = block["attn.sdpa"], block["attn_norm"]
    assert (sdpa.bytes_read, sdpa.bytes_written) == (4 * (20 * 128 + 2 * 20 * 128), 4 * 20 * 128)
    assert (norm.flops, norm.bytes_read, norm.bytes_written) == (
        0,
        4 * (20 * 128 + 128),
        4 * 20 * 128,
    )


def test_decoding_reads_the_whole_cache_through_skinny_matmuls():
    late = op_graph(spec(), WORKLOADS["decode-late"])  # 8 sequences, 255 tokens in the cache
    matmuls = {op.op_class for op in late if op.flops > 0 and op.name != "attn.sdpa"}
    assert matmuls == {"gemv-decode"}
    sdpa = next(op for op in late if op.name == "attn.sdpa")
    assert sdpa.bytes_read == 4 * (8 * 128 + 2 * 8 * 256 * 128)  # 8 queries; 256 keys and values
    cache = next(op for op in late if op.name == "attn.cache")
    assert cache.bytes_written == 4 * 2 * 8 * 128  # the new token's key and value
    assert "attn.cache" not in {op.name for op in op_graph(spec(), WORKLOADS["prefill"])}


@pytest.mark.parametrize("batch", [1, 8, 64, 512])
@pytest.mark.parametrize("name", LADDER)
def test_decoding_does_under_half_a_flop_per_byte_per_sequence(name: str, batch: int):
    # Section 5.3: decoding B sequences does about B FLOPs per byte in 16-bit, B / 2 in
    # float32, since every weight is read once for only B tokens; attention less still.
    ops = op_graph(spec(name), WorkloadSpec(phase="decode", batch=batch, tokens=1, past=100))
    assert max(op.intensity for op in ops) < batch / 2


# --- training -------------------------------------------------------------------------------


def test_training_runs_the_gradients_last_layer_first():
    forward, training = op_graph(spec(), WORKLOADS["prefill"]), op_graph(spec(), WORKLOADS["train"])
    assert training[: len(forward)] == forward  # the same 2 x 10 tokens
    backward = training[len(forward) :]
    assert (backward[0].name, backward[-1].name) == ("head.grad_input", "embed.grad")
    assert [op.layer for op in backward] == sorted((op.layer for op in backward), reverse=True)
    grads = {op.name: op for op in backward if op.layer == 1}
    assert grads["attn.q.grad_input"].bytes_read == 4 * (20 * 128 + 128 * 128)  # dY and W
    assert grads["attn.q.grad_weight"].bytes_written == 4 * 128 * 128  # dW
    assert grads["attn.sdpa.grad"].flops == 2 * 4 * 110 * 128  # twice the forward's


def test_checkpointing_reruns_the_forward_pass():
    forward = op_graph(spec(), WORKLOADS["prefill"])
    plain = op_graph(spec(), WORKLOADS["train"])
    assert op_graph(spec(), WORKLOADS["train-checkpointed"]) == forward + plain


@pytest.mark.parametrize("attn", ["mla", "swa"])
def test_mla_and_sliding_window_arrive_later(attn: str):
    model = ModelSpec.model_validate(spec().model_dump() | {"attn": attn})
    with pytest.raises(NotImplementedError, match="M6"):
        op_graph(model, WORKLOADS["prefill"])


# --- fp16 under autocast (m3s18) ------------------------------------------------------------


def test_m1_reading_two_prompts_in_fp16_by_hand():
    ops = op_graph(spec(), fp16(WORKLOADS["prefill"]))  # 20 new tokens
    block = {op.name: op for op in ops if op.layer == 1}
    assert list(block) == [
        "attn_norm", "cast attn.q.weight", "cast attn.q", "attn.q", "attn.q_rope",
        "cast attn.k.weight", "cast attn.k", "attn.k", "attn.k_rope", "cast attn.v.weight",
        "cast attn.v", "attn.v", "attn.sdpa", "cast attn.o.weight", "attn.o", "attn_residual",
        "mlp_norm", "cast mlp.gate.weight", "cast mlp.gate", "mlp.gate", "cast mlp.up.weight",
        "cast mlp.up", "mlp.up", "mlp.act", "cast mlp.down.weight", "mlp.down", "mlp_residual",
    ]  # fmt: skip
    assert block["attn.q"] == Op(
        name="attn.q",
        layer=1,
        op_class="gemm",
        flops=2 * 20 * 128 * 128,
        bytes_read=2 * (20 * 128 + 128 * 128),  # float16 inputs and weight
        bytes_written=2 * 20 * 128,
        dtype="fp16",
    )
    cast, weight = block["cast attn.q"], block["cast attn.q.weight"]
    assert (cast.bytes_read, cast.bytes_written) == (4 * 20 * 128, 2 * 20 * 128)
    assert (weight.bytes_read, weight.bytes_written) == (4 * 128 * 128, 2 * 128 * 128)
    residual = block["attn_residual"]  # the float32 stream plus attention's float16 output
    assert (residual.bytes_read, residual.bytes_written) == (6 * 20 * 128, 4 * 20 * 128)
    plain = {op.name: op for op in op_graph(spec(), WORKLOADS["prefill"]) if op.layer == 1}
    assert block["attn_norm"] == plain["attn_norm"]  # the norms stay float32
    assert block["attn.sdpa"].dtype == "fp16"


def test_fp16_decoding_reuses_the_weights_its_prefill_cast():
    late = op_graph(spec(), fp16(WORKLOADS["decode-late"]))
    casts = [op.name for op in late if op.name.startswith("cast ")]
    assert len(casts) == 4 * 5 + 1  # q, k, v, gate and up read a norm's output; so does head
    assert not [name for name in casts if name.endswith(".weight")]
    cache = next(op for op in late if op.name == "attn.cache")
    assert cache.bytes_written == 2 * 2 * 8 * 128  # the cache holds float16


def test_fp16_training_casts_each_gradient_back_to_float32():
    ops = op_graph(spec(), fp16(WORKLOADS["train"]))
    names = [op.name for op in ops]
    first = names.index("attn.q.grad_input")  # the last layer's
    assert names[first : first + 4] == [
        "attn.q.grad_input", "attn.q.grad_weight", "cast attn.q.grad", "cast attn.q.weight.grad",
    ]  # fmt: skip
    weight = ops[first + 3]  # dW, into the float32 .grad
    assert (weight.bytes_read, weight.bytes_written) == (2 * 128 * 128, 4 * 128 * 128)
    assert {op.dtype for op in ops if op.flops > 0} == {"fp16"}  # every matmul and attention


@pytest.mark.parametrize("workload", WORKLOADS.values(), ids=WORKLOADS.keys())
def test_fp16_matmuls_and_attention_move_half_the_bytes(workload: WorkloadSpec):
    def moved(ops: list[Op]) -> int:
        return sum(op.bytes_read + op.bytes_written for op in ops if op.flops > 0)

    model = spec("m3")
    assert 2 * moved(op_graph(model, fp16(workload))) == moved(op_graph(model, workload))
