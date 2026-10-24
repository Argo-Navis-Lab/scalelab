"""FLOP counts and the op graph: the arithmetic a workload does and the memory it moves,
worked out from shapes alone."""

from typing import Literal

from scalelab.core.config import StrictModel
from scalelab.costmodel.specs import ModelSpec, WorkloadSpec


class FlopCounts(StrictModel):
    """Floating-point operations, by part of the model. Each multiply-add counts as 2."""

    projections: int  # attention's q, k, v and o matrices
    attention: int  # scoring each query against each key it may see, and mixing the values
    mlp: int
    logits: int  # the output head

    @property
    def dense(self) -> int:
        """Everything except attention: matrix multiplies whose cost per token is fixed."""
        return self.projections + self.mlp + self.logits

    @property
    def total(self) -> int:
        return self.dense + self.attention

    def __add__(self, other: "FlopCounts") -> "FlopCounts":
        """Two counts together, part by part, so `a + b` and sum(...) work."""
        return FlopCounts(
            projections=self.projections + other.projections,
            attention=self.attention + other.attention,
            mlp=self.mlp + other.mlp,
            logits=self.logits + other.logits,
        )


NO_FLOPS = FlopCounts(projections=0, attention=0, mlp=0, logits=0)  # where sums start


def count_flops(model: ModelSpec, workload: WorkloadSpec) -> FlopCounts:
    """The exact FLOPs of a workload: one forward pass, or forward and backward to train."""
    if model.attn in ("mla", "swa"):
        raise NotImplementedError(f"{model.attn} FLOPs arrive with that variant, in M6")
    b, n, past = workload.batch, workload.tokens, workload.past
    tokens = b * n  # new tokens across the batch
    # Query-key pairs: new token i (counting from 0) sees the past, the new tokens before it,
    # and itself. Causal attention never spends work on the pairs it masks out.
    pairs = b * (n * past + n * (n + 1) // 2)
    d, layers = model.d_model, model.n_layers
    q_width = model.n_heads * model.head_dim
    kv_width = model.n_kv_heads * model.head_dim
    ffn_matrices = 3 if model.ffn == "swiglu" else 2
    forward = FlopCounts(
        projections=layers * 2 * tokens * d * (2 * q_width + 2 * kv_width),
        attention=layers * 4 * pairs * q_width,  # 2 for scores + 2 for mixing, per dimension
        mlp=layers * 2 * tokens * ffn_matrices * d * model.d_ff,
        logits=2 * tokens * d * model.vocab,
    )
    if workload.phase != "train":
        return forward
    # The backward pass costs twice the forward; checkpointing reruns the forward once more.
    passes = 4 if workload.activation_ckpt else 3
    return FlopCounts(
        projections=passes * forward.projections,
        attention=passes * forward.attention,
        mlp=passes * forward.mlp,
        logits=passes * forward.logits,
    )


# --- the op graph (m3s1) ---------------------------------------------------------------------

# Every run so far is float32: 4 bytes a value. fp16 (m3s15) will need more than one number
# (autocast keeps the norms in float32), so there is no dtype choice yet.
VALUE_BYTES = 4
ID_BYTES = 8  # a token id (int64)

# gemm: activations times a weight matrix. gemv-decode: the same while decoding, one token per
# sequence, so reading the weights dominates. attention: the fused SDPA kernel. elementwise:
# ops that touch each value a few times (norms, RoPE, the MLP's activation, residual adds).
OpClass = Literal["gemm", "gemv-decode", "attention", "elementwise"]


class Op(StrictModel):
    """One operation of a workload, about one GPU kernel: the arithmetic it does and the bytes
    it moves. Timing (m3s2) asks which of the two takes longer on a device."""

    name: str  # its module in the model ("attn.q"), then ".grad_input" etc. for gradients
    layer: int  # numbered as telemetry does: 0 the embedding, 1..n_layers blocks, n + 1 head
    op_class: OpClass
    flops: int  # matrix multiplies only, as in count_flops: elementwise ops count 0
    bytes_read: int
    bytes_written: int

    @property
    def intensity(self) -> float:
        """FLOPs per byte moved. Below a device's ridge point (peak FLOP/s over memory
        bandwidth), memory sets the op's speed; above it, arithmetic does."""
        return self.flops / (self.bytes_read + self.bytes_written)


def matmul_op(name: str, layer: int, rows: int, inner: int, cols: int, op_class: OpClass) -> Op:
    """(rows × inner) @ (inner × cols): reads both matrices, writes the rows × cols result."""
    return Op(
        name=name,
        layer=layer,
        op_class=op_class,
        flops=2 * rows * inner * cols,
        bytes_read=VALUE_BYTES * (rows * inner + inner * cols),
        bytes_written=VALUE_BYTES * rows * cols,
    )


def grad_op(op: Op) -> Op:
    """A rule of thumb for the gradient of an op that isn't a matmul: it reads what the op read
    and the gradient of what it wrote, writes a gradient for everything it read, and does twice
    the FLOPs."""
    return Op(
        name=f"{op.name}.grad",
        layer=op.layer,
        op_class=op.op_class,
        flops=2 * op.flops,
        bytes_read=op.bytes_read + op.bytes_written,
        bytes_written=op.bytes_read,
    )


class _Ops:
    """A forward pass's ops as they are added, and the backward pass's that go with them."""

    def __init__(self) -> None:
        self.forward: list[Op] = []
        self.backward: list[Op] = []  # in forward order; they run in reverse

    def matmul(
        self, name: str, layer: int, rows: int, inner: int, cols: int, kind: OpClass
    ) -> None:
        """Activations (rows × inner) times a weight matrix (inner × cols)."""
        self.forward.append(matmul_op(name, layer, rows, inner, cols, kind))
        # The gradient goes back through two matmuls of the same size, so backward costs twice
        # the forward: the weight's (inputs transposed @ dY) and the input's (dY @ W transposed).
        self.backward.append(matmul_op(f"{name}.grad_weight", layer, inner, rows, cols, "gemm"))
        self.backward.append(matmul_op(f"{name}.grad_input", layer, rows, cols, inner, "gemm"))

    def other(
        self,
        name: str,
        layer: int,
        read: int,
        written: int,
        kind: OpClass = "elementwise",
        flops: int = 0,
    ) -> None:
        """Any other op: `read` bytes in, `written` bytes out."""
        op = Op(
            name=name,
            layer=layer,
            op_class=kind,
            flops=flops,
            bytes_read=read,
            bytes_written=written,
        )
        self.forward.append(op)
        self.backward.append(grad_op(op))


def op_graph(model: ModelSpec, workload: WorkloadSpec) -> list[Op]:
    """Every op of a workload in the order they run: the forward pass, then (to train) the
    backward pass, last layer first. Their FLOPs add up to count_flops' exactly."""
    if model.attn in ("mla", "swa"):
        raise NotImplementedError(f"{model.attn} ops arrive with that variant, in M6")
    s, b, n, past = VALUE_BYTES, workload.batch, workload.tokens, workload.past
    rows = b * n  # new tokens across the batch: the rows of every matmul
    keys = b * (past + n)  # tokens whose keys and values attention reads
    pairs = b * (n * past + n * (n + 1) // 2)  # query-key pairs, as in count_flops
    d, d_head, d_ff = model.d_model, model.head_dim, model.d_ff
    q_width, kv_width = model.n_heads * d_head, model.n_kv_heads * d_head
    x = s * rows * d  # bytes of the residual stream: one d_model vector per new token
    matmul: OpClass = "gemv-decode" if workload.phase == "decode" else "gemm"
    ops = _Ops()
    ops.other("embed", 0, read=x + ID_BYTES * rows, written=x)
    for layer in range(1, model.n_layers + 1):
        ops.other("attn_norm", layer, read=x + s * d, written=x)
        for part, width in (("q", q_width), ("k", kv_width)):
            ops.matmul(f"attn.{part}", layer, rows, d, width, matmul)
            if model.qk_norm:
                ops.other(
                    f"attn.{part}_norm",
                    layer,
                    read=s * (rows * width + d_head),
                    written=s * rows * width,
                )
            # RoPE also reads the cos and sin of each position's angles.
            ops.other(
                f"attn.{part}_rope",
                layer,
                read=s * (rows * width + 2 * n * d_head),
                written=s * rows * width,
            )
        ops.matmul("attn.v", layer, rows, d, kv_width, matmul)
        if workload.phase == "decode":  # the new keys and values go into the cache
            ops.other(
                "attn.cache", layer, read=s * 2 * rows * kv_width, written=s * 2 * rows * kv_width
            )
        # One fused kernel: every query, key and value read once, one output per query.
        ops.other(
            "attn.sdpa",
            layer,
            read=s * (rows * q_width + 2 * keys * kv_width),
            written=s * rows * q_width,
            kind="attention",
            flops=4 * pairs * q_width,
        )
        ops.matmul("attn.o", layer, rows, q_width, d, matmul)
        ops.other("attn_residual", layer, read=2 * x, written=x)
        ops.other("mlp_norm", layer, read=x + s * d, written=x)
        if model.ffn == "swiglu":
            ops.matmul("mlp.gate", layer, rows, d, d_ff, matmul)
        ops.matmul("mlp.up", layer, rows, d, d_ff, matmul)
        inputs = 2 if model.ffn == "swiglu" else 1  # silu(gate) * up, or gelu(up)
        ops.other("mlp.act", layer, read=s * inputs * rows * d_ff, written=s * rows * d_ff)
        ops.matmul("mlp.down", layer, rows, d_ff, d, matmul)
        ops.other("mlp_residual", layer, read=2 * x, written=x)
    last = model.n_layers + 1
    ops.other("norm", last, read=x + s * d, written=x)
    ops.matmul("head", last, rows, d, model.vocab, matmul)
    if workload.phase != "train":
        return ops.forward
    # Checkpointing reruns the forward pass during the backward one: count_flops' 4 passes.
    rerun = ops.forward if workload.activation_ckpt else []
    return ops.forward + rerun + ops.backward[::-1]
