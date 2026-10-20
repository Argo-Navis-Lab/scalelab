"""FLOP counts: how much arithmetic a workload costs, worked out from shapes alone (Section 5.2)."""

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
