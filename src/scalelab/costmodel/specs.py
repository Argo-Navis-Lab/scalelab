"""What the cost model knows about a model: its shape, written down without building it.

Everything here is plain arithmetic on a few numbers, so the cost model can answer "how big,
how slow, how much memory" before anything runs, and without PyTorch.
"""

from typing import Literal

from pydantic import Field, model_validator

from scalelab.core.config import ModelConfig, StrictModel
from scalelab.tasks.tokenizer import VOCAB_SIZE


class ParamCounts(StrictModel):
    """How many parameters each part of a model has."""

    embedding: int  # the token embedding table
    attention: int  # every layer's q, k, v and o (and QK-norm weights, when on)
    mlp: int  # every layer's MLP
    norms: int  # every RMSNorm weight: two per layer, and the final one
    head: int  # the output head (0 when it shares the embedding's weights)

    @property
    def non_embedding(self) -> int:
        """N, the size the scaling fits use."""
        return self.attention + self.mlp + self.norms

    @property
    def total(self) -> int:
        return self.embedding + self.non_embedding + self.head


class WorkloadSpec(StrictModel):
    """What the model is asked to do, in a shape the cost model can count (Section 5.1).

    Every phase processes `tokens` new tokens per sequence after `past` tokens already in the
    KV cache: training and reading a prompt start from an empty cache, and generating adds
    one token at a time.
    """

    phase: Literal["train", "prefill", "decode"]
    batch: int = Field(gt=0)  # sequences processed together
    tokens: int = Field(gt=0)  # new tokens per sequence
    past: int = Field(default=0, ge=0)  # tokens already in the KV cache (decode only)
    activation_ckpt: bool = False  # training only: redo the forward pass to save memory

    @model_validator(mode="after")
    def check_phase(self) -> "WorkloadSpec":
        if self.phase == "decode" and self.tokens != 1:
            raise ValueError("decode adds exactly one token per sequence")
        if self.phase != "decode" and self.past != 0:
            raise ValueError(f"{self.phase} starts from an empty cache, so past must be 0")
        if self.activation_ckpt and self.phase != "train":
            raise ValueError("activation checkpointing only applies to training")
        return self


class ModelSpec(StrictModel):
    """A model's shape, as the cost model sees it (blueprint Section 5.1)."""

    n_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_kv_heads: int = Field(gt=0)  # heads with their own keys and values: n_heads for MHA
    head_dim: int = Field(gt=0)
    attn: Literal["mha", "gqa", "mla", "swa"]
    qk_norm: bool = False
    ffn: Literal["swiglu", "gelu"]
    d_ff: int = Field(gt=0)
    vocab: int = Field(gt=0)
    tied_embeddings: bool

    @model_validator(mode="after")
    def check_kv_heads(self) -> "ModelSpec":
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a multiple of n_kv_heads ({self.n_kv_heads})"
            )
        return self

    @classmethod
    def from_config(cls, config: ModelConfig) -> "ModelSpec":
        """The spec of a Track S transformer: MHA, SwiGLU, separate embedding and head."""
        return cls(
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_heads,
            head_dim=config.head_dim,
            attn=config.attn,
            qk_norm=config.qk_norm,
            ffn="swiglu",
            d_ff=config.d_ff,
            vocab=VOCAB_SIZE,
            tied_embeddings=False,
        )

    def param_counts(self) -> ParamCounts:
        """Exact parameter counts, worked out from the shape alone."""
        if self.attn == "mla":
            raise NotImplementedError("MLA's parameters arrive with MLA itself, in M6")
        d, layers = self.d_model, self.n_layers
        q_and_o = 2 * d * self.n_heads * self.head_dim  # queries in, results out
        k_and_v = 2 * d * self.n_kv_heads * self.head_dim  # fewer of these with GQA
        qk_norm = 2 * self.head_dim if self.qk_norm else 0
        matrices = 3 if self.ffn == "swiglu" else 2  # gate, up and down; or just up and down
        return ParamCounts(
            embedding=self.vocab * d,
            attention=layers * (q_and_o + k_and_v + qk_norm),
            mlp=layers * matrices * d * self.d_ff,
            norms=(2 * layers + 1) * d,
            head=0 if self.tied_embeddings else d * self.vocab,
        )
