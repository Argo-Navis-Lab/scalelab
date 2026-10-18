"""Transformer tests: shapes, no peeking ahead, the ladder's sizes, starting weights, learning."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from scalelab.core.config import ModelConfig, load_config
from scalelab.models.attention.mha import MultiHeadAttention
from scalelab.models.rope import RotaryEmbedding
from scalelab.models.transformer import Block, Transformer
from scalelab.tasks.arithmetic import ArithmeticTask
from scalelab.tasks.tokenizer import EOS, PAD, VOCAB_SIZE, encode

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
M1 = ModelConfig(name="m1", n_layers=4, d_model=128, n_heads=2)
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1)


def make(config: ModelConfig = M1) -> Transformer:
    """The same starting weights every time, without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(config)


def random_tokens(batch: int, seq: int, seed: int = 0) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (batch, seq), generator=generator)


def ladder(name: str) -> ModelConfig:
    return load_config(HELLO, [f"model=ladder_s/{name}"]).model


# --- what goes in and out ------------------------------------------------------------------


def test_one_score_per_vocabulary_token_at_every_position():
    assert make()(random_tokens(2, 10)).shape == (2, 10, VOCAB_SIZE)


def test_a_position_never_sees_the_tokens_after_it():
    model, tokens = make(), random_tokens(2, 10)
    changed = tokens.clone()
    changed[:, 6:] = random_tokens(2, 4, seed=1)
    assert torch.equal(model(tokens)[:, :6], model(changed)[:, :6])


def test_an_untrained_model_guesses_evenly():
    # Even guessing among 64 tokens gives a cross-entropy loss of ln(64) = 4.16.
    tokens = random_tokens(4, 30)
    loss = F.cross_entropy(make()(tokens).flatten(0, 1), tokens.flatten())
    assert abs(loss.item() - math.log(VOCAB_SIZE)) < 0.1


def test_sequences_longer_than_the_context_are_refused():
    with pytest.raises(ValueError, match="longer than the context of 256"):
        make(TINY)(random_tokens(1, 257))


def test_other_attention_variants_arrive_later():
    with pytest.raises(NotImplementedError, match="M6"):
        Transformer(ModelConfig(name="g", n_layers=1, d_model=64, n_heads=1, attn="gqa"))


# --- the ladder's sizes --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "d_ff", "n_params"),
    [
        ("m1", 352, 803_968),
        ("m2", 704, 4_820_224),
        ("m3", 1024, 14_162_304),
        ("m4", 1376, 31_631_872),
        ("m5", 1728, 59_489_920),
        ("m6", 2048, 99_112_704),
        ("m7", 2752, 202_408_960),
    ],
)
def test_the_ladder(name: str, d_ff: int, n_params: int):
    config = ladder(name)
    with torch.device("meta"):  # shapes without numbers: even m7 takes no memory
        model = Transformer(config)
    assert config.d_ff == d_ff
    assert model.num_parameters() == n_params  # golden: the cost model must match these
    assert abs(n_params / (12 * config.n_layers * config.d_model**2) - 1) < 0.03  # ≈ 12·L·d²
    assert model.num_parameters(embeddings=True) - n_params == 2 * VOCAB_SIZE * config.d_model


# --- starting weights ----------------------------------------------------------------------


def test_starting_weights():
    model = make()
    residual_std = 0.02 / math.sqrt(2 * M1.n_layers)
    for name, weight in model.named_parameters():
        if name.endswith("norm.weight"):
            assert torch.equal(weight, torch.ones_like(weight)), name
        elif name.endswith(("attn.o.weight", "mlp.down.weight")):
            assert abs(weight.std().item() / residual_std - 1) < 0.05, name
        else:
            assert abs(weight.std().item() / 0.02 - 1) < 0.05, name


def test_the_same_seed_gives_the_same_model():
    first, second = make().state_dict(), make().state_dict()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_a_block_with_silent_branches_passes_its_input_through():
    # With attention's and the MLP's output layers at zero, both branches add nothing, so a
    # block with residual connections returns its input exactly.
    block = Block(M1)
    assert isinstance(block.attn, MultiHeadAttention)  # the variant with an "o" layer
    with torch.no_grad():
        block.attn.o.weight.zero_()
        block.mlp.down.weight.zero_()
    x = torch.randn(2, 10, 128, generator=torch.Generator().manual_seed(0))
    rope = RotaryEmbedding(M1.head_dim, M1.context)
    assert torch.equal(block(x, rope, torch.arange(10)), x)


# --- learning ------------------------------------------------------------------------------


def test_gradients_reach_every_parameter():
    model, tokens = make(TINY), random_tokens(2, 10)
    F.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten()).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_a_tiny_model_can_learn_four_worked_solutions_by_heart():
    task = ArithmeticTask()
    problems = task.sample(np.random.default_rng(0), 4, difficulty=3)
    rows = [task.format_prompt(p) + encode(task.worked_solution(p)) + [EOS] for p in problems]
    width = max(len(row) for row in rows)
    tokens = torch.tensor([row + [PAD] * (width - len(row)) for row in rows])
    inputs, targets = tokens[:, :-1], tokens[:, 1:]  # predict each next token
    model = make(TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses: list[float] = []
    for _ in range(60):
        loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[0] > 3.5  # it starts out guessing
    assert losses[-1] < 0.5  # and ends up knowing them
