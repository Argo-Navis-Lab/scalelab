"""RL step FLOP tests: counts add up, the cost model's count of generation and of an update
equals what PyTorch itself measures, the chunking plan generate() follows, and the step's
phases."""

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from scalelab.core.config import AlgoConfig, ModelConfig, SamplingParams
from scalelab.costmodel.ops import NO_FLOPS, count_flops
from scalelab.costmodel.rlstep import decode_chunks
from scalelab.costmodel.specs import ModelSpec, WorkloadSpec
from scalelab.models.transformer import Transformer
from scalelab.rl.flops import logprob_flops, rl_step_flops, sampling_flops, training_flops
from scalelab.rl.rollout import RolloutBatch, generate
from scalelab.rl.trainer import rl_update
from scalelab.tasks.tokenizer import encode

SHORT = ModelConfig(name="short", n_layers=2, d_model=64, n_heads=1, context=32)
SPEC = ModelSpec.from_config(SHORT)
PROMPTS = [encode("<bos>12+34="), encode("<bos>123*456="), encode("<bos>56-78=")]  # 7, 9, 7
ADVANTAGES = torch.tensor([1.0, -1.0, 0.5, 0.0, 2.0, -0.5, -1.0, 1.0, 0.0])


def make() -> Transformer:
    """The same starting weights every time, without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(SHORT)


def roll(model: Transformer, batch_size: int = 512) -> RolloutBatch:
    """Three answers to each of PROMPTS: 9 rows."""
    generator = torch.Generator().manual_seed(0)
    return generate(model, PROMPTS, 3, SamplingParams(), generator, batch_size=batch_size)


def matmul_flops(counter: FlopCounterMode) -> int:
    """PyTorch's own count of the matrix multiplies it ran (aten.mm). On the CPU it doesn't
    count attention's kernel, so this is exactly the cost model's dense part."""
    return counter.get_flop_counts()["Global"].get(torch.ops.aten.mm, 0)


def test_flop_counts_add_up_part_by_part():
    a = count_flops(SPEC, WorkloadSpec(phase="prefill", batch=2, tokens=10))
    b = count_flops(SPEC, WorkloadSpec(phase="decode", batch=2, tokens=1, past=10))
    both = a + b
    assert (both.projections, both.attention) == (
        a.projections + b.projections,
        a.attention + b.attention,
    )
    assert (both.mlp, both.logits) == (a.mlp + b.mlp, a.logits + b.logits)
    assert NO_FLOPS + a == a  # where a sum starts


def test_generate_decodes_rows_with_equal_prompt_lengths_together():
    lengths = [7, 9, 7, 7, 9]
    assert decode_chunks(lengths, batch_size=512) == [[0, 2, 3], [1, 4]]
    assert decode_chunks(lengths, batch_size=2) == [[0, 2], [3], [1, 4]]


@pytest.mark.parametrize("batch_size", [512, 2])
def test_generations_flops_are_what_pytorch_measures(batch_size: int):
    model = make()
    with FlopCounterMode(display=False) as counter:
        batch = roll(model, batch_size)
    assert sampling_flops(SPEC, batch, batch_size).dense == matmul_flops(counter)


def test_an_updates_flops_are_what_pytorch_measures():
    model, reference = make(), make()
    batch = roll(model)
    algo = AlgoConfig(
        name="test",
        estimator="grpo",
        clip_low=0.2,
        clip_high=0.2,
        normalizer="seq_mean",
        kl_coef=0.04,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with FlopCounterMode(display=False) as counter:
        rl_update(
            model,
            optimizer,
            batch,
            ADVANTAGES,
            batch.response_mask,
            algo,
            temperature=1.0,
            epochs=2,
            micro_batch=2,
            reference=reference,
        )
    counted = 2 * logprob_flops(SPEC, batch).dense + training_flops(SPEC, batch, 2, 2).dense
    assert counted == matmul_flops(counter)  # old and reference log-probs, then 2 epochs


def test_decoding_rows_one_at_a_time_wastes_nothing():
    # In a chunk, rows that finished keep being computed until the longest is done.
    batch = roll(make())
    assert sampling_flops(SPEC, batch, 1).total < sampling_flops(SPEC, batch, 512).total


def test_a_steps_flops_add_up_by_phase():
    batch = roll(make())
    step = rl_step_flops(
        SPEC, batch, batch, batch_size=512, micro_batch=2, epochs=3, reference=True
    )
    assert step.sample == sampling_flops(SPEC, batch, 512).total
    assert step.logprobs == 2 * logprob_flops(SPEC, batch).total
    assert step.train == training_flops(SPEC, batch, 2, 3).total
    assert step.total == step.sample + step.logprobs + step.train
    alone = rl_step_flops(
        SPEC, batch, batch, batch_size=512, micro_batch=2, epochs=3, reference=False
    )
    assert alone.logprobs == step.logprobs // 2  # without a reference, one pass


def test_with_nothing_trained_only_sampling_counts():
    batch = roll(make())
    step = rl_step_flops(SPEC, batch, None, batch_size=512, micro_batch=2, epochs=3, reference=True)
    assert (step.logprobs, step.train) == (0, 0)
    assert step.total == step.sample
