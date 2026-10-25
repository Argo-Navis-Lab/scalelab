"""Trainer tests. Log-probs: one forward pass agrees with what the sampler recorded (at the
same temperature only), nothing leaks outside the answers, and gradients flow. Updates:
micro-batches add up to the whole batch, clipping and the KL start at 0 and grow, the
gradient is clipped, and nothing to learn changes nothing. The padding cut (m3s16) changes
only how much padding is computed."""

import copy

import pytest
import torch

from scalelab.core.config import AlgoConfig, ModelConfig, Normalizer, SamplingParams
from scalelab.models.transformer import Transformer
from scalelab.rl.rollout import RolloutBatch, generate
from scalelab.rl.trainer import UpdateMetrics, answer_logprobs, rl_update
from scalelab.tasks.tokenizer import encode
from scalelab.telemetry.hooks import ForwardEvent, Hook

SHORT = ModelConfig(name="short", n_layers=2, d_model=64, n_heads=1, context=32)
PROMPTS = [encode("<bos>12+34="), encode("<bos>123*456="), encode("<bos>56-78=")]  # 7, 9, 7
# Made-up advantages for the 9 rows that roll() gives: what matters here is the training.
ADVANTAGES = torch.tensor([1.0, -1.0, 0.5, 0.0, 2.0, -0.5, -1.0, 1.0, 0.0])


def algo(normalizer: Normalizer = "seq_mean", kl_coef: float = 0.0) -> AlgoConfig:
    return AlgoConfig(
        name="test",
        estimator="grpo",
        clip_low=0.2,
        clip_high=0.2,
        normalizer=normalizer,
        normalizer_constant=32,
        kl_coef=kl_coef,
    )


def make() -> Transformer:
    """The same starting weights every time, without disturbing PyTorch's global random state."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(SHORT)


def roll(model: Transformer, temperature: float) -> RolloutBatch:
    """Three answers to each of PROMPTS (9 rows; some end at <eos>, some fill the context)."""
    params = SamplingParams(temperature=temperature)
    return generate(model, PROMPTS, 3, params, torch.Generator().manual_seed(0))


def one_row(batch: RolloutBatch, row: int) -> RolloutBatch:
    """Row `row` of a batch on its own, without the padding the other rows needed."""
    end = int(batch.lengths[row])
    return RolloutBatch(
        tokens=batch.tokens[row : row + 1, :end],
        prompt_lengths=batch.prompt_lengths[row : row + 1],
        lengths=batch.lengths[row : row + 1],
        finish_reasons=batch.finish_reasons[row : row + 1],
        sampler_logprobs=batch.sampler_logprobs[row : row + 1, :end],
        group_ids=batch.group_ids[row : row + 1],
        policy_version=batch.policy_version,
    )


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_the_trainer_agrees_with_the_sampler(temperature: float):
    # The sampling test of the blueprint's Section 11, end to end: the trainer's forward pass
    # and the sampler's cached decoding give the same log-probs, up to float rounding.
    model = make()
    batch = roll(model, temperature)
    with torch.no_grad():
        trainer = answer_logprobs(model, batch, temperature)
    torch.testing.assert_close(trainer, batch.sampler_logprobs)


def test_a_temperature_left_out_shows_up_at_once():
    # Section 7.4's "temperature mismatch" bug: sampled at 0.7, recomputed at 1.
    model = make()
    batch = roll(model, 0.7)
    with torch.no_grad():
        gap = (answer_logprobs(model, batch, 1.0) - batch.sampler_logprobs).abs().max()
    assert gap > 0.1  # against about 5e-7 when the temperatures agree


def test_only_the_answers_have_log_probs():
    model = make()
    batch = roll(model, 1.0)
    with torch.no_grad():
        logp = answer_logprobs(model, batch, 1.0)
    assert torch.all(logp.isfinite())  # PAD's -inf never leaks out
    assert torch.all(logp[batch.response_mask] < 0)
    assert torch.all(logp[~batch.response_mask] == 0)


def test_other_rows_and_padding_change_nothing():
    model = make()
    batch = roll(model, 1.0)
    with torch.no_grad():
        together = answer_logprobs(model, batch, 1.0)
        for row in range(len(batch.finish_reasons)):
            alone = answer_logprobs(model, one_row(batch, row), 1.0)
            torch.testing.assert_close(alone[0], together[row, : alone.shape[1]])


def test_gradients_reach_the_weights():
    model = make()
    answer_logprobs(model, roll(model, 1.0), 1.0).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.all(parameter.grad.isfinite()), name
    assert model.head.weight.grad is not None and model.head.weight.grad.abs().sum() > 0


# --- updates --------------------------------------------------------------------------------


@pytest.mark.parametrize("normalizer", ["seq_mean", "token_mean", "constant"])
def test_micro_batches_add_up_to_the_whole_batch(normalizer: Normalizer):
    whole, parts = make(), make()
    batch = roll(whole, 1.0)
    trained: list[Transformer] = []
    for model, micro_batch in ((whole, 9), (parts, 2)):  # all 9 rows at once, or 2 at a time
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        rl_update(
            model,
            optimizer,
            batch,
            ADVANTAGES,
            batch.response_mask,
            algo(normalizer),
            temperature=1.0,
            micro_batch=micro_batch,
            grad_clip=1e9,  # no clipping: the gradients themselves must match, size and all
        )
        trained.append(model)
    for a, b in zip(trained[0].parameters(), trained[1].parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_updates_raise_the_objective_and_clipping_grows_from_0():
    model = make()
    batch = roll(model, 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    metrics = rl_update(
        model,
        optimizer,
        batch,
        ADVANTAGES,
        batch.response_mask,
        algo(),
        temperature=1.0,
        epochs=3,
    )
    assert len(metrics) == 3
    assert metrics[0].clip_fraction == 0  # the first update starts where the batch was drawn
    assert metrics[2].clip_fraction > 0
    assert metrics[2].loss < metrics[0].loss  # the loss is minus the objective


def test_the_kl_to_the_reference_starts_at_0_and_grows():
    model = make()
    reference = copy.deepcopy(model)  # the warm start, frozen
    batch = roll(model, 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    metrics = rl_update(
        model,
        optimizer,
        batch,
        ADVANTAGES,
        batch.response_mask,
        algo(kl_coef=0.04),
        temperature=1.0,
        epochs=3,
        reference=reference,
    )
    assert metrics[0].kl < 1e-6
    assert metrics[2].kl > metrics[0].kl


def test_a_kl_penalty_needs_the_reference_model():
    model = make()
    batch = roll(model, 1.0)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="needs the reference model"):
        rl_update(
            model,
            optimizer,
            batch,
            ADVANTAGES,
            batch.response_mask,
            algo(kl_coef=0.04),
            temperature=1.0,
        )


def test_the_gradient_is_clipped():
    model = make()
    start = [p.detach().clone() for p in model.parameters()]
    batch = roll(model, 1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)  # the step is the gradient itself
    (metrics,) = rl_update(
        model,
        optimizer,
        batch,
        ADVANTAGES,
        batch.response_mask,
        algo(),
        temperature=1.0,
        grad_clip=1e-3,
    )
    pairs = zip(model.parameters(), start, strict=True)
    moved = torch.cat([(p.detach() - s).flatten() for p, s in pairs])  # what the step changed
    assert metrics.grad_norm > 1e-3  # the gradient was bigger than the limit...
    assert moved.norm().item() == pytest.approx(1e-3, rel=1e-3)  # ...and cut down to it


@pytest.mark.parametrize("normalizer", ["seq_mean", "token_mean", "constant"])
def test_nothing_to_learn_changes_nothing(normalizer: Normalizer):
    model = make()
    start = [p.detach().clone() for p in model.parameters()]
    batch = roll(model, 1.0)
    nothing = torch.zeros_like(batch.response_mask)  # every token left out of the loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    # m2s29: "constant" divides by a number of rows, never 0, and so it used to go on and
    # divide the clipped tokens by 0 tokens.
    mask, settings = nothing, algo(normalizer)
    assert rl_update(model, optimizer, batch, ADVANTAGES, mask, settings, temperature=1.0) == []
    for p, s in zip(model.parameters(), start, strict=True):
        assert torch.equal(p, s)


# --- m3s16: the padding cut -----------------------------------------------------------------


class Lengths(Hook):
    """Each micro-batch's row lengths, in the order the update ran them."""

    def __init__(self) -> None:
        self.parts: list[list[int]] = []

    def on_forward(self, event: ForwardEvent) -> None:
        self.parts.append(event.batch.lengths.tolist())


def test_log_probs_by_micro_batch_are_the_same_numbers():
    model = make()
    batch = roll(model, 1.0)
    with torch.no_grad():
        whole = answer_logprobs(model, batch, 1.0)
        parts = answer_logprobs(model, batch, 1.0, micro_batch=2)
    assert parts.shape == whole.shape
    torch.testing.assert_close(parts, whole)


@pytest.mark.parametrize("normalizer", ["seq_mean", "token_mean", "constant"])
def test_the_padding_cut_makes_the_same_update(normalizer: Normalizer):
    # Shortest first changes which rows share a micro-batch, never the batch's gradient.
    batch = roll(make(), 1.0)
    trained: list[Transformer] = []
    results: list[list[UpdateMetrics]] = []
    for padding_cut in (False, True):
        model = make()
        results.append(
            rl_update(
                model,
                torch.optim.SGD(model.parameters(), lr=0.1),
                batch,
                ADVANTAGES,
                batch.response_mask,
                algo(normalizer, kl_coef=0.04),
                temperature=1.0,
                micro_batch=3,
                grad_clip=1e9,  # no clipping: the gradients themselves must match
                reference=make(),
                padding_cut=padding_cut,
            )
        )
        trained.append(model)
    for a, b in zip(trained[0].parameters(), trained[1].parameters(), strict=True):
        torch.testing.assert_close(a, b)
    (off,), (on,) = results
    assert on.loss == pytest.approx(off.loss, abs=1e-6)
    assert on.grad_norm == pytest.approx(off.grad_norm, rel=1e-5)
    assert on.kl == pytest.approx(off.kl, abs=1e-6)


def test_the_padding_cut_runs_the_shortest_rows_first():
    batch = roll(make(), 1.0)
    computed: dict[bool, int] = {}  # positions the micro-batches computed, padding and all
    for padding_cut in (False, True):
        model, lengths = make(), Lengths()
        rl_update(
            model,
            torch.optim.SGD(model.parameters(), lr=0.1),
            batch,
            ADVANTAGES,
            batch.response_mask,
            algo(),
            temperature=1.0,
            micro_batch=3,
            hooks=[lengths],
            padding_cut=padding_cut,
        )
        rows = [length for part in lengths.parts for length in part]
        assert rows == (sorted(rows) if padding_cut else batch.lengths.tolist())
        computed[padding_cut] = sum(len(part) * max(part) for part in lengths.parts)
    assert computed[True] <= computed[False]  # micro-batches of one size: sorted pads least
