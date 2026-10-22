"""Collector tests: the rollout, policy, optimization and system metrics by hand, the entropy
estimate against the exact entropy, the policy metrics over real updates against the
trainer's own, and a real run recording every metric at every step. Also the loop's new
order: a step ends before its checkpoint, and the hooks hear of a checkpoint before it is
saved. Then (m2s30) the detailed collectors: their cadence, the layers, the layer and Adam
metrics by hand, the steps they skip, and a real run measuring every layer when they are due.
Then (m2s31) the activation collector: its metrics by hand, the updates it skips, and a run."""

import copy
import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import yaml

from scalelab.core.checkpoint import CheckpointPolicy, latest_checkpoint
from scalelab.core.config import AlgoConfig, ModelConfig, RLConfig, SamplingParams
from scalelab.core.determinism import seed_everything
from scalelab.models.transformer import Block, Transformer
from scalelab.rl.flops import StepFlops
from scalelab.rl.loop import RLStepMetrics, rl_train
from scalelab.rl.rollout import RolloutBatch, generate
from scalelab.rl.sampling import log_probs
from scalelab.rl.trainer import UpdateMetrics, rl_update
from scalelab.tasks.arithmetic import ArithmeticTask
from scalelab.tasks.tokenizer import EOS, PAD, encode
from scalelab.telemetry.collectors import (
    ADAM_METRICS,
    ATTENTION_METRICS,
    OUTPUT_METRICS,
    STREAM_METRICS,
    ActivationCollector,
    AdamCollector,
    Cadence,
    LayerCollector,
    OptimizationCollector,
    PolicyCollector,
    RolloutCollector,
    SystemCollector,
    layers,
    make_collectors,
)
from scalelab.telemetry.hooks import (
    CheckpointEvent,
    ForwardEvent,
    Hook,
    OptimizerStepEvent,
    RolloutEvent,
    StepEndEvent,
)
from scalelab.telemetry.sink import MetricsSink, read_metrics

ALGOS = Path(__file__).resolve().parents[2] / "configs" / "algos"
TINY = ModelConfig(name="tiny", n_layers=2, d_model=64, n_heads=1, context=32)
TASK = ArithmeticTask()
# Two groups of two answers: one right and one wrong; both right, the last one cut off by the
# end of the context before its <eos>. Answers of 4, 4, 5 and 4 tokens.
GROUPS = [("<bos>12+34=", ["#46<eos>", "#47<eos>"]), ("<bos>56-78=", ["#-22<eos>", "#-22"])]
REWARDS = torch.tensor([1.0, 0.0, 1.0, 1.0])
ADVANTAGES = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25])  # made up
# Every metric the collectors record, without its "rollout/", "policy/", "optim/" or "system/".
ROLLOUT = {"reward_mean", "reward_std", "length_mean", "length_p95", "length_max", "truncated"}
ANSWERS = {"entropy", "flat_groups"}  # the policy metrics read from the answers
UPDATES = {"ratio_mean", "ratio_max", "clip_fraction", "kl_old", "advantage_mean", "advantage_std"}
OPTIMIZATION = {"loss", "grad_norm", "grad_clipped"}
PHASES = ("data", "generate", "score", "logprobs", "train")  # a step's, when it updates
SYSTEM = {f"{phase}_s" for phase in (*PHASES, "step")} | {
    "sample_tokens_per_s",
    "train_tokens_per_s",
    "sample_flops",
    "logprobs_flops",
    "train_flops",
}  # and on a GPU (not here), peak_memory


def batch_of(groups: list[tuple[str, list[str]]]) -> RolloutBatch:
    """A RolloutBatch built by hand. Each row's sampler log-probs are -0.1 * (row + 1) on its
    answer tokens; an answer without <eos> ran out of context."""
    rows = [
        (group, encode(prompt), encode(text))
        for group, (prompt, texts) in enumerate(groups)
        for text in texts
    ]
    lengths = [len(prompt) + len(answer) for _, prompt, answer in rows]
    tokens = torch.full((len(rows), max(lengths)), PAD)
    sampler_logprobs = torch.zeros(len(rows), max(lengths))
    for row, (_, prompt, answer) in enumerate(rows):
        tokens[row, : lengths[row]] = torch.tensor(prompt + answer)
        sampler_logprobs[row, len(prompt) : lengths[row]] = -0.1 * (row + 1)
    return RolloutBatch(
        tokens=tokens,
        prompt_lengths=torch.tensor([len(prompt) for _, prompt, _ in rows]),
        lengths=torch.tensor(lengths),
        finish_reasons=["eos" if answer[-1] == EOS else "length" for _, _, answer in rows],
        sampler_logprobs=sampler_logprobs,
        group_ids=torch.tensor([group for group, _, _ in rows]),
        policy_version=0,
    )


def found(sink: MetricsSink, home: Path, step: int = 1) -> dict[str, float]:
    """What the sink holds for `step`, by metric name."""
    sink.flush()
    return {row.metric: row.value for row in read_metrics(home, sink.run_id) if row.step == step}


def ended(
    step: int, phases: dict[str, float] | None = None, peak_memory: int | None = None
) -> StepEndEvent:
    """The end of a step of `phases` (by default none), 8 seconds long, that cost 300 FLOPs
    to sample, 20 to work out log-probs and 100 to train."""
    return StepEndEvent(
        step=step,
        seconds=8.0,
        phases=phases or {},
        flops=StepFlops(sample=300, logprobs=20, train=100),
        peak_memory=peak_memory,
    )


def preset(name: str) -> AlgoConfig:
    return AlgoConfig.model_validate(yaml.safe_load((ALGOS / f"{name}.yaml").read_text()))


def make() -> Transformer:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return Transformer(TINY)


def roll(model: Transformer, problems: int, seed: int = 0) -> RolloutBatch:
    """2 answers to each of `problems` two-digit problems."""
    drawn = TASK.sample(np.random.default_rng(seed), problems, 2)
    prompts = [TASK.format_prompt(problem) for problem in drawn]
    return generate(model, prompts, 2, SamplingParams(), torch.Generator().manual_seed(seed))


def update(model: Transformer, hooks: list[Hook], epochs: int) -> list[UpdateMetrics]:
    """rl_update with GRPO's settings (a KL penalty, so a reference) on 8 rolled-out rows, in
    micro-batches of 4, with Adam at a high learning rate so that the ratio moves."""
    batch = roll(model, 4)
    return rl_update(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-2),
        batch,
        ADVANTAGES,
        batch.response_mask,
        preset("grpo"),
        temperature=1.0,
        epochs=epochs,
        micro_batch=4,
        reference=copy.deepcopy(model),
        hooks=hooks,
        step=1,
    )


def train(algo: str, hooks: list[Hook], home: Path) -> list[RLStepMetrics]:
    """3 RL steps from a tiny untrained model, 8 problems x 8 answers each, saving a
    checkpoint at steps 2 and 3."""
    seeds = seed_everything(0, deterministic=True)
    config = RLConfig(
        from_run="0123456789abcdef-s0-01234567", flops=1e15, max_steps=3, prompts_per_step=8
    )
    model = Transformer(TINY)
    return rl_train(
        model,
        TASK,
        [2],
        preset(algo),
        config,
        np.random.default_rng(seeds.data),
        torch.Generator().manual_seed(seeds.sampling),
        home / "ckpt",
        home,
        policy=CheckpointPolicy(every_steps=2),
        reference=copy.deepcopy(model),
        hooks=hooks,
    )


# --- the rollout collector ------------------------------------------------------------------


def test_the_rollout_metrics_by_hand(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    RolloutCollector(sink).on_rollout(RolloutEvent(step=1, batch=batch_of(GROUPS), rewards=REWARDS))
    assert found(sink, tmp_path) == pytest.approx(
        {
            "rollout/reward_mean": 0.75,
            "rollout/reward_std": math.sqrt(0.75 * 0.25),  # 3 right of 4
            "rollout/length_mean": 4.25,
            "rollout/length_p95": 4.85,  # of 4, 4, 4, 5: 85% of the way from the 3rd to the 4th
            "rollout/length_max": 5.0,
            "rollout/truncated": 0.25,
        },
        abs=1e-6,
    )


# --- the policy collector -------------------------------------------------------------------


def test_the_policy_metrics_of_the_answers_by_hand(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = PolicyCollector(sink, preset("grpo"))
    collector.on_rollout(RolloutEvent(step=1, batch=batch_of(GROUPS), rewards=REWARDS))
    collector.on_step_end(ended(step=1))  # no update: nothing more to record
    assert found(sink, tmp_path) == pytest.approx(
        {
            "policy/entropy": 4.3 / 17,  # (4 x 0.1 + 4 x 0.2 + 5 x 0.3 + 4 x 0.4) / 17 tokens
            "policy/flat_groups": 0.5,  # the second group's answers both scored 1
        },
        abs=1e-6,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_entropy_estimate_agrees_with_the_exact_entropy(tmp_path: Path, seed: int):
    model = make()
    with torch.no_grad():
        model.head.weight.mul_(20)  # sharper guesses: an entropy far from the uniform one
    batch = roll(model, 64, seed)
    with torch.no_grad():  # each answer token's whole distribution, from the position before it
        logp = log_probs(model(batch.tokens[:, :-1]), 1.0)
        entropy = torch.special.entr(logp.exp()).sum(dim=-1)  # -sum of p log p, 0 log 0 = 0
    aligned = torch.cat([torch.zeros_like(entropy[:, :1]), entropy], dim=1)
    exact = aligned[batch.response_mask].mean().item()
    sink = MetricsSink(tmp_path, "run-a")
    PolicyCollector(sink, preset("grpo")).on_rollout(
        RolloutEvent(step=1, batch=batch, rewards=torch.zeros(len(batch.finish_reasons)))
    )
    estimate = found(sink, tmp_path)["policy/entropy"]
    assert 0.9 < exact < 1.6  # far from the uniform distribution's ln 22 = 3.09
    assert estimate == pytest.approx(exact, abs=0.15)  # 100 seeds: never more than 0.07 off


def test_with_one_update_the_ratio_starts_and_stays_at_1(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = PolicyCollector(sink, preset("grpo"))
    update(make(), [collector], epochs=1)
    collector.on_step_end(ended(step=1))
    metrics = found(sink, tmp_path)
    assert metrics["policy/ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["policy/ratio_max"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["policy/clip_fraction"] == 0.0
    assert metrics["policy/kl_old"] == pytest.approx(0.0, abs=1e-9)


def test_over_several_updates_the_policy_metrics_agree_with_the_trainers(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = PolicyCollector(sink, preset("grpo"))
    updates = update(make(), [collector], epochs=3)
    collector.on_step_end(ended(step=1))
    metrics = found(sink, tmp_path)
    # Every epoch counts the same tokens, so the step's share is the epochs' mean share.
    clip_fraction = sum(u.clip_fraction for u in updates) / 3
    assert clip_fraction > 0  # the ratio moved far enough for clipping to engage
    assert metrics["policy/clip_fraction"] == pytest.approx(clip_fraction, rel=1e-9)
    assert metrics["policy/kl_ref"] == pytest.approx(sum(u.kl for u in updates) / 3, rel=1e-5)
    assert metrics["policy/ratio_max"] > 1.28  # GRPO clips above 1 + 0.2
    # The 8 made-up advantages, though each was seen in 3 epochs.
    assert metrics["policy/advantage_mean"] == pytest.approx(ADVANTAGES.mean().item(), abs=1e-6)
    advantage_std = ADVANTAGES.std(correction=0).item()
    assert metrics["policy/advantage_std"] == pytest.approx(advantage_std, rel=1e-6)


def test_each_step_starts_its_sums_afresh(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = PolicyCollector(sink, preset("grpo"))
    update(make(), [collector], epochs=3)
    collector.on_step_end(ended(step=1))
    collector.on_step_end(ended(step=2))  # a step with no update
    assert found(sink, tmp_path, step=2) == {}


# --- the optimization collector -------------------------------------------------------------


def test_the_optimization_metrics_by_hand(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = OptimizationCollector(sink)
    model = make()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch, (loss, grad_norm) in enumerate([(0.3, 0.5), (0.2, 1.5), (0.1, 3.0)]):
        collector.on_optimizer_step(
            OptimizerStepEvent(
                step=1,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=loss,
                grad_norm=grad_norm,
                grad_clip=1.0,
            )
        )
    collector.on_step_end(ended(step=1))
    collector.on_step_end(ended(step=2))  # a step with no update
    assert found(sink, tmp_path) == pytest.approx(
        {
            "optim/loss": 0.2,
            "optim/grad_norm": 5 / 3,
            "optim/grad_clipped": 2 / 3,  # 1.5 and 3.0 were cut down to 1
        }
    )
    assert found(sink, tmp_path, step=2) == {}


# --- the system collector -------------------------------------------------------------------

SECONDS = {"data": 0.5, "generate": 2.0, "score": 0.25, "logprobs": 1.0, "train": 4.0}


def test_the_system_metrics_by_hand(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = SystemCollector(sink)
    collector.on_rollout(RolloutEvent(step=1, batch=batch_of(GROUPS), rewards=REWARDS))
    update(make(), [collector], epochs=3)
    collector.on_step_end(ended(step=1, phases=SECONDS))
    batch = roll(make(), 4)  # the rows update() trained on, in micro-batches of 4
    read = 3 * sum(batch.rows(start, start + 4).tokens[:, :-1].numel() for start in (0, 4))
    assert found(sink, tmp_path) == pytest.approx(
        {
            "system/data_s": 0.5,
            "system/generate_s": 2.0,
            "system/score_s": 0.25,
            "system/logprobs_s": 1.0,
            "system/train_s": 4.0,
            "system/step_s": 8.0,
            "system/sample_tokens_per_s": 17 / 2.0,  # GROUPS' answers: 4 + 4 + 5 + 4 tokens
            "system/train_tokens_per_s": read / 4.0,
            "system/sample_flops": 300,
            "system/logprobs_flops": 20,
            "system/train_flops": 100,
        }
    )


def test_a_step_without_an_update_starts_afresh_and_a_gpu_records_its_peak(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = SystemCollector(sink)
    update(make(), [collector], epochs=1)  # step 1 updates...
    collector.on_step_end(ended(step=1, phases=SECONDS))
    collector.on_rollout(RolloutEvent(step=2, batch=batch_of(GROUPS), rewards=REWARDS))
    no_update = {"data": 0.5, "generate": 2.0, "score": 0.25}  # ...and step 2 doesn't
    collector.on_step_end(ended(step=2, phases=no_update, peak_memory=3 * 2**30))
    metrics = found(sink, tmp_path, step=2)
    assert "system/train_s" not in metrics and "system/train_tokens_per_s" not in metrics
    assert metrics["system/sample_tokens_per_s"] == 17 / 2.0
    assert metrics["system/peak_memory"] == 3 * 2**30


# --- a real run -----------------------------------------------------------------------------


@pytest.mark.parametrize(("algo", "reference"), [("grpo", True), ("rloo", False)])
def test_a_run_records_every_metric_at_every_step(tmp_path: Path, algo: str, reference: bool):
    sink = MetricsSink(tmp_path, "run-a")
    metrics = train(algo, [*make_collectors(sink, preset(algo)), sink], tmp_path)
    rows = read_metrics(tmp_path, "run-a")
    names = (
        {f"rollout/{name}" for name in ROLLOUT}
        | {f"policy/{name}" for name in ANSWERS | UPDATES | ({"kl_ref"} if reference else set())}
        | {f"optim/{name}" for name in OPTIMIZATION}
        | {f"system/{name}" for name in SYSTEM}
    )
    spent = 0
    for step in metrics:
        at_step = {row.metric: row.value for row in rows if row.step == step.step}
        assert at_step.keys() == names
        assert all(math.isfinite(value) for value in at_step.values())
        assert at_step["rollout/reward_mean"] == step.reward  # the same mean of the same rewards
        assert at_step["optim/loss"] == step.loss  # one update a step: its mean is that one
        assert at_step["optim/grad_norm"] == step.grad_norm
        assert at_step["optim/grad_clipped"] == (step.grad_norm > 1.0)  # RLConfig's grad_clip
        phases = sum(at_step[f"system/{phase}_s"] for phase in PHASES)
        assert 0 < phases <= at_step["system/step_s"]
        cost = sum(at_step[f"system/{phase}_flops"] for phase in ("sample", "logprobs", "train"))
        assert spent + cost == step.flops  # the cost model's count of this step alone
        spent = step.flops


def test_a_step_ends_before_its_checkpoint_which_hooks_hear_of_before_it_is_saved(
    tmp_path: Path,
):
    class Watcher(Hook):
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, str | None]] = []

        def on_step_end(self, event: StepEndEvent) -> None:
            self.calls.append(("step_end", event.step, None))

        def on_checkpoint(self, event: CheckpointEvent) -> None:
            latest = latest_checkpoint(tmp_path / "ckpt")  # the newest checkpoint already saved
            self.calls.append(("checkpoint", event.step, None if latest is None else latest.name))

    watcher = Watcher()
    train("rloo", [watcher], tmp_path)
    assert watcher.calls == [
        ("step_end", 1, None),
        ("step_end", 2, None),
        ("checkpoint", 2, None),  # step 2's checkpoint isn't saved yet
        ("step_end", 3, None),
        ("checkpoint", 3, "step-000000002.json"),
    ]


# --- added in m2s30: the detailed collectors ------------------------------------------------

DETAILED = {"optim/layer_grad_norm", "optim/update_ratio"} | {
    f"adam/{name}" for name in ADAM_METRICS
}
LAYERS = range(TINY.n_layers + 2)  # the embedding, the blocks, the head


def by_layer(sink: MetricsSink, home: Path, metric: str, step: int = 1) -> dict[int | None, float]:
    """What the sink holds for `metric` at `step`, by layer."""
    sink.flush()
    rows = read_metrics(home, sink.run_id)
    return {row.layer: row.value for row in rows if row.step == step and row.metric == metric}


def stepped(model: Transformer, optimizer: torch.optim.Optimizer, epoch: int) -> OptimizerStepEvent:
    """Update number `epoch` of step 1 is done (its loss and gradient norm made up)."""
    return OptimizerStepEvent(
        step=1,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        loss=0.0,
        grad_norm=0.0,
        grad_clip=1.0,
    )


def flat(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().flatten() for t in tensors])


def test_the_cadence_measures_every_k_steps_and_each_step_for_a_while_after_an_alert():
    assert (Cadence().every, Cadence().after_alert) == (10, 50)  # the blueprint's defaults
    cadence = Cadence(every=10, after_alert=50)
    assert [step for step in range(1, 101) if cadence.due(step)] == list(range(10, 101, 10))
    due: list[int] = []
    for step in range(1, 101):  # a run's steps, in order
        if cadence.due(step):
            due.append(step)
        if step in (25, 30):  # a detector fires at the end of the step: steps 26 to 75
            cadence.alert(step)  # then another, inside that window: now to 80
    assert due == [10, 20, *range(26, 81), 90, 100]


def test_the_layers_hold_every_parameter_once_in_order():
    model = make()
    parts = layers(model)
    assert len(parts) == len(LAYERS)
    assert len(parts[0]) == 1 and parts[0][0] is model.embed.weight
    assert parts[-1][-1] is model.head.weight
    assert all(a is b for a, b in zip(parts[1], model.blocks[0].parameters(), strict=True))
    held = [id(p) for part in parts for p in part]
    assert sorted(held) == sorted(id(p) for p in model.parameters())  # each one, once


def test_the_layer_metrics_by_hand(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = LayerCollector(sink, Cadence(every=1))
    model = make()
    before = copy.deepcopy(model)
    (metrics,) = update(model, [collector], epochs=1)
    collector.on_step_end(ended(step=1))
    # The gradients are still there, clipped (to rl_update's grad_clip, 1): clipping scaled
    # every one of them by the same factor.
    scale = min(1.0, 1.0 / (metrics.grad_norm + 1e-6))
    grads: list[float] = []
    ratios: list[float] = []
    for new, old in zip(layers(model), layers(before), strict=True):
        grads.append(float(flat([p.grad for p in new if p.grad is not None]).norm()) / scale)
        change = flat(new) - flat(old)
        ratios.append(float(change.square().mean().sqrt() / flat(old).square().mean().sqrt()))
    assert math.hypot(*grads) == pytest.approx(metrics.grad_norm, rel=1e-5)
    expected = dict(enumerate(grads))
    assert by_layer(sink, tmp_path, "optim/layer_grad_norm") == pytest.approx(expected, rel=1e-5)
    expected = dict(enumerate(ratios))
    assert by_layer(sink, tmp_path, "optim/update_ratio") == pytest.approx(expected, rel=1e-5)


def test_the_adam_metrics_by_hand(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = AdamCollector(sink, Cadence(every=1))
    model = make()
    parts = layers(model)
    # make_optimizer's two groups, the norms' weights (vectors) with a far bigger eps: 1e-2,
    # so 10 eps is 0.1. The matrices' eps is the default, 1e-8: 10 eps is 1e-7.
    matrices = [p for p in model.parameters() if p.dim() >= 2]
    vectors = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": matrices}, {"params": vectors, "eps": 1e-2}]
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    for epoch in range(2):  # two updates in step 1, with the same gradients
        for p in model.parameters():
            p.grad = torch.full_like(p, 1e-3)  # sqrt(v) is 1e-3
        for p in parts[2]:
            p.grad = torch.zeros_like(p)  # the second block has no gradient at all
        grad = torch.full_like(model.embed.weight, 1e-3)
        grad[0] = 1e-9  # token 0's row, 64 entries: sqrt(v) is 1e-9, below 10 eps
        grad[1] = 0.0  # token 1's row: no gradient, so not counted
        model.embed.weight.grad = grad
        optimizer.step()
        collector.on_optimizer_step(stepped(model, optimizer, epoch))
    collector.on_step_end(ended(step=1))
    # With the same gradient g at every update, the corrected v is g squared after each
    # one. The uncorrected v after the second is 0.001999 g squared.
    smallest = {0: 1e-18, 1: 1e-6, 3: 1e-6}  # nothing for layer 2
    # abs=0: approx also allows an absolute 1e-12 by default, which would pass any v near 1e-18.
    assert by_layer(sink, tmp_path, "adam/v_min") == pytest.approx(smallest, rel=1e-5, abs=0)
    everywhere = {0: 1e-6, 1: 1e-6, 3: 1e-6}
    assert by_layer(sink, tmp_path, "adam/v_median") == pytest.approx(everywhere, rel=1e-5)
    assert by_layer(sink, tmp_path, "adam/v_max") == pytest.approx(everywhere, rel=1e-5)

    def share_of_vectors(part: list[torch.Tensor]) -> float:
        return sum(p.numel() for p in part if p.dim() < 2) / sum(p.numel() for p in part)

    shares = {0: 64 / (64 * 64 - 64), 1: share_of_vectors(parts[1]), 3: share_of_vectors(parts[3])}
    assert by_layer(sink, tmp_path, "adam/eps_dominated") == pytest.approx(shares)


def test_the_detailed_collectors_skip_steps_not_due_and_optimizers_without_v(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    cadence = Cadence(every=2)
    detailed: list[Hook] = [LayerCollector(sink, cadence), AdamCollector(sink, cadence)]
    update(make(), detailed, epochs=1)  # step 1: not due
    for collector in detailed:
        collector.on_step_end(ended(step=1))
    assert found(sink, tmp_path) == {}
    model = make()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    adam = AdamCollector(sink, Cadence(every=1))
    adam.on_optimizer_step(stepped(model, optimizer, 0))
    adam.on_step_end(ended(step=1))
    assert found(sink, tmp_path) == {}
    assert optimizer.state == {}  # looked at, not touched


def test_a_run_measures_every_layer_on_the_steps_its_cadence_picks(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    train("grpo", [*make_collectors(sink, preset("grpo"), Cadence(every=2)), sink], tmp_path)
    rows = read_metrics(tmp_path, "run-a")
    per_layer = {(row.step, row.metric, row.layer) for row in rows if row.metric in DETAILED}
    # Of steps 1 to 3, only step 2. The tiny untrained model's answers may all score the
    # same, leaving every gradient zero: then Adam's v has nothing to show yet.
    names = {metric for _, metric, _ in per_layer}
    assert {"optim/layer_grad_norm", "optim/update_ratio"} <= names <= DETAILED
    assert per_layer == {(2, name, layer) for name in names for layer in LAYERS}
    grad_norm = found(sink, tmp_path, step=2)["optim/grad_norm"]
    grads = by_layer(sink, tmp_path, "optim/layer_grad_norm", step=2)
    assert math.hypot(*grads.values()) == pytest.approx(grad_norm, rel=1e-5)


# --- added in m2s31: the activation collector -----------------------------------------------

STREAM = range(TINY.n_layers + 1)  # the residual stream: after the embedding, after each block
BLOCKS = range(1, TINY.n_layers + 1)


def forwarded(
    model: Transformer, batch: RolloutBatch, step: int = 1, epoch: int = 0
) -> ForwardEvent:
    """A forward pass of update `epoch` of `step` over `batch`. Its numbers are made up: the
    activation collector reads only the model and the rows."""
    zeros = torch.zeros(batch.tokens.shape)
    return ForwardEvent(
        step=step,
        epoch=epoch,
        model=model,
        batch=batch,
        logp=zeros,
        old_logp=zeros,
        ref_logp=None,
        advantages=torch.zeros(batch.tokens.shape[0]),
        mask=batch.response_mask,
        loss=torch.tensor(0.0),
    )


def test_the_activation_metrics_by_hand(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = ActivationCollector(sink, Cadence(every=1))
    model = make()
    with torch.no_grad():  # the second block's queries are all 0, and so are its scores
        dict(model.named_parameters())["blocks.1.attn.q.weight"].zero_()
    batch = batch_of(GROUPS)  # 4 rows of 11, 11, 12 and 11 tokens
    for start in (0, 2):  # two micro-batches, each cut to its own width
        collector.on_forward(forwarded(model, batch.rows(start, start + 2)))
    collector.on_step_end(ended(step=1))
    hooked = [m for m in model.modules() if m._forward_hooks]  # pyright: ignore[reportPrivateUsage]
    assert hooked == []  # the collector took its hooks off again
    assert all(p.grad is None for p in model.parameters())
    # By hand, the whole batch at once, picking out the positions whose next token is real.
    tokens = batch.tokens[:, :-1]
    positions = torch.arange(tokens.shape[1])
    real = positions < batch.lengths[:, None] - 1
    stream: list[torch.Tensor] = []
    entropy: list[float] = []
    top: list[float] = []
    with torch.no_grad():
        x = model.embed(tokens)
        stream.append(x[real])
        for block in [cast(Block, block) for block in model.blocks]:
            scores = block.attn.scores(block.attn_norm(x), model.rope, positions)
            weights = scores.transpose(1, 2)[real].softmax(dim=-1)  # (positions, heads, keys)
            entropy.append(float(-torch.xlogy(weights, weights).sum(dim=-1).mean()))
            top.append(float(scores.transpose(1, 2)[real].max()))
            x = block(x, model.rope, positions)
            stream.append(x[real])
        logits = model(tokens)[real]  # (positions, vocab)
    rms = {layer: float(s.square().mean().sqrt()) for layer, s in zip(STREAM, stream, strict=True)}
    assert by_layer(sink, tmp_path, "activations/rms") == pytest.approx(rms, rel=1e-5)
    biggest = {layer: float(s.abs().max()) for layer, s in zip(STREAM, stream, strict=True)}
    assert by_layer(sink, tmp_path, "activations/abs_max") == pytest.approx(biggest, rel=1e-5)
    # With every score 0, the query at position t spreads its weight evenly over the t + 1
    # keys it may see: an entropy of log(t + 1).
    even = float(torch.log(positions + 1.0).expand_as(real)[real].mean())
    assert entropy[1] == pytest.approx(even, rel=1e-5)
    assert by_layer(sink, tmp_path, "attention/entropy") == pytest.approx(
        {1: entropy[0], 2: even}, rel=1e-5
    )
    assert by_layer(sink, tmp_path, "attention/max_logit") == pytest.approx(
        {1: top[0], 2: 0.0}, rel=1e-5
    )
    output = found(sink, tmp_path)
    assert output["output/max_logit"] == pytest.approx(float(logits.max()), rel=1e-5)
    z_mean = float(logits.logsumexp(dim=-1).mean())
    assert output["output/z_mean"] == pytest.approx(z_mean, rel=1e-5)


def test_the_activation_collector_reads_only_the_first_update_of_a_due_step(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    collector = ActivationCollector(sink, Cadence(every=2))
    model, batch = make(), batch_of(GROUPS)
    passes: list[int] = []

    def count(*_: object) -> None:
        passes.append(1)

    model.register_forward_hook(count)  # counts the model's forward passes
    collector.on_forward(forwarded(model, batch, step=1))  # not due
    collector.on_step_end(ended(step=1))
    collector.on_forward(forwarded(model, batch, step=2, epoch=1))  # due, but the second update
    collector.on_step_end(ended(step=2))
    assert passes == []
    assert found(sink, tmp_path, step=1) == found(sink, tmp_path, step=2) == {}
    collector.on_forward(forwarded(model, batch, step=4))  # due, the first update
    collector.on_step_end(ended(step=4))
    assert passes == [1]
    assert set(OUTPUT_METRICS) <= found(sink, tmp_path, step=4).keys()


def test_a_run_measures_the_activations_on_the_steps_its_cadence_picks(tmp_path: Path):
    sink = MetricsSink(tmp_path, "run-a")
    train("grpo", [*make_collectors(sink, preset("grpo"), Cadence(every=2)), sink], tmp_path)
    groups = ("activations/", "attention/", "output/")
    rows = [row for row in read_metrics(tmp_path, "run-a") if row.metric.startswith(groups)]
    expected = (
        {(2, name, layer) for name in STREAM_METRICS for layer in STREAM}
        | {(2, name, layer) for name in ATTENTION_METRICS for layer in BLOCKS}
        | {(2, name, None) for name in OUTPUT_METRICS}
    )
    assert {(row.step, row.metric, row.layer) for row in rows} == expected  # only step 2
    assert all(math.isfinite(row.value) for row in rows)
