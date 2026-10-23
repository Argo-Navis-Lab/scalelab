"""The `scalelab` command."""

import argparse
import contextlib
import json
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import torch

from scalelab.core import hashing
from scalelab.core.checkpoint import CheckpointPolicy
from scalelab.core.config import (
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    load_config,
    parse_override,
)
from scalelab.core.determinism import MAX_SEED, default_deterministic
from scalelab.core.envcapture import EnvInfo, capture_env
from scalelab.core.registry import AnomalyRecord, EvalRecord, OutdatedRegistryError, Registry
from scalelab.core.storage import atomic_write_bytes, registry_path, scalelab_home
from scalelab.models.transformer import Transformer
from scalelab.rl.evaluation import EvalResult, SampledResult
from scalelab.rl.experiment import (
    grade_by_part,
    load_eval_sets,
    make_task,
    run_experiment,
    run_rl_experiment,
)
from scalelab.rl.finished import load_finished_run
from scalelab.rl.gradings import Grading, gradings_folder, save_grading
from scalelab.rl.loop import SNAPSHOTS, RLStepMetrics
from scalelab.rl.overhead import OVERHEAD_BUDGET, measure_overhead
from scalelab.rl.warmstart_loop import StepMetrics
from scalelab.telemetry.anomalies import AnomalyEvent, RunHalted
from scalelab.telemetry.sink import MetricsSink, metrics_folder

EXIT_USAGE = 2  # the command line asked for something impossible
EXIT_INTERRUPTED = 130  # stopped early with a checkpoint saved; run again to resume
EXIT_HALTED = 1  # m2s35: a detector's HALT stopped the run, recorded failed (as a crash is)
EXIT_OVER_BUDGET = 1  # m2s36: `scalelab overhead` found telemetry costs more than its budget
METRIC = "greedy_pass@1"  # the greedy grade's name in the evals table
SAMPLED_KS = (1, 4)  # RL's sampled grades: pass@1, and pass@4 (how often 4 tries find it)


class UsageError(Exception):
    """A problem with what was asked for, reported as one clear line instead of a traceback."""


def split_overrides(overrides: Sequence[str]) -> tuple[int, dict[str, Any], list[str]]:
    """Sort key=value arguments into the seed, runtime settings, and experiment overrides."""
    seed = 0
    runtime: dict[str, Any] = {}
    config: list[str] = []
    for text in overrides:
        key, value = parse_override(text)
        if key == "seed":
            if type(value) is not int or not 0 <= value <= MAX_SEED:
                raise ValueError(f"seed must be a whole number from 0 to {MAX_SEED}, got {value!r}")
            seed = value
        elif key.startswith("runtime."):
            runtime[key.removeprefix("runtime.")] = value
        else:
            config.append(text)
    return seed, runtime, config


def duration(seconds: float) -> str:
    """A length of time, rounded the way a person would say it: "40 s", "12 min", "3.2 h"."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def speed(tokens: int, seconds: float, done: int, budget: int) -> str:
    """How fast `tokens` went by in `seconds`, and how long the rest of the budget will take
    at that speed, with `done` of its `budget` tokens read so far."""
    per_second = tokens / seconds
    left = max(0, budget - done) / per_second  # the last step overshoots the budget a little
    return f"{per_second:,.0f} tokens/s, about {duration(left)} to go"


def flops_speed(flops: float, seconds: float, done: float, budget: float) -> str:
    """The same for RL, whose budget is in FLOPs: how fast `flops` went by in `seconds`, and
    how long the rest of the budget will take at that speed, with `done` of it spent."""
    per_second = flops / seconds
    left = max(0.0, budget - done) / per_second
    return f"{per_second:.2e} FLOPs/s, about {duration(left)} to go"


def eval_records(
    run_id: str,
    step: int,
    greedy: Sequence[EvalResult],
    sampled: Sequence[SampledResult] = (),
) -> list[EvalRecord]:
    """One grading's rows for the evals table: greedy pass@1 for each greedy result, then
    sampled pass@1 and pass@4 for each sampled one."""
    records = [
        EvalRecord(
            run_id=run_id,
            step=step,
            task=result.name,
            metric=METRIC,
            value=result.pass_rate,
            n=result.n,
            stderr=result.stderr,
        )
        for result in greedy
    ]
    for result in sampled:
        for k in SAMPLED_KS:
            records.append(
                EvalRecord(
                    run_id=run_id,
                    step=step,
                    task=result.name,
                    metric=f"sampled_pass@{k}",
                    value=result.pass_at(k),
                    n=result.n,
                    stderr=result.stderr(k),
                )
            )
    return records


def load_warm_start(
    from_run: str, model: ModelConfig, registry: Registry, home: Path
) -> Transformer:
    """The model an RL experiment starts from: run `from_run`'s last checkpoint. Refused with
    a UsageError, before anything is recorded, unless that run is finished in `registry` and
    trained the same model as the experiment's."""
    try:
        finished = load_finished_run(registry, home, from_run)
    except ValueError as err:
        raise UsageError(f"rl.from_run: {err}") from None
    if finished.model.config != model:
        raise UsageError(
            f"rl.from_run {from_run} trained model {finished.model.config.name}, "
            f"not this experiment's model {model.name}"
        )
    return finished.model


def same_uncommitted_changes(recorded_env_json: str, env: EnvInfo) -> bool:
    """Was the run recorded with exactly the uncommitted changes we have now (or none)?"""
    recorded = EnvInfo.model_validate_json(recorded_env_json)
    return recorded.git.diff_sha256 == env.git.diff_sha256


def resolve(
    experiment: Path, overrides: Sequence[str]
) -> tuple[ExperimentConfig, int, RuntimeConfig]:
    """The experiment's config with `overrides` applied, the seed, and the runtime settings."""
    try:
        seed, runtime_values, config_overrides = split_overrides(overrides)
        config = load_config(experiment, config_overrides)
        runtime = RuntimeConfig.model_validate(runtime_values)
    except (ValueError, FileNotFoundError) as err:  # pydantic's ValidationError is a ValueError
        raise UsageError(str(err)) from None
    return config, seed, runtime


@contextlib.contextmanager
def registry_copy(home: Path) -> Generator[Registry]:
    """A private copy of <home>'s registry, for commands that only read it: the registry
    itself is never locked or changed (on Colab it lives on Google Drive, where SQLite is
    unreliable). The copy is deleted when the `with` block ends."""
    source = registry_path(home)
    if not source.is_file():
        raise UsageError(f"no registry at {source}")
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder) / "registry.db"
        shutil.copyfile(source, copy)
        try:
            registry = Registry.sqlite(copy)
        except OutdatedRegistryError as err:
            raise UsageError(str(err)) from None
        yield registry


def run(experiment: Path, overrides: Sequence[str]) -> int:
    """`scalelab run`: resolve the experiment, record it, run it (or resume it), finish it."""
    config, seed, runtime = resolve(experiment, overrides)

    home = scalelab_home(default=runtime.output_dir)
    at_home = registry_path(home)
    live = at_home  # the registry this run writes to
    if runtime.local_registry:
        live = Path(runtime.local_registry).expanduser().resolve()
    env = capture_env()
    run_id = hashing.run_id(config, seed, env.git.sha)
    deterministic = runtime.deterministic
    if deterministic is None:
        deterministic = default_deterministic()
    code = f"{env.git.sha[:8]} ({'uncommitted changes' if env.git.dirty else 'clean'})"
    print(f"experiment:    {config.name} (config {hashing.config_hash(config)}), seed {seed}")
    print(f"run:           {run_id}")
    print(f"code:          {code}")
    print(f"deterministic: {'on' if deterministic else 'off'}")
    print(f"home:          {home}")

    if live != at_home:
        print(f"registry:      {live}, copied to home after every change")
        if not live.exists() and at_home.exists():
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(at_home, live)  # a new machine carries on from the last copy

    def copy_home() -> None:
        """Copy the live registry to <home>, if it lives elsewhere. Only this process writes
        it, and between writes the file is complete, so a plain copy is a consistent one."""
        if live != at_home:
            atomic_write_bytes(at_home, live.read_bytes())

    try:
        registry = Registry.sqlite(live)
    except OutdatedRegistryError as err:
        raise UsageError(f"{err}. If it only holds old demo runs, delete that file.") from None
    existing = registry.get_run(run_id)
    if existing is not None and not same_uncommitted_changes(existing.env_json, env):
        raise UsageError(
            f"run {run_id} was recorded with different uncommitted changes. "
            "Commit your changes (a new commit gives a new run_id) and run again."
        )
    if existing is not None and existing.status == "done":
        print("already done: nothing to do")
        return 0
    start = None  # RL's starting model, checked before anything is recorded
    if config.rl is not None:
        start = load_warm_start(config.rl.from_run, config.model, registry, home)
        print(f"warm start:    {config.rl.from_run}")
        telemetry = metrics_folder(home, run_id) if runtime.telemetry else "off"  # m2s36
        print(f"metrics:       {telemetry}")
        print(f"gradings:      {gradings_folder(home, run_id)}")  # m2s38

    # Speed is measured between gradings, from the time and the token (or FLOP) count at the
    # last one. A resumed run's count isn't known here, so its first grading shows none.
    last_tokens = 0 if existing is None else None
    last_flops = 0 if existing is None else None

    def report(metrics: StepMetrics, results: list[EvalResult]) -> None:
        """Record each eval result in the registry, and print one line of progress: the
        grades, and the speed since the last grading, with the time it gives to the end."""
        nonlocal last_time, last_tokens
        for record in eval_records(run_id, metrics.step, results):
            registry.record_eval(record)
        copy_home()
        rates = "  ".join(f"{result.name} {result.pass_rate:.1%}" for result in results)
        now, pace = time.monotonic(), ""
        if last_tokens is not None and now > last_time:
            read = metrics.tokens - last_tokens  # tokens read since the last grading
            pace = f"  ({speed(read, now - last_time, metrics.tokens, config.warm_start.tokens)})"
        last_time, last_tokens = now, metrics.tokens
        print(f"step {metrics.step:,}: loss {metrics.loss:.3f}  {rates}{pace}")

    def rl_report(
        metrics: RLStepMetrics | None, greedy: list[EvalResult], sampled: list[SampledResult]
    ) -> None:
        """The same for RL: keep the grading problem by problem in its file, record it (greedy
        and sampled, whole and by part), and print the whole eval sets' greedy and sampled
        pass@1. metrics is None at step 0."""
        nonlocal last_time, last_flops
        step = 0 if metrics is None else metrics.step
        grading = Grading(run_id=run_id, step=step, greedy=tuple(greedy), sampled=tuple(sampled))
        save_grading(home, grading)  # m2s38: first the outcomes, then the rows that sum them up
        for record in eval_records(run_id, step, greedy, sampled):
            registry.record_eval(record)
        copy_home()
        by_name = {result.name: result for result in sampled}
        rates = "  ".join(
            f"{result.name} {result.pass_rate:.1%} (sampled {by_name[result.name].pass_at(1):.1%})"
            for result in greedy
            if "/" not in result.name  # whole sets only: the parts are in the registry
        )
        now, pace = time.monotonic(), ""
        if metrics is None:
            print(f"step 0 (the warm start): {rates}")
        else:
            if last_flops is not None and now > last_time and config.rl is not None:
                done = metrics.flops - last_flops  # FLOPs spent since the last grading
                end = config.rl.flops  # where the run will end, in FLOPs: its budget, or...
                if config.rl.max_steps is not None:  # ...max_steps, at the FLOPs a step so far
                    end = min(end, metrics.flops / metrics.step * config.rl.max_steps)
                pace = f"  ({flops_speed(done, now - last_time, metrics.flops, end)})"
            last_flops = metrics.flops
            print(f"step {metrics.step:,}: reward {metrics.reward:.3f}  {rates}{pace}")
        last_time = now

    def record_anomaly(event: AnomalyEvent) -> None:
        """Record one event of the online detectors, and say what was found."""
        registry.record_anomaly(
            AnomalyRecord(
                run_id=run_id,
                step=event.step,
                detector=event.detector,
                metric=event.metric,
                severity=event.severity,
                signature_json=json.dumps(event.window_metrics, sort_keys=True),
            )
        )
        copy_home()
        numbers = "  ".join(f"{name} {value:.4g}" for name, value in event.window_metrics.items())
        print(f"step {event.step}: {event.severity} {event.detector} in {event.metric}: {numbers}")

    policy = CheckpointPolicy(every_steps=runtime.checkpoint_every)
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    policy.stop_on_signals()  # from here on, Ctrl+C and SIGTERM mean "save and stop"
    try:
        if existing is None:
            registry.add_run(config, seed, env, executor="local", deterministic=deterministic)
            print("started")
        else:
            registry.reopen_run(run_id)
            print(f"resuming (it was {existing.status})")
        copy_home()
        started = last_time = time.monotonic()  # the clock starts with training
        try:
            checkpoints = home / "checkpoints" / run_id
            if start is None:
                metrics = run_experiment(
                    config, seed, checkpoints, home, policy, deterministic, report=report
                )
            else:
                # What the collectors measure goes here (m2s36: unless telemetry is off)
                sink = MetricsSink(home, run_id) if runtime.telemetry else None
                metrics = run_rl_experiment(
                    config,
                    seed,
                    start,
                    checkpoints,
                    home,
                    policy,
                    deterministic,
                    rl_report,
                    sink,
                    record_anomaly,
                )
        except RunHalted as halted:
            registry.finish_run(run_id, "failed")
            snapshots = home / "checkpoints" / run_id / SNAPSHOTS
            print(f"halted at {halted}; the step's state is in {snapshots}")
            print("run the same command again to retry from the last checkpoint")
            return EXIT_HALTED
        except Exception:
            registry.finish_run(run_id, "failed")
            raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        copy_home()

    if policy.stop_requested:
        print("stopped early, checkpoint saved: run the same command again to resume")
        return EXIT_INTERRUPTED
    registry.finish_run(run_id, "done")
    copy_home()

    if not metrics:
        print("done: the last checkpoint was already saved, so there was nothing left to train")
        return 0
    last, took = metrics[-1], duration(time.monotonic() - started)
    if isinstance(last, RLStepMetrics):
        print(
            f"done: ran {len(metrics)} steps in {took} ({last.flops:.2e} FLOPs in all), "
            f"last reward {last.reward:.3f}"
        )
    else:
        print(
            f"done: ran {len(metrics)} steps in {took} "
            f"({last.tokens:,} tokens and {last.flops:.2e} FLOPs in all), last loss {last.loss!r}"
        )
    return 0


def grade(run_id: str, problems: int) -> int:
    """`scalelab grade`: greedy pass@1 of a finished run's model on the frozen eval sets of
    its task, whole and by part (arithmetic's operations). Prints the grades, records nothing."""
    if problems < 1:
        raise UsageError(f"--problems must be at least 1, got {problems}")
    home = scalelab_home()
    with registry_copy(home) as registry:
        try:
            finished = load_finished_run(registry, home, run_id)
        except ValueError as err:
            raise UsageError(str(err)) from None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task, problem_type = make_task(finished.task.name)
    eval_sets = load_eval_sets(task, problem_type, finished.task.difficulties, problems)
    levels = ", ".join(str(level) for level in finished.task.difficulties)
    print(f"run:   {run_id}")
    print(
        f"model: {finished.model.config.name}, step {finished.step:,}, "
        f"trained on {task.name} at difficulties {levels}"
    )
    for result in grade_by_part(finished.model.to(device), task, eval_sets):
        rate = f"{result.pass_rate:6.1%} ± {result.stderr:.1%}"
        print(f"{result.name:<15}{rate}  ({result.passed} of {result.n})")
    return 0


def overhead(experiment: Path, overrides: Sequence[str], steps: int, warmup: int) -> int:
    """`scalelab overhead`: what telemetry adds to an RL experiment's step time. Runs its
    first steps twice from its warm start, without telemetry and with it, and records
    nothing. Exit code 1 if telemetry costs more than its budget."""
    config, seed, runtime = resolve(experiment, overrides)
    if config.rl is None:
        raise UsageError(f"{config.name} is not an RL experiment: only RL has telemetry")
    if steps < 1 or warmup < 1:
        raise UsageError(f"--steps and --warmup must be at least 1, got {steps} and {warmup}")
    home = scalelab_home(default=runtime.output_dir)
    with registry_copy(home) as registry:
        start = load_warm_start(config.rl.from_run, config.model, registry, home)
    deterministic = runtime.deterministic
    if deterministic is None:
        deterministic = default_deterministic()
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(f"experiment:    {config.name} (config {hashing.config_hash(config)}), seed {seed}")
    print(f"warm start:    {config.rl.from_run}")
    print(f"device:        {device}")
    print(f"steps:         {warmup} to warm up, then {steps} timed; without, then with telemetry")
    result = measure_overhead(config, seed, start, deterministic, steps, warmup)
    events = len(result.on.events)
    print(f"without:       {result.off.seconds:.3f} s a step")
    print(f"with:          {result.on.seconds:.3f} s a step ({events} detector events)")
    within = result.overhead <= OVERHEAD_BUDGET
    verdict = f"{'within' if within else 'OVER'} the {OVERHEAD_BUDGET:.0%} budget"
    print(f"overhead:      {result.overhead:+.1%} of step time, {verdict}")
    return 0 if within else EXIT_OVER_BUDGET


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scalelab", description="ScaleLab command line.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run one experiment and record it")
    run_parser.add_argument("experiment", type=Path, help="e.g. configs/experiments/hello.yaml")
    run_parser.add_argument(
        "overrides",
        nargs="*",
        help="key=value changes, e.g. seed=2 model=ladder_s/m2 warm_start.lr=3e-4",
    )
    grade_parser = commands.add_parser("grade", help="grade a finished run on its eval sets")
    grade_parser.add_argument("run_id", help="e.g. 71cd3add8c11e667-s0-2aeaf16b")
    grade_parser.add_argument(
        "--problems", type=int, default=1000, help="the first N of each eval set (default 1000)"
    )
    overhead_parser = commands.add_parser(
        "overhead", help="time an RL experiment's steps without and with telemetry"
    )
    overhead_parser.add_argument("experiment", type=Path, help="e.g. configs/experiments/rl.yaml")
    overhead_parser.add_argument("overrides", nargs="*", help="key=value changes, as for run")
    overhead_parser.add_argument(
        "--steps", type=int, default=30, help="steps timed in each run (default 30)"
    )
    overhead_parser.add_argument(
        "--warmup", type=int, default=5, help="steps before the timed ones (default 5)"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "grade":
            return grade(args.run_id, args.problems)
        if args.command == "overhead":
            return overhead(args.experiment, args.overrides, args.steps, args.warmup)
        return run(args.experiment, args.overrides)
    except UsageError as err:
        print(f"scalelab: error: {err}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
