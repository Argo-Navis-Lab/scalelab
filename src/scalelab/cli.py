"""The `scalelab` command."""

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import signal
import socket
import sys
import tempfile
import time
import warnings
from collections import Counter
from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, get_args

import torch
import yaml

from scalelab.analysis.bands import CEILING, FLOOR, pick_bands
from scalelab.analysis.costreport import load_measurements, write_report
from scalelab.analysis.reports import gains
from scalelab.analysis.scaling import (
    ALL_OPS,
    NEAR,
    cumulative_compute,
    fit_run,
    run_points,
    step_reaching,
)
from scalelab.analysis.stats import Interval
from scalelab.bench.comms import SIZES, Backend, bench_collectives
from scalelab.bench.suites import (
    HALF_SUITES,
    PRECISIONS,
    SUITES,
    Suite,
    bench_folder,
    load_samples,
    plan_suite,
    results_file,
    save_results,
)
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
from scalelab.core.registry import (
    AnomalyRecord,
    EvalRecord,
    Job,
    JobState,
    OutdatedRegistryError,
    Registry,
    RunRecord,
)
from scalelab.core.storage import (
    atomic_write_bytes,
    atomic_write_text,
    registry_path,
    scalelab_home,
)
from scalelab.core.sweep import Sweep, expand, load_sweep
from scalelab.costmodel.calibrate import calibrate, env_versions, load_calibration, save_calibration
from scalelab.costmodel.comms import COLLECTIVES, collective_time, fit_link
from scalelab.costmodel.memory import RESERVED_FRACTION
from scalelab.costmodel.report import (
    CHANGE_RUNTIMES,
    CHANGES,
    Change,
    DecisionReport,
    Measurement,
    reports_folder,
)
from scalelab.costmodel.rlstep import StepPrediction
from scalelab.costmodel.roofline import HALF, Efficiency, Group
from scalelab.costmodel.specs import CPU, HARDWARE, HardwareSpec, Precision
from scalelab.costmodel.validate import (
    PARTS,
    TARGET_MAPE,
    VALIDATION_FILE,
    StepCheck,
    Validation,
    band,
    check_step,
    summarize,
)
from scalelab.models.transformer import Transformer
from scalelab.rl.costing import (
    measured_step,
    padding_cut,
    predict_gradings,
    predict_job,
    predict_rl_run,
    predict_run_step,
    predict_step,
)
from scalelab.rl.evaluation import EvalResult, SampledResult, evaluate
from scalelab.rl.experiment import (
    grade_by_part,
    load_eval_sets,
    make_task,
    operation,
    run_experiment,
    run_rl_experiment,
)
from scalelab.rl.finished import load_finished_run
from scalelab.rl.gradings import Grading, gradings_folder, load_gradings, save_grading
from scalelab.rl.loop import SNAPSHOTS, RLStepMetrics
from scalelab.rl.measure import measure_change
from scalelab.rl.overhead import OVERHEAD_BUDGET, measure_overhead
from scalelab.rl.warmstart_loop import StepMetrics
from scalelab.tasks.arithmetic import ArithmeticProblem
from scalelab.telemetry.anomalies import AnomalyEvent, RunHalted
from scalelab.telemetry.detectors import REGRESSION_STEPS, ThroughputDetector, regression_limit
from scalelab.telemetry.sink import MetricsSink, metrics_folder, read_metrics

EXIT_USAGE = 2  # the command line asked for something impossible
EXIT_INTERRUPTED = 130  # stopped early with a checkpoint saved; run again to resume
EXIT_HALTED = 1  # m2s35: a detector's HALT stopped the run, recorded failed (as a crash is)
EXIT_OVER_BUDGET = 1  # m2s36: `scalelab overhead` found telemetry costs more than its budget
# m3s6: `scalelab bench --models` names files in this folder, as in model=ladder_s/m2
MODELS_FOLDER = Path("configs") / "models"
BENCH_MODELS = ("ladder_s/m1", "ladder_s/m2", "ladder_s/m3", "ladder_s/m4")  # what M1/M2 ran
CALIBRATION_FILE = "calibration.json"  # m3s9: saved beside the bench results it was fitted to
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


def sampled_rate(result: SampledResult | None) -> str:
    """A sampled result's pass@1 for a progress line, " (sampled 12.3%)"; "" without one: the
    gradings between the first and the last sample nothing (m4s3)."""
    return "" if result is None else f" (sampled {result.pass_at(1):.1%})"


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


# m3s23: the settings a resumed run must keep. Each changes how its steps compute and how
# long they take, and cost validate times a run's steps as one runtime's.
SAME_ON_RESUME = ("precision", "padding_cut")


def runtime_changes(recorded: RuntimeConfig, runtime: RuntimeConfig) -> list[str]:
    """The SAME_ON_RESUME settings `runtime` changes, each as the override that keeps it."""
    before, now = recorded.model_dump(), runtime.model_dump()
    return [
        f"runtime.{name}={before[name]}" for name in SAME_ON_RESUME if before[name] != now[name]
    ]


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


@contextlib.contextmanager
def registry_edit(home: Path) -> Generator[Registry]:
    """<home>'s registry, for a command that changes it (m4s5): changed in a private copy, then
    put back in one atomic write, so SQLite never opens the file at home (on Colab, on Google
    Drive). Nothing is put back if the `with` block fails. One writer at a time."""
    source = registry_path(home)
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder) / "registry.db"
        if source.is_file():
            shutil.copyfile(source, copy)
        try:
            registry = Registry.sqlite(copy)
        except OutdatedRegistryError as err:
            raise UsageError(str(err)) from None
        yield registry
        registry.engine.dispose()
        atomic_write_bytes(source, copy.read_bytes())


def run(
    experiment: Path, overrides: Sequence[str], on_step: Callable[[object], None] | None = None
) -> int:
    """`scalelab run`: resolve the experiment, record it, run it (or resume it), finish it.
    `on_step`, if given, is called after every training step (m4s6: the worker's heartbeat)."""
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
    print(f"precision:     {runtime.precision}")  # m3s21: the run_id doesn't show it
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
    recorded = None if existing is None else registry.get_runtime(run_id)
    changed = [] if recorded is None else runtime_changes(recorded, runtime)  # m3s23
    if changed:
        raise UsageError(
            f"run {run_id} started with {' '.join(changed)}: resume it the same way "
            "(a run's steps all compute and are timed one way)"
        )
    start = None  # RL's starting model, checked before anything is recorded
    if config.rl is not None:
        start = load_warm_start(config.rl.from_run, config.model, registry, home)
        print(f"warm start:    {config.rl.from_run}")
        print(f"gradings:      {gradings_folder(home, run_id)}")  # m2s38
    telemetry = metrics_folder(home, run_id) if runtime.telemetry else "off"  # m2s36
    print(f"metrics:       {telemetry}")  # m3s11: a warm start's step times go there too
    throughput = None  # m3s25: an RL run's steps against the cost model's, with telemetry on
    if config.rl is not None and runtime.telemetry:
        throughput = watch_throughput(config, runtime)

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
        by_name = {result.name: result for result in sampled}  # m4s3: none between the ends
        rates = "  ".join(
            f"{result.name} {result.pass_rate:.1%}{sampled_rate(by_name.get(result.name))}"
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
            registry.add_run(  # m3s23: with its runtime
                config, seed, env, executor="local", deterministic=deterministic, runtime=runtime
            )
            print("started")
        else:
            registry.reopen_run(run_id)
            print(f"resuming (it was {existing.status})")
        copy_home()
        started = last_time = time.monotonic()  # the clock starts with training
        try:
            checkpoints = home / "checkpoints" / run_id
            # What the collectors measure goes here (m2s36: unless telemetry is off), and a warm
            # start's step times (m3s11)
            sink = MetricsSink(home, run_id) if runtime.telemetry else None
            if start is None:
                metrics = run_experiment(
                    config,
                    seed,
                    checkpoints,
                    home,
                    policy,
                    deterministic,
                    report,
                    sink,
                    padding_cut=runtime.padding_cut,  # m3s16
                    precision=runtime.precision,  # m3s21
                    on_step=on_step,  # m4s6
                )
            else:
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
                    padding_cut=runtime.padding_cut,  # m3s16
                    precision=runtime.precision,  # m3s22
                    throughput=throughput,  # m3s25
                    on_step=on_step,
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


def bands(run_ids: Sequence[str], problems: int) -> int:
    """`scalelab bands` (m4s7): finished warm starts' greedy pass@1 on the multiplications
    of each band's eval set, smallest model first, and the bands the M4 RL task keeps by the
    rule the M4 plan fixed (analysis/bands.py). Prints them, records nothing."""
    if len(run_ids) < 2 or len(set(run_ids)) < len(run_ids):
        raise UsageError(
            "give two or more different runs: the rule compares the smallest model with the largest"
        )
    if problems < 1:
        raise UsageError(f"--problems must be at least 1, got {problems}")
    home = scalelab_home()
    with registry_copy(home) as registry:
        try:
            runs = [(run_id, load_finished_run(registry, home, run_id)) for run_id in run_ids]
        except ValueError as err:
            raise UsageError(str(err)) from None
    difficulties = runs[0][1].task.difficulties
    if any(
        run.task.name != "arithmetic" or run.task.difficulties != difficulties for _, run in runs
    ):
        raise UsageError("the runs must all be trained on arithmetic at the same difficulties")
    runs.sort(key=lambda pair: sum(p.numel() for p in pair[1].model.parameters()))  # smallest first

    task, _ = make_task("arithmetic")
    eval_sets = load_eval_sets(task, ArithmeticProblem, difficulties, problems)
    times = {  # each band's multiplications: the RL task's operation
        band: [problem for problem in eval_set if operation(problem) == "*"]
        for band, eval_set in zip(difficulties, eval_sets.values(), strict=True)
    }
    if not all(times.values()):
        raise UsageError(f"the first {problems} problems of a band hold no *: grade more of them")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts = ", ".join(str(len(items)) for items in times.values())
    print(f"graded:        greedy pass@1 on each band's * problems ({counts} of them)")
    print(f"{'model':<6}" + "".join(f"{f'band {band}':>9}" for band in times) + "  run")
    rates: list[dict[int, float]] = []
    for run_id, run in runs:
        model = run.model.to(device)
        rates.append(
            {
                band: evaluate(model, task, items, f"{task.name}-{band}/*").pass_rate
                for band, items in times.items()
            }
        )
        cells = "".join(f"{rate:9.1%}" for rate in rates[-1].values())
        print(f"{run.model.config.name:<6}{cells}  {run_id}")
    smallest, largest = runs[0][1].model.config.name, runs[-1][1].model.config.name
    print(f"rule:          {smallest} at least {FLOOR:.0%} and {largest} at most {CEILING:.0%}")
    kept = pick_bands(rates[0], rates[-1])
    print(f"bands:         {', '.join(str(band) for band in kept) if kept else 'none'}")
    return 0


def fit(run_ids: Sequence[str]) -> int:
    """`scalelab fit` (m4s8): finished RL runs' pass rate R against their compute C at each
    grading, and the saturating sigmoid fit through them (analysis/scaling.py). R is greedy
    pass@1 on the problems of the run's operations (task.ops), pooled over its eval sets; C
    is A's FLOPs, summed from its metrics. Prints them, records nothing."""
    home = scalelab_home()
    found: list[tuple[str, str]] = []  # (run_id, its operations)
    with registry_copy(home) as registry:
        for run_id in run_ids:
            run = registry.get_run(run_id)
            if run is None:
                raise UsageError(f"no run {run_id} in the registry")
            if run.status != "done":
                raise UsageError(f"{run_id} is {run.status}: only finished runs are fit")
            config_json = registry.experiment_json(run.config_hash)
            assert config_json is not None  # every run row points at an experiment row
            recorded = json.loads(config_json)  # read raw, as old runs' configs may not load
            if recorded.get("rl") is None:
                raise UsageError(f"{run_id} is a warm start: only RL runs have a curve")
            found.append((run_id, recorded["task"].get("ops") or ALL_OPS))
    for run_id, ops in found:
        rows = read_metrics(home, run_id)
        try:
            points = run_points(load_gradings(home, run_id), rows, ops)
            result = fit_run(points)
        except ValueError as err:
            raise UsageError(f"{run_id}: {err}") from None
        compute = cumulative_compute(rows)  # every step's C (run_points has checked them)
        curve = result.curve
        print(f"run:           {run_id}")
        print(f"R:             greedy pass@1 on its {ops} problems, pooled ({points[0].n} of them)")
        print(f"{'step':>6}{'C':>12}{'R':>8}{'fitted':>8}")
        for point, fitted in zip(points, curve([point.compute for point in points]), strict=True):
            print(f"{point.step:>6}{point.compute:>12.3g}{point.rate:>8.1%}{fitted:>8.1%}")
        print(f"fit:           A {curve.a:.1%}, C_mid {curve.c_mid:.3g}, B {curve.b:.2f}", end="")
        print(f" (R_0 {curve.r0:.1%}: step 0's)")
        print(f"residuals:     {100 * result.rmse:.2f} pp rms over {result.points} gradings")
        if result.bounded:
            loose = ", ".join(result.bounded)
            print(f"on a bound:    {loose}, so these gradings don't pin the curve down")
        near = curve.compute_at(NEAR)
        step = step_reaching(compute, near)
        where = f"step {step}" if step is not None else f"not reached by step {max(compute)}"
        print(f"near A:        {NEAR:.0%} of the way from R_0 at C {near:.3g} ({where})")
        print()
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


def points(interval: Interval) -> str:
    """A difference of two rates and its interval, in percentage points."""
    low, high = 100 * interval.low, 100 * interval.high
    return f"{100 * interval.estimate:+.1f} [{low:+.1f}, {high:+.1f}] pp"


def report_gains() -> int:
    """`scalelab report gains`: what RL did to greedy pass@1, from step 0 (the warm start) to
    the last grading, for each RL experiment and commit whose finished runs kept their
    gradings (m2s38), its seeds paired problem by problem. Records nothing."""
    home = scalelab_home()
    seeds: dict[tuple[str, str], list[RunRecord]] = {}  # (config_hash, git_sha) -> its runs
    labels: dict[str, str] = {}  # config_hash -> "<model> <algo>"
    left_out = 0
    with registry_copy(home) as registry:
        for run in registry.list_runs():
            config_json = registry.experiment_json(run.config_hash)
            assert config_json is not None  # every run row points at an experiment row
            recorded = json.loads(config_json)  # read raw, as old runs' configs may not load
            if recorded.get("rl") is None:
                continue  # a warm start
            if run.status != "done" or not gradings_folder(home, run.run_id).is_dir():
                left_out += 1
                continue
            labels[run.config_hash] = f"{recorded['model']['name']} {recorded['algo']['name']}"
            seeds.setdefault((run.config_hash, run.git_sha), []).append(run)
    if not seeds:
        raise UsageError("no finished RL run kept its gradings (runs from m2s38 on do)")
    for (config_hash, git_sha), runs in seeds.items():
        gradings = [load_gradings(home, run.run_id) for run in runs]
        dirty = ", uncommitted changes" if any(run.git_dirty for run in runs) else ""
        print(f"{labels[config_hash]}: experiment {config_hash} at {git_sha[:8]}{dirty}")
        listed = ", ".join(str(run.seed) for run in runs)
        steps = ", ".join(str(graded[-1].step) for graded in gradings)
        print(f"  seeds {listed}; last steps {steps}")
        print(
            f"  {'eval set':<16}{'n':>5}{'first':>8}{'last':>8}  "
            f"{'gain, mean [95% CI]':<26}{'gain, IQM [95% CI]':<26}verdict"
        )
        for gain in gains(gradings):
            print(
                f"  {gain.name:<16}{gain.n:>5}{gain.first:>8.1%}{gain.last:>8.1%}  "
                f"{points(gain.mean):<26}{points(gain.iqm):<26}{gain.verdict}"
            )
        print()
    if left_out:
        print(f"left out {left_out} of the RL runs: unfinished, or from before m2s38 (no gradings)")
    return 0


def load_model(choice: str) -> ModelConfig:
    """A model config by its name under configs/models/, as in model=ladder_s/m2."""
    path = MODELS_FOLDER / f"{choice}.yaml"
    if not path.is_file():
        raise UsageError(f"no model config {choice!r}: {path} doesn't exist")
    return ModelConfig.model_validate(yaml.safe_load(path.read_text()))


def bench(
    hardware_name: str,
    choices: Sequence[str],
    suites: Sequence[Suite] | None,
    precision: Precision,
    dry_run: bool,
) -> int:
    """`scalelab bench`: times the cost model's ops and whole decode steps, shaped as the
    models' own, on this machine's GPU (its CPU for --hardware cpu), and saves each suite's
    results to <home>/bench/<hardware>/<suite>.parquet (m3s19: in fp16, <suite>-fp16.parquet,
    and only the suites fp16 changes). Records nothing in the registry. A dry run only says
    what it would time."""
    if suites is None:
        suites = SUITES if precision == "fp32" else HALF_SUITES
    if precision != "fp32" and not set(suites) <= set(HALF_SUITES):
        raise UsageError(f"--precision {precision} times only {' '.join(HALF_SUITES)}")
    hardware = HARDWARE.get(hardware_name)  # None: the CPU, whose memory isn't modelled
    models = [load_model(choice) for choice in choices]
    device = torch.device("cpu" if hardware is None else "cuda")
    if hardware is not None and not dry_run:
        if not torch.cuda.is_available():
            raise UsageError(f"--hardware {hardware_name} needs its GPU; this machine has none")
        gpu = torch.cuda.get_device_name()
        if hardware_name.split("-")[0] not in gpu.lower():  # a100-40g: "NVIDIA A100-SXM4-40GB"
            raise UsageError(f"--hardware {hardware_name}, but this machine's GPU is a {gpu}")
    folder = bench_folder(scalelab_home(), hardware_name)
    print(f"hardware:      {hardware_name}")
    print(f"models:        {' '.join(choices)}")
    print(f"precision:     {precision}")
    print(f"results:       {folder}{' (dry run: nothing is timed or saved)' if dry_run else ''}")
    env = capture_env()
    for suite in suites:
        plan = plan_suite(suite, models, hardware, precision)
        line = f"{suite + ':':<15}{plan.cases} cases"
        if plan.skipped:
            line += f" ({plan.skipped} more too big for {hardware_name})"
        if plan.cases and not dry_run:
            start = time.perf_counter()
            results = plan.run(device)
            save_results(folder / results_file(suite, precision), results, env)
            line += f": {len(results)} timings in {duration(time.perf_counter() - start)}"
        print(line, flush=True)
    return 0


def hardware_spec(name: str) -> HardwareSpec:
    """A hardware profile by its name; cpu is this machine's CPU, as the cost model stands it in."""
    return CPU if name == "cpu" else HARDWARE[name]


def report_cost(hardware_names: Sequence[str], out: Path) -> int:
    """`scalelab report cost` (m3s26): Write-up 1's figures and tables, from each device's last
    cost validate (validation.json, with its runs' steps) and the changes cost measure saved in
    <home>/reports/ for those devices. Records nothing in the registry."""
    if not hardware_names:
        raise UsageError("report cost needs --hardware: the devices to draw")
    home = scalelab_home()
    validations: list[Validation] = []
    for name in sorted(set(hardware_names)):
        path = bench_folder(home, name) / VALIDATION_FILE
        if not path.is_file():
            raise UsageError(f"no {path}: run scalelab cost validate --hardware {name} first")
        validation = Validation.model_validate_json(path.read_text())
        if not validation.checks:  # saved before m3s26
            raise UsageError(
                f"{path} has no runs' steps: rerun scalelab cost validate --hardware {name}"
            )
        validations.append(validation)
        print(f"validation:    {path} ({len(validation.checks)} runs)")
    folder = reports_folder(home)
    measurements = [m for m in load_measurements(folder) if m.report.hardware in hardware_names]
    print(f"measured:      {len(measurements)} changes in {folder}")
    for path in write_report(validations, measurements, out):
        print(f"saved:         {path}")
    return 0


def cost_calibrate(hardware_name: str) -> int:
    """`scalelab cost calibrate`: fits the cost model's efficiencies to the results `scalelab
    bench` saved for this hardware, and saves them beside those results. Records nothing in the
    registry."""
    folder = bench_folder(scalelab_home(), hardware_name)
    try:
        samples, versions = load_samples(folder)
    except (FileNotFoundError, ValueError) as err:
        raise UsageError(str(err)) from None
    calibration = calibrate(samples, hardware_spec(hardware_name), versions)
    path = folder / CALIBRATION_FILE
    save_calibration(path, calibration)
    print(f"hardware:      {hardware_name}")
    print(f"measured on:   {', '.join(f'{key} {value}' for key, value in versions.items())}")
    print(f"samples:       {len(samples)} from {folder}")
    print(f"{'group':<15}{'compute':>8}{'memory':>8}{'launch':>11}   (1.000: never the limit)")
    for name, efficiency in calibration.efficiencies.items():
        launch = f"{efficiency.launch_s * 1e6:.2f} us"
        print(f"{name:<15}{efficiency.compute:8.3f}{efficiency.memory:8.3f}{launch:>11}")
    print(f"{'suite':<15}{'samples':>8}{'MAPE':>8}{'bias':>8}{'worst':>8}")
    for suite, errors in calibration.errors.items():
        mape, bias, worst = f"{errors.mape:.1%}", f"{errors.bias:+.1%}", f"{errors.worst:+.1%}"
        print(f"{suite:<15}{errors.samples:8d}{mape:>8}{bias:>8}{worst:>8}")
    print(f"saved:         {path}")
    return 0


def efficiencies_for(hardware_name: str) -> dict[Group, Efficiency] | None:
    """This hardware's efficiencies from `scalelab cost calibrate`, printing where they came
    from (and why they may be stale); None, the datasheet's peaks, when it has none (m3s10)."""
    path = bench_folder(scalelab_home(), hardware_name) / CALIBRATION_FILE
    if not path.is_file():
        print("calibration:   none, so the datasheet's peaks and free launches (a lower bound)")
        return None
    with warnings.catch_warnings(record=True) as caught:  # StaleCalibration, one a version
        warnings.simplefilter("always")
        efficiencies = load_calibration(path, env_versions(capture_env())).efficiencies
    print(f"calibration:   {path}")
    for warning in caught:
        print(f"warning:       {warning.message}")
    return efficiencies


def missing_half(efficiencies: Mapping[Group, Efficiency] | None) -> list[str]:
    """The fp16 groups a calibration has no fit for: it was fitted before the fp16 benches
    (m3s20). The datasheet's peaks (None) price fp16 too."""
    if efficiencies is None:
        return []
    return sorted(set(HALF.values()) - set(efficiencies))


def runtime_label(runtime: RuntimeConfig) -> str:
    """How a run computed its steps (m3s23): its precision, and "cut" if the cut was on."""
    return f"{runtime.precision} cut" if runtime.padding_cut else runtime.precision


def cost_predict(
    experiment: Path, overrides: Sequence[str], hardware_name: str, session_hours: float
) -> int:
    """`scalelab cost predict`: how long an RL experiment's steps would take on this hardware,
    phase by phase, and whether its whole run fits in a session. Uses the hardware's calibration
    from `scalelab cost calibrate` if there is one, else the datasheet's peaks with free
    launches (a lower bound). Runs nothing, records nothing."""
    config, _, _ = resolve(experiment, overrides)
    if config.rl is None:
        raise UsageError(f"cost predict times RL steps, and {experiment} is a warm start")
    print(f"experiment:    {config.name} (config {hashing.config_hash(config)})")
    print(f"hardware:      {hardware_name}")
    efficiencies = efficiencies_for(hardware_name)
    hardware = hardware_spec(hardware_name)
    run = predict_rl_run(config, hardware, efficiencies)
    answers = [length for lengths in run.spec.answer_lengths.values() for length in lengths]
    print(f"answers:       {sum(answers) / len(answers):.1f} tokens on average (worked solutions)")
    print(f"{'phase':<15}{'seconds':>8}{'share':>7}  bound    slowest op class")
    for name, phase in run.step.phases.items():
        share = f"{phase.seconds / run.step.seconds:.0%}"
        top = max(phase.by_class, key=phase.by_class.__getitem__)
        slowest = f"{top} {phase.by_class[top]:.3f} s"
        print(f"{name:<15}{phase.seconds:8.3f}{share:>7}  {phase.bottleneck:<9}{slowest}")
    print(f"{'step':<15}{run.step.seconds:8.3f}{'':>7}  {run.step.flops:.3g} FLOPs")
    line = f"run:           {run.steps} steps, {duration(run.seconds)}"
    if run.steps < run.budget_steps:
        line += f" (rl.max_steps: the FLOPs budget buys {run.budget_steps})"
    print(line)
    gradings = predict_gradings(config, run.steps, hardware, efficiencies)  # m4s3
    print(f"gradings:      {len(gradings.steps)}, at steps {', '.join(map(str, gradings.steps))}")
    print(
        f"               greedy {gradings.greedy:.1f} s each, sampled {gradings.sampled:.1f} s "
        f"at the first and last: {duration(gradings.seconds)}"
    )
    sessions = (run.seconds + gradings.seconds) / (session_hours * 3600)
    if sessions <= 1:
        fits = f"fits in one of {session_hours:g} h ({sessions:.1%} of it)"
    else:
        fits = f"needs {sessions:.1f} of {session_hours:g} h (it resumes from checkpoints)"
    print(f"session:       {fits}; gradings included, checkpoints come on top")
    return 0


# m4s4: sweep plan's session fit allows 1.35x the cost model's hours, its worst optimistic miss
# on a runtime M4 uses (m4s2 judged: a warm start of m1 in fp16)
SESSION_MARGIN = 1.35


class PlannedJob(NamedTuple):
    """One job of a sweep as sweep plan resolved and priced it (m4s5: what submit records)."""

    overrides: tuple[str, ...]
    config: ExperimentConfig
    seed: int
    seconds: float


def sweep_plan(
    path: Path, hardware_name: str, session_hours: float
) -> tuple[Sweep, list[PlannedJob], int]:
    """`scalelab sweep plan` (m4s4): every job a sweep file asks for, in order, each priced by
    the cost model on this hardware: its steps, hours and memory, and the sessions it needs at
    SESSION_MARGIN times its predicted hours. Checks that every job's config composes and that
    no two jobs are the same run. Runs nothing, records nothing. Returns the sweep, its jobs
    (m4s5: for sweep submit) and the exit code: 1 if a job needs more memory than the device
    has."""
    try:
        sweep = load_sweep(path)
    except (ValueError, OSError) as err:  # pydantic's ValidationError is a ValueError
        raise UsageError(f"{path}: {err}") from None
    jobs = expand(sweep)
    print(f"sweep:         {sweep.name}: {len(jobs)} jobs of {sweep.experiment}")
    print(f"hardware:      {hardware_name}")
    efficiencies = efficiencies_for(hardware_name)
    hardware = hardware_spec(hardware_name)
    usable = hardware.memory_bytes * (1 - RESERVED_FRACTION)
    seen: dict[tuple[str, int], int] = {}  # (config hash, seed) -> its job's number
    total = 0.0
    sessions: list[int] = []  # each job's, at SESSION_MARGIN
    too_big: list[int] = []  # the numbers of the jobs that need more memory than there is
    planned: list[PlannedJob] = []  # m4s5
    print(f"job  {'config':<18}seed  runtime     steps  s/step  hours   GiB  sessions")
    for number, overrides in enumerate(jobs, start=1):
        try:
            config, seed, runtime = resolve(sweep.experiment, overrides)
        except UsageError as err:
            raise UsageError(f"job {number} ({' '.join(overrides)}): {err}") from None
        run = (hashing.config_hash(config), seed)
        if run in seen:
            raise UsageError(
                f"jobs {seen[run]} and {number} are the same run: config {run[0]}, seed {seed}"
            )
        seen[run] = number
        job = predict_job(config, runtime, hardware, efficiencies)
        planned.append(PlannedJob(overrides, config, seed, job.seconds))
        total += job.seconds
        sessions.append(math.ceil(job.seconds * SESSION_MARGIN / (session_hours * 3600)))
        if job.memory > usable:
            too_big.append(number)
        fit = "too big" if number in too_big else str(sessions[-1])
        print(
            f"{number:>3}  {run[0]:<18}{seed:>4}  {runtime_label(runtime):<10}{job.steps:>7}"
            f"{job.step:>8.3f}{job.seconds / 3600:>7.2f}{job.memory / 2**30:>6.1f}  {fit}"
        )
    print(
        f"total:         {total / 3600:.1f} h predicted, "
        f"{total * SESSION_MARGIN / 3600:.1f} h at {SESSION_MARGIN}x"
    )
    one = f"one {session_hours:g} h session at {SESSION_MARGIN}x"
    if sessions.count(1) == len(jobs):
        print(f"sessions:      every job fits in {one}")
    else:
        print(
            f"sessions:      {sessions.count(1)} of {len(jobs)} jobs fit in {one}; "
            "the rest resume from checkpoints"
        )
    print("not priced:    checkpoints, start-up, and warm starts' gradings")
    if too_big:
        jobs_text = "job " if len(too_big) == 1 else "jobs "
        verb = "needs" if len(too_big) == 1 else "need"
        print(
            f"memory:        {jobs_text}{', '.join(map(str, too_big))} {verb} more than the "
            f"{usable / 2**30:.1f} GiB usable"
        )
        return sweep, planned, 1
    print(f"memory:        every job fits in the {usable / 2**30:.1f} GiB usable")
    return sweep, planned, 0


def sweep_submit(path: Path, hardware_name: str, session_hours: float) -> int:
    """`scalelab sweep submit` (m4s5): sweep plan, then, if every job fits the device, the sweep
    and its jobs into <home>'s registry, in plan's order, each with its predicted seconds, queued
    for `scalelab worker`. A job whose run is already done is recorded done. A sweep is
    submitted once at a commit. Refuses uncommitted changes: every job runs at this commit, and
    its run id names it."""
    env = capture_env()
    if env.git.dirty:
        raise UsageError(
            "commit your changes first: a sweep's jobs run at the commit it was submitted at"
        )
    sweep, planned, code = sweep_plan(path, hardware_name, session_hours)
    if code != 0:
        print("submitted:     nothing")
        return code
    sweep_id = f"{sweep.name}-{env.git.sha[:8]}"
    home = scalelab_home()
    with registry_edit(home) as registry:
        if registry.get_sweep(sweep_id) is None:
            done = {run.run_id for run in registry.list_runs() if run.status == "done"}
            new_jobs: list[Job] = []
            for number, job in enumerate(planned, start=1):
                run_id = hashing.run_id(job.config, job.seed, env.git.sha)
                new_jobs.append(
                    Job(
                        job_id=f"{sweep_id}-{number}",
                        sweep_id=sweep_id,
                        number=number,
                        run_id=run_id,
                        experiment=str(sweep.experiment),
                        overrides=job.overrides,
                        hardware=hardware_name,
                        seconds=job.seconds,
                        state="done" if run_id in done else "queued",
                    )
                )
            registry.enqueue(sweep_id, sweep.model_dump_json(), env.git.sha, new_jobs)
            print(f"submitted:     sweep {sweep_id}, {len(new_jobs)} jobs")
        else:
            print(f"submitted:     already (sweep {sweep_id}): nothing added")
        states = Counter(job.state for job in registry.list_jobs(sweep_id))
    counts = [f"{states[state]} {state}" for state in get_args(JobState) if states[state]]
    print(f"jobs:          {', '.join(counts)}")
    print(f"registry:      {registry_path(home)}")
    return 0


HEARTBEAT_S = 60.0  # m4s6: how often a worker renews the lease of the job it runs
LEASE_TTL_S = 600.0  # a lease lasts ten heartbeats: a long grading between two steps keeps it


def worker(overrides: Sequence[str], session_hours: float) -> int:
    """`scalelab worker` (m4s6): run the queue's jobs one after another, each in this process
    through `scalelab run`'s own code (resume, "already done" and exit codes are run's). It
    leases jobs of sweeps submitted at its own commit, priced on its own hardware, that fit the
    session's time left at SESSION_MARGIN; only its first lease may take a longer job (one
    longer than a whole session resumes from its checkpoints, session after session). A
    heartbeat every HEARTBEAT_S keeps a running job's lease. A run stopped by Ctrl+C gives its
    job back and stops the worker; a crash fails its job and stops it. `overrides`: runtime
    settings for every job, such as runtime.local_registry on Colab."""
    if any(not override.startswith("runtime.") for override in overrides):
        raise UsageError("a worker takes runtime settings only (runtime.<field>=<value>)")
    try:
        runtime = RuntimeConfig.model_validate(split_overrides(overrides)[1])
    except ValueError as err:  # pydantic's ValidationError is a ValueError
        raise UsageError(str(err)) from None
    env = capture_env()
    if env.git.dirty:
        raise UsageError("commit your changes first: a worker runs its jobs at its own commit")
    hardware = this_hardware()
    if hardware is None:
        raise UsageError(f"no hardware profile for {torch.cuda.get_device_name()}")
    name = f"{socket.gethostname()}-{os.getpid()}"  # the lease owner: this process
    home = scalelab_home(default=runtime.output_dir)
    at_home = registry_path(home)
    live = at_home  # as in run(): the registry the jobs' runs write to
    if runtime.local_registry:
        live = Path(runtime.local_registry).expanduser().resolve()
        if at_home.is_file():  # always: sweep submit wrote the copy at home
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(at_home, live)
    try:
        registry = Registry.sqlite(live)
    except OutdatedRegistryError as err:
        raise UsageError(str(err)) from None

    def copy_home() -> None:
        if live != at_home:
            atomic_write_bytes(at_home, live.read_bytes())

    def work(job: Job) -> int:
        """Run one leased job, renewing its lease every HEARTBEAT_S; its run's exit code."""
        last_beat = time.monotonic()

        def beat(_: object) -> None:  # after every training step
            nonlocal last_beat
            if time.monotonic() - last_beat >= HEARTBEAT_S:
                registry.heartbeat(job.job_id, name, LEASE_TTL_S)
                copy_home()
                last_beat = time.monotonic()

        registry.heartbeat(job.job_id, name, LEASE_TTL_S)  # leased -> running
        copy_home()
        try:
            return run(Path(job.experiment), [*job.overrides, *overrides], on_step=beat)
        except UsageError as err:  # it can't run (say, its warm start isn't done): it fails
            print(f"scalelab: error: {err}", file=sys.stderr)
            return EXIT_USAGE
        except Exception:
            registry.complete(job.job_id, name, "failed")
            copy_home()
            raise

    deadline = time.monotonic() + session_hours * 3600
    print(f"worker:        {name} at {env.git.sha[:8]} on {hardware}, {session_hours:g} h session")
    print(f"registry:      {live}")
    finished: Counter[str] = Counter()
    while True:
        fit = (deadline - time.monotonic()) / SESSION_MARGIN
        job = registry.lease(name, env.git.sha, hardware, fit, LEASE_TTL_S)
        if job is None and not finished:  # the session's first job may outlast it
            job = registry.lease(name, env.git.sha, hardware, math.inf, LEASE_TTL_S)
        if job is None:
            break
        print(f"job:           {job.job_id}, attempt {job.attempts}, {duration(job.seconds)}")
        code = work(job)
        if code == EXIT_INTERRUPTED:
            registry.release(job.job_id, name)
            copy_home()
            print(f"job:           {job.job_id} stopped: back in the queue")
            return EXIT_INTERRUPTED
        state = "done" if code == 0 else "failed"
        registry.complete(job.job_id, name, state)
        copy_home()
        finished[state] += 1
        print(f"job:           {job.job_id} {state}")
    left = max(deadline - time.monotonic(), 0) / 3600
    ran = ", ".join(f"{count} {state}" for state, count in finished.items()) or "none"
    print(f"worker:        nothing left to lease in the {left:.1f} h left (jobs run: {ran})")
    return 0


def ran_on(label: str, hardware_name: str) -> bool:
    """Did a run whose registry hardware label is `label` run on this one device? Runs on
    several GPUs ("2x Tesla T4") don't count: the step model times one device."""
    if hardware_name == "cpu":
        return label == "cpu"
    return re.match(r"\d+x ", label) is None and hardware_name.split("-")[0] in label.lower()


def this_hardware() -> str | None:
    """The hardware profile of the device runs use here (m3s25): cpu without a GPU, else the
    profile whose name its GPU's name holds as a word ("a100-40g": "NVIDIA A100-SXM4-40GB");
    None if none does."""
    if not torch.cuda.is_available():
        return "cpu"
    gpu = torch.cuda.get_device_name().lower()
    return next((name for name in HARDWARE if re.search(rf"\b{name.split('-')[0]}\b", gpu)), None)


def watch_throughput(config: ExperimentConfig, runtime: RuntimeConfig) -> ThroughputDetector | None:
    """The throughput detector for an RL run here (m3s25): the cost model's step as the run
    computes it, and the slowest step allowed by the errors cost validate saved for this device.
    None when there is nothing to judge by. Prints the prediction and the limit, or why not."""
    hardware_name = this_hardware()
    if hardware_name is None:
        print(f"throughput:    not watched ({torch.cuda.get_device_name()} has no profile)")
        return None
    path = bench_folder(scalelab_home(), hardware_name) / VALIDATION_FILE
    found = None
    if path.is_file():
        found = band(Validation.model_validate_json(path.read_text()), runtime_label(runtime))
    if found is None:
        print(f"throughput:    not watched (run cost validate --hardware {hardware_name} first)")
        return None
    part, median, spread = found
    efficiencies = efficiencies_for(hardware_name)  # what cost validate predicted with
    if runtime.precision == "fp16" and missing_half(efficiencies):
        print("throughput:    not watched (the calibration has no fp16 fit)")
        return None
    hardware = hardware_spec(hardware_name)
    predicted = predict_run_step(config, runtime, hardware, efficiencies).seconds
    limit = regression_limit(predicted, median, spread)
    if math.isinf(limit):
        print(f"throughput:    not watched ({part}: spread {spread:.1%}, too wide)")
        return None
    print(
        f"throughput:    {predicted:.3f} s a step predicted; {REGRESSION_STEPS} in a row above "
        f"{limit:.3f} s are a regression ({part}: median {median:+.1%}, spread {spread:.1%})"
    )
    return ThroughputDetector(predicted, limit)


def cost_validate(hardware_name: str) -> int:
    """`scalelab cost validate`: the cost model's step against the steps runs on this hardware
    measured (telemetry's laps: RL runs' since m2s29, warm starts' since m3s11), phase by phase.
    Every whole step is held out: the fit saw only single ops and decode steps (m3s7). Saves the
    errors beside calibration.json, for cost compare (m3s15); records nothing in the registry."""
    home = scalelab_home()
    hardware = hardware_spec(hardware_name)
    print(f"hardware:      {hardware_name}")
    efficiencies = efficiencies_for(hardware_name)
    # m3s23: (config_hash, runtime) -> its step: seeds share it, runtimes don't
    predictions: dict[tuple[str, str], StepPrediction] = {}
    checks: list[StepCheck] = []
    labels: dict[str, str] = {}  # run_id -> its runtime_label
    left_out: Counter[str] = Counter()  # why -> how many runs
    missing = missing_half(efficiencies)
    with registry_copy(home) as registry:
        for run in registry.list_runs():
            config_json = registry.experiment_json(run.config_hash)
            assert config_json is not None  # every run row points at an experiment row
            if not ran_on(run.hardware, hardware_name):
                continue  # another device
            try:
                config = ExperimentConfig.model_validate_json(config_json)
            except ValueError:  # recorded before a field was renamed or dropped
                left_out["with a config that no longer loads"] += 1
                continue
            steps, measured = measured_step(read_metrics(home, run.run_id))
            if steps == 0:  # from before m2s29, or with telemetry off
                left_out["with no timed steps"] += 1
                continue
            # m3s23: as the run ran. A run from before m3s23 recorded none: the defaults
            runtime = registry.get_runtime(run.run_id) or RuntimeConfig()
            if runtime.precision == "fp16" and missing:
                left_out["in fp16, with no fp16 fit"] += 1
                continue
            label = labels[run.run_id] = runtime_label(runtime)
            key = (run.config_hash, label)
            if key not in predictions:
                predictions[key] = predict_run_step(config, runtime, hardware, efficiencies)
            checks.append(check_step(run.run_id, predictions[key], steps, measured))
    if not checks:
        since = "RL runs time their steps since m2s29, warm starts since m3s11"
        raise UsageError(f"no run on {hardware_name} timed its steps ({since})")
    line = f"runs:          {len(checks)} runs timed their steps"
    if left_out:
        why = "; ".join(f"{count} {reason}" for reason, count in left_out.items())
        line += f" (left out {why})"
    print(line)
    parts = "".join(f"{part:>10}" for part in PARTS)
    print(f"{'run':<30}{'runtime':<10}{'steps':>6}{parts}  step, predicted vs measured")
    for check in checks:
        errors = "".join(  # m3s11: a warm start has train and step alone, blanks for the rest
            f"{check.errors[part]:>+10.1%}" if part in check.errors else " " * 10 for part in PARTS
        )
        step = f"{check.predicted['step']:.3f} s vs {check.measured['step']:.3f} s"
        print(f"{check.run_id:<30}{labels[check.run_id]:<10}{check.steps:6d}{errors}  {step}")
    summary = summarize(checks)
    if len(set(labels.values())) > 1:  # m3s23: each runtime's steps on their own too
        for label in sorted(set(labels.values())):
            same = [check for check in checks if labels[check.run_id] == label]
            summary[f"step, {label}"] = summarize(same)["step"]
    columns = f"{'MAPE':>8}{'bias':>8}{'worst':>8}{'median':>8}{'spread':>8}"  # m3s25: the band
    print(f"{'part':<15}{'runs':>6}{columns}")
    for part, errors in summary.items():
        mape, bias, worst = f"{errors.mape:.1%}", f"{errors.bias:+.1%}", f"{errors.worst:+.1%}"
        robust = f"{errors.median:>+8.1%}{errors.spread:>8.1%}"  # just computed: never None
        print(f"{part:<15}{errors.samples:6d}{mape:>8}{bias:>8}{worst:>8}{robust}")
    step = summary["step"].mape
    verdict = "within" if step <= TARGET_MAPE else "above"
    print(
        f"verdict:       step MAPE {step:.1%}, {verdict} the {TARGET_MAPE:.0%} M3's exit asks for"
    )
    validation = Validation(hardware=hardware_name, errors=summary, checks=checks)  # m3s26: checks
    path = bench_folder(home, hardware_name) / VALIDATION_FILE
    atomic_write_text(path, validation.model_dump_json(indent=2))
    print(f"saved:         {path}")
    return 0


def decision_report(
    config: ExperimentConfig,
    hardware_name: str,
    change: Change = "padding-cut",
    micro_batch: int | None = None,
) -> DecisionReport:
    """What a change would do to `config`'s steps on this hardware (m3s15; fp16: m3s20), with the
    last cost validate's error band there (validation.json), printing where both came from."""
    hardware = hardware_spec(hardware_name)
    efficiencies = efficiencies_for(hardware_name)
    missing = missing_half(efficiencies)
    runtime = CHANGE_RUNTIMES[change]  # m4s2
    if runtime.precision == "fp16" and missing:  # m3s20: fitted before the fp16 benches?
        raise UsageError(
            f"the {hardware_name} calibration has no {', '.join(missing)}: run `scalelab "
            f"bench --hardware {hardware_name} --precision fp16`, then `scalelab cost "
            f"calibrate --hardware {hardware_name}`"
        )
    if runtime.padding_cut:
        candidate = padding_cut(config, hardware, efficiencies, micro_batch, runtime.precision)
    else:
        candidate = predict_step(config, hardware, efficiencies, runtime.precision)
    path = bench_folder(scalelab_home(), hardware_name) / VALIDATION_FILE
    validation = Validation.model_validate_json(path.read_text()) if path.is_file() else None
    print(f"validation:    {path if validation else 'none (cost validate saves one)'}")
    return DecisionReport(
        change=change,
        experiment=config.name,
        config_hash=hashing.config_hash(config),
        hardware=hardware_name,
        baseline=predict_step(config, hardware, efficiencies),
        candidate=candidate,
        validated=validation.errors["step"] if validation else None,
    )


def save_report(stem: Path, report: DecisionReport | Measurement) -> None:
    """A report as Markdown and as JSON: <stem>.md and <stem>.json (m3s17)."""
    atomic_write_text(stem.with_suffix(".md"), report.markdown())
    atomic_write_text(stem.with_suffix(".json"), report.model_dump_json(indent=2))


def cost_compare(
    experiment: Path,
    overrides: Sequence[str],
    hardware_name: str,
    change: Change,
    micro_batch: int | None,
) -> int:
    """`scalelab cost compare` (m3s15): what a change (the padding cut, or fp16: m3s20) would do
    to an experiment's steps on this hardware, as a DecisionReport, printed and saved under
    <home>/reports/ as Markdown and JSON. Its error band is the last cost validate's on this
    hardware (validation.json). Runs nothing, records nothing in the registry."""
    if micro_batch is not None and micro_batch < 1:
        raise UsageError(f"--micro-batch must be at least 1, not {micro_batch}")
    if micro_batch is not None and not CHANGE_RUNTIMES[change].padding_cut:
        raise UsageError(f"--micro-batch sizes the padding cut's pieces, not {change}'s")
    config, _, _ = resolve(experiment, overrides)
    home = scalelab_home()
    report = decision_report(config, hardware_name, change, micro_batch)
    stem = reports_folder(home) / f"{report.change}-{report.config_hash}-{hardware_name}"
    save_report(stem, report)
    print(report.markdown())
    print(f"saved:         {stem}.md and .json")
    return 0


def cost_measure(
    experiment: Path,
    overrides: Sequence[str],
    hardware_name: str,
    change: Change,
    steps: int,
    warmup: int,
) -> int:
    """`scalelab cost measure` (m3s17): a change (the padding cut, or fp16: m3s24), measured.
    Predicts it as cost compare does and saves that report first; then runs the experiment's
    first steps twice on this machine, in fp32 without the cut and then with the change (RL from
    its warm start), and saves each part's measured seconds beside the prediction under
    <home>/reports/. Records nothing in the registry."""
    if steps < 1 or warmup < 1:
        raise UsageError(f"--steps and --warmup must be at least 1, got {steps} and {warmup}")
    config, seed, runtime = resolve(experiment, overrides)
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    if not ran_on(device, hardware_name):
        raise UsageError(f"--hardware {hardware_name}, but this machine runs steps on its {device}")
    home = scalelab_home()
    start = None
    if config.rl is not None:
        with registry_copy(home) as registry:
            start = load_warm_start(config.rl.from_run, config.model, registry, home)
    print(f"experiment:    {config.name} (config {hashing.config_hash(config)}), seed {seed}")
    print(f"device:        {device}")
    report = decision_report(config, hardware_name, change)
    stem = reports_folder(home) / f"{report.change}-{report.config_hash}-{hardware_name}"
    save_report(stem, report)  # the prediction, saved before anything runs
    before, after = report.baseline.seconds, report.candidate.seconds
    print(f"predicted:     step {before:.4f} -> {after:.4f} s ({-report.saving:+.1%}), {stem}.md")
    print(f"steps:         {warmup} to warm up, then {steps} timed; without, then with {change}")
    deterministic = runtime.deterministic
    if deterministic is None:
        deterministic = default_deterministic()
    try:
        timed = measure_change(config, seed, change, start, deterministic, steps, warmup)
    except ValueError as err:  # a warm start whose budget ends sooner
        raise UsageError(str(err)) from None
    measurement = Measurement(report=report, steps=steps, baseline=timed.off, candidate=timed.on)
    measured = stem.with_name(f"{stem.name}-measured")
    save_report(measured, measurement)
    print(measurement.markdown())
    print(f"saved:         {measured}.md and .json")
    return 0


def cost_comms(worlds: Sequence[int], backend: Backend) -> int:
    """`scalelab cost comms`: times all-reduce, all-gather and reduce-scatter among processes on
    this machine (gloo: CPU processes; nccl: one GPU each) at each world size, fits a link
    (alpha, beta) to the all-reduce at every other message size, and checks the formulas on the
    other timings: the all-reduce's other sizes and every size of the other two. One link a
    world size: processes on one machine share its memory, so more of them get less bandwidth
    each. Saves and records nothing."""
    if min(worlds) < 2:
        raise UsageError("--world: a collective needs 2 processes or more")
    gpus = torch.cuda.device_count()
    if backend == "nccl" and gpus < max(worlds):
        raise UsageError(
            f"--world {max(worlds)} over nccl needs as many GPUs; this machine has {gpus}"
        )
    sizes = ", ".join(f"{n >> 20} MB" if n >= 1 << 20 else f"{n >> 10} KB" for n in SIZES[::2])
    print(f"backend:       {backend}")
    print(f"fit:           all_reduce at {sizes}; checked on the rest")
    rows: list[str] = []
    for world in worlds:
        start = time.perf_counter()
        measured = bench_collectives(world, backend)
        by_collective = {c: [m for m in measured if m.collective == c] for c in COLLECTIVES}
        fit = by_collective["all_reduce"][::2]  # each collective's timings are in SIZES order
        link = fit_link("all_reduce", world, [(m.nbytes, m.seconds) for m in fit])
        print(
            f"world {world}:       {len(measured)} timings in "
            f"{duration(time.perf_counter() - start)}: "
            f"alpha {link.alpha * 1e6:.1f} us, beta {link.beta / 1e9:.2f} GB/s",
            flush=True,
        )
        for collective in COLLECTIVES:
            timings = by_collective[collective]
            held_out = timings[1::2] if collective == "all_reduce" else timings
            errors = [
                collective_time(collective, m.nbytes, world, link) / m.seconds - 1 for m in held_out
            ]
            mape = f"{sum(abs(e) for e in errors) / len(errors):.1%}"
            bias, worst = f"{sum(errors) / len(errors):+.1%}", f"{max(errors, key=abs):+.1%}"
            rows.append(f"{collective:<15}{world:6d}{len(errors):6d}{mape:>8}{bias:>8}{worst:>8}")
    print(f"{'collective':<15}{'world':>6}{'sizes':>6}{'MAPE':>8}{'bias':>8}{'worst':>8}")
    print("\n".join(rows))
    return 0


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
    bands_parser = commands.add_parser(
        "bands", help="warm starts graded on * band by band, and the bands M4's RL task keeps"
    )
    bands_parser.add_argument("run_ids", nargs="+", help="two or more finished warm starts")
    bands_parser.add_argument(
        "--problems", type=int, default=1000, help="the first N of each eval set (default 1000)"
    )
    fit_parser = commands.add_parser(
        "fit", help="RL runs' pass rate against compute, and the sigmoid fit through it"
    )
    fit_parser.add_argument("run_ids", nargs="+", help="one or more finished RL runs")
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
    report_parser = commands.add_parser("report", help="print a report of the recorded runs")
    report_parser.add_argument(
        "report",
        choices=["gains", "cost"],
        help="gains: what RL did to greedy pass@1, over seeds; cost: Write-up 1's figures, tables",
    )
    report_parser.add_argument(
        "--hardware", nargs="+", default=[], help="cost: the devices, each cost validated"
    )
    report_parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/writeups/figs"),
        help="cost: where the figures and tables go (default docs/writeups/figs)",
    )

    bench_parser = commands.add_parser(
        "bench", help="time the cost model's ops on this machine, for calibration"
    )
    bench_parser.add_argument(
        "--hardware",
        required=True,
        choices=["cpu", *HARDWARE],
        help="the device's profile: results go to <home>/bench/<hardware>/",
    )
    bench_parser.add_argument(
        "--models",
        nargs="+",
        default=list(BENCH_MODELS),
        help="model configs whose shapes are timed (default: ladder_s/m1 to ladder_s/m4)",
    )
    bench_parser.add_argument(
        "--suites",
        nargs="+",
        choices=SUITES,
        help="default: all five (fp16: gemm attention decode)",
    )
    bench_parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="fp32",
        help="fp16: float16 kernels as autocast runs them, saved as <suite>-fp16.parquet",
    )
    bench_parser.add_argument(
        "--dry-run", action="store_true", help="say what would be timed, and time nothing"
    )

    cost_parser = commands.add_parser("cost", help="the cost model: fit it, predict with it")
    cost_commands = cost_parser.add_subparsers(dest="cost_command", required=True)
    calibrate_parser = cost_commands.add_parser(
        "calibrate", help="fit the cost model to this hardware's bench results"
    )
    calibrate_parser.add_argument(
        "--hardware",
        required=True,
        choices=["cpu", *HARDWARE],
        help="reads <home>/bench/<hardware>/ and saves calibration.json there",
    )
    predict_parser = cost_commands.add_parser(
        "predict", help="how long an RL experiment's steps and whole run would take"
    )
    predict_parser.add_argument("experiment", type=Path, help="e.g. configs/experiments/rl.yaml")
    predict_parser.add_argument("overrides", nargs="*", help="key=value changes, as for run")
    predict_parser.add_argument(
        "--hardware",
        required=True,
        choices=["cpu", *HARDWARE],
        help="the device: its calibration.json from cost calibrate, when it has one",
    )
    predict_parser.add_argument(
        "--session-hours",
        type=float,
        default=12.0,
        help="how long a session lasts (default 12: Colab's free tier)",
    )
    validate_parser = cost_commands.add_parser(
        "validate", help="the cost model's steps against the RL steps timed on this hardware"
    )
    validate_parser.add_argument(
        "--hardware",
        required=True,
        choices=["cpu", *HARDWARE],
        help="its RL runs, and its calibration.json when it has one",
    )
    compare_parser = cost_commands.add_parser(
        "compare", help="what a change would do to an experiment's steps (a DecisionReport)"
    )
    compare_parser.add_argument("experiment", type=Path, help="e.g. configs/experiments/rl.yaml")
    compare_parser.add_argument("overrides", nargs="*", help="key=value changes, as for run")
    compare_parser.add_argument(
        "--hardware",
        required=True,
        choices=["cpu", *HARDWARE],
        help="the device: its calibration.json, and validation.json for the error band",
    )
    compare_parser.add_argument(
        "--change",
        choices=list(CHANGES),
        default="padding-cut",
        help="the padding cut (DecisionReport 1, the default), fp16 autocast (2) or both",
    )
    compare_parser.add_argument(
        "--micro-batch",
        type=int,
        help="rows a micro-batch under the cut (default: rl.micro_batch, or a quarter of a warm "
        "start's batch)",
    )
    measure_parser = cost_commands.add_parser(
        "measure", help="a change, measured: the same steps without it, then with it"
    )
    measure_parser.add_argument("experiment", type=Path, help="e.g. configs/experiments/rl.yaml")
    measure_parser.add_argument("overrides", nargs="*", help="key=value changes, as for run")
    measure_parser.add_argument(
        "--change",
        choices=list(CHANGES),
        default="padding-cut",
        help="the padding cut (the default), fp16 autocast (m3s24) or both (m4s2)",
    )
    measure_parser.add_argument(
        "--hardware",
        required=True,
        choices=["cpu", *HARDWARE],
        help="this machine's device: the prediction's calibration and error band",
    )
    measure_parser.add_argument(
        "--steps", type=int, default=20, help="steps timed in each run (default 20)"
    )
    measure_parser.add_argument(
        "--warmup", type=int, default=3, help="steps before the timed ones (default 3)"
    )

    comms_parser = cost_commands.add_parser(
        "comms", help="time collectives among processes here, and check the formulas on them"
    )
    comms_parser.add_argument(
        "--world", nargs="+", type=int, default=[2, 4], help="processes taking part (default 2 4)"
    )
    comms_parser.add_argument(
        "--backend",
        choices=["gloo", "nccl"],
        default="gloo",
        help="gloo: processes on the CPU (default); nccl: one GPU each",
    )

    sweep_parser = commands.add_parser("sweep", help="many runs of one experiment")
    sweep_commands = sweep_parser.add_subparsers(dest="sweep_command", required=True)
    sweep_help = {
        "plan": "list a sweep's jobs and price each one; runs and records nothing",
        "submit": "sweep plan, then record the sweep and queue its jobs for scalelab worker",
    }
    for name, text in sweep_help.items():  # m4s5: submit takes plan's arguments
        sweep_command = sweep_commands.add_parser(name, help=text)
        sweep_command.add_argument("sweep", type=Path, help="e.g. configs/sweeps/warmstarts.yaml")
        sweep_command.add_argument(
            "--hardware",
            required=True,
            choices=["cpu", *HARDWARE],
            help="the device the jobs will run on: its calibration.json, when it has one",
        )
        sweep_command.add_argument(
            "--session-hours",
            type=float,
            default=12.0,
            help="how long a session lasts (default 12: Colab's free tier)",
        )
    worker_parser = commands.add_parser(
        "worker", help="run the queued jobs one after another, while they fit the time left"
    )
    worker_parser.add_argument(
        "overrides",
        nargs="*",
        help="runtime settings for every job, e.g. runtime.local_registry=/content/registry.db",
    )
    worker_parser.add_argument(
        "--session-hours",
        type=float,
        default=12.0,
        help="how long this session lasts (default 12: Colab's free tier)",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "grade":
            return grade(args.run_id, args.problems)
        if args.command == "bands":
            return bands(args.run_ids, args.problems)
        if args.command == "fit":
            return fit(args.run_ids)
        if args.command == "overhead":
            return overhead(args.experiment, args.overrides, args.steps, args.warmup)
        if args.command == "report" and args.report == "cost":
            return report_cost(args.hardware, args.out)
        if args.command == "report":
            return report_gains()
        if args.command == "bench":
            return bench(args.hardware, args.models, args.suites, args.precision, args.dry_run)
        if args.command == "cost" and args.cost_command == "calibrate":
            return cost_calibrate(args.hardware)
        if args.command == "cost" and args.cost_command == "validate":
            return cost_validate(args.hardware)
        if args.command == "cost" and args.cost_command == "compare":
            return cost_compare(
                args.experiment, args.overrides, args.hardware, args.change, args.micro_batch
            )
        if args.command == "cost" and args.cost_command == "measure":
            return cost_measure(
                args.experiment,
                args.overrides,
                args.hardware,
                args.change,
                args.steps,
                args.warmup,
            )
        if args.command == "cost" and args.cost_command == "comms":
            return cost_comms(args.world, args.backend)
        if args.command == "sweep" and args.sweep_command == "submit":
            return sweep_submit(args.sweep, args.hardware, args.session_hours)
        if args.command == "sweep":
            return sweep_plan(args.sweep, args.hardware, args.session_hours)[2]
        if args.command == "cost":
            return cost_predict(args.experiment, args.overrides, args.hardware, args.session_hours)
        if args.command == "worker":
            return worker(args.overrides, args.session_hours)
        return run(args.experiment, args.overrides)
    except UsageError as err:
        print(f"scalelab: error: {err}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
