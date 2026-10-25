"""The `scalelab` command."""

import argparse
import contextlib
import json
import re
import shutil
import signal
import sys
import tempfile
import time
import warnings
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from scalelab.analysis.reports import gains
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
from scalelab.costmodel.calibrate import calibrate, env_versions, load_calibration, save_calibration
from scalelab.costmodel.comms import COLLECTIVES, collective_time, fit_link
from scalelab.costmodel.report import CHANGES, Change, DecisionReport, Measurement, reports_folder
from scalelab.costmodel.rlstep import StepPrediction
from scalelab.costmodel.roofline import HALF, Efficiency, Group
from scalelab.costmodel.specs import CPU, HARDWARE, HardwareSpec, Precision
from scalelab.costmodel.validate import (
    PARTS,
    TARGET_MAPE,
    VALIDATION_FILE,
    StepCheck,
    Validation,
    check_step,
    summarize,
)
from scalelab.models.transformer import Transformer
from scalelab.rl.costing import measured_step, padding_cut, predict_rl_run, predict_step
from scalelab.rl.evaluation import EvalResult, SampledResult
from scalelab.rl.experiment import (
    grade_by_part,
    load_eval_sets,
    make_task,
    run_experiment,
    run_rl_experiment,
)
from scalelab.rl.finished import load_finished_run
from scalelab.rl.gradings import Grading, gradings_folder, load_gradings, save_grading
from scalelab.rl.loop import SNAPSHOTS, RLStepMetrics
from scalelab.rl.measure import measure_padding_cut
from scalelab.rl.overhead import OVERHEAD_BUDGET, measure_overhead
from scalelab.rl.warmstart_loop import StepMetrics
from scalelab.telemetry.anomalies import AnomalyEvent, RunHalted
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
    start = None  # RL's starting model, checked before anything is recorded
    if config.rl is not None:
        if runtime.precision != "fp32":  # m3s21: warm starts only, until m3s22
            raise UsageError(f"runtime.precision={runtime.precision}: RL runs in fp32 until m3s22")
        start = load_warm_start(config.rl.from_run, config.model, registry, home)
        print(f"warm start:    {config.rl.from_run}")
        print(f"gradings:      {gradings_folder(home, run_id)}")  # m2s38
    telemetry = metrics_folder(home, run_id) if runtime.telemetry else "off"  # m2s36
    print(f"metrics:       {telemetry}")  # m3s11: a warm start's step times go there too

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
    run = predict_rl_run(config, hardware_spec(hardware_name), efficiencies)
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
    sessions = run.seconds / (session_hours * 3600)
    if sessions <= 1:
        fits = f"fits in one of {session_hours:g} h ({sessions:.1%} of it)"
    else:
        fits = f"needs {sessions:.1f} of {session_hours:g} h (it resumes from checkpoints)"
    print(f"session:       {fits}; gradings and checkpoints come on top")
    return 0


def ran_on(label: str, hardware_name: str) -> bool:
    """Did a run whose registry hardware label is `label` run on this one device? Runs on
    several GPUs ("2x Tesla T4") don't count: the step model times one device."""
    if hardware_name == "cpu":
        return label == "cpu"
    return re.match(r"\d+x ", label) is None and hardware_name.split("-")[0] in label.lower()


def cost_validate(hardware_name: str) -> int:
    """`scalelab cost validate`: the cost model's step against the steps runs on this hardware
    measured (telemetry's laps: RL runs' since m2s29, warm starts' since m3s11), phase by phase.
    Every whole step is held out: the fit saw only single ops and decode steps (m3s7). Saves the
    errors beside calibration.json, for cost compare (m3s15); records nothing in the registry."""
    home = scalelab_home()
    hardware = hardware_spec(hardware_name)
    print(f"hardware:      {hardware_name}")
    efficiencies = efficiencies_for(hardware_name)
    predictions: dict[str, StepPrediction] = {}  # config_hash -> its step: seeds share it
    checks: list[StepCheck] = []
    left_out = 0
    with registry_copy(home) as registry:
        for run in registry.list_runs():
            config_json = registry.experiment_json(run.config_hash)
            assert config_json is not None  # every run row points at an experiment row
            if not ran_on(run.hardware, hardware_name):
                continue  # another device
            try:
                config = ExperimentConfig.model_validate_json(config_json)
            except ValueError:  # recorded before a field was renamed or dropped
                left_out += 1
                continue
            steps, measured = measured_step(read_metrics(home, run.run_id))
            if steps == 0:  # from before m2s29, or with telemetry off
                left_out += 1
                continue
            if run.config_hash not in predictions:
                predictions[run.config_hash] = predict_step(config, hardware, efficiencies)
            checks.append(check_step(run.run_id, predictions[run.config_hash], steps, measured))
    if not checks:
        since = "RL runs time their steps since m2s29, warm starts since m3s11"
        raise UsageError(f"no run on {hardware_name} timed its steps ({since})")
    line = f"runs:          {len(checks)} runs timed their steps"
    if left_out:
        line += f" (left out {left_out}: no timed steps, or a config that no longer loads)"
    print(line)
    parts = "".join(f"{part:>10}" for part in PARTS)
    print(f"{'run':<30}{'steps':>6}{parts}  step, predicted vs measured")
    for check in checks:
        errors = "".join(  # m3s11: a warm start has train and step alone, blanks for the rest
            f"{check.errors[part]:>+10.1%}" if part in check.errors else " " * 10 for part in PARTS
        )
        step = f"{check.predicted['step']:.3f} s vs {check.measured['step']:.3f} s"
        print(f"{check.run_id:<30}{check.steps:6d}{errors}  {step}")
    summary = summarize(checks)
    print(f"{'part':<15}{'runs':>6}{'MAPE':>8}{'bias':>8}{'worst':>8}")
    for part, errors in summary.items():
        mape, bias, worst = f"{errors.mape:.1%}", f"{errors.bias:+.1%}", f"{errors.worst:+.1%}"
        print(f"{part:<15}{errors.samples:6d}{mape:>8}{bias:>8}{worst:>8}")
    step = summary["step"].mape
    verdict = "within" if step <= TARGET_MAPE else "above"
    print(
        f"verdict:       step MAPE {step:.1%}, {verdict} the {TARGET_MAPE:.0%} M3's exit asks for"
    )
    validation = Validation(hardware=hardware_name, errors=summary)  # m3s15: for cost compare
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
    if change == "fp16" and efficiencies is not None:  # m3s20: fitted before the fp16 benches?
        missing = sorted(set(HALF.values()) - set(efficiencies))
        if missing:
            raise UsageError(
                f"the {hardware_name} calibration has no {', '.join(missing)}: run `scalelab "
                f"bench --hardware {hardware_name} --precision fp16`, then `scalelab cost "
                f"calibrate --hardware {hardware_name}`"
            )
    if change == "fp16":
        candidate = predict_step(config, hardware, efficiencies, precision="fp16")
    else:
        candidate = padding_cut(config, hardware, efficiencies, micro_batch)
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
    if micro_batch is not None and change != "padding-cut":
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
    experiment: Path, overrides: Sequence[str], hardware_name: str, steps: int, warmup: int
) -> int:
    """`scalelab cost measure` (m3s17): the padding cut, measured. Predicts it as cost compare
    does and saves that report first; then runs the experiment's first steps twice on this
    machine, without the cut and with it (RL from its warm start), and saves each part's measured
    seconds beside the prediction under <home>/reports/. Records nothing in the registry."""
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
    report = decision_report(config, hardware_name)
    stem = reports_folder(home) / f"{report.change}-{report.config_hash}-{hardware_name}"
    save_report(stem, report)  # the prediction, saved before anything runs
    before, after = report.baseline.seconds, report.candidate.seconds
    print(f"predicted:     step {before:.4f} -> {after:.4f} s ({-report.saving:+.1%}), {stem}.md")
    print(f"steps:         {warmup} to warm up, then {steps} timed; without, then with the cut")
    deterministic = runtime.deterministic
    if deterministic is None:
        deterministic = default_deterministic()
    try:
        timed = measure_padding_cut(config, seed, start, deterministic, steps, warmup)
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
        "report", choices=["gains"], help="gains: what RL did to greedy pass@1, over seeds"
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
        help="the padding cut (DecisionReport 1, the default) or fp16 autocast (2)",
    )
    compare_parser.add_argument(
        "--micro-batch",
        type=int,
        help="rows a micro-batch under the cut (default: rl.micro_batch, or a quarter of a warm "
        "start's batch)",
    )
    measure_parser = cost_commands.add_parser(
        "measure", help="the padding cut, measured: the same steps without it, then with it"
    )
    measure_parser.add_argument("experiment", type=Path, help="e.g. configs/experiments/rl.yaml")
    measure_parser.add_argument("overrides", nargs="*", help="key=value changes, as for run")
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

    args = parser.parse_args(argv)
    try:
        if args.command == "grade":
            return grade(args.run_id, args.problems)
        if args.command == "overhead":
            return overhead(args.experiment, args.overrides, args.steps, args.warmup)
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
                args.experiment, args.overrides, args.hardware, args.steps, args.warmup
            )
        if args.command == "cost" and args.cost_command == "comms":
            return cost_comms(args.world, args.backend)
        if args.command == "cost":
            return cost_predict(args.experiment, args.overrides, args.hardware, args.session_hours)
        return run(args.experiment, args.overrides)
    except UsageError as err:
        print(f"scalelab: error: {err}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
