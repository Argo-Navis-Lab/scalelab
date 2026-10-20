"""The `scalelab` command."""

import argparse
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from scalelab.core import hashing
from scalelab.core.checkpoint import CheckpointPolicy
from scalelab.core.config import RuntimeConfig, load_config, parse_override
from scalelab.core.determinism import MAX_SEED, default_deterministic
from scalelab.core.envcapture import EnvInfo, capture_env
from scalelab.core.registry import EvalRecord, OutdatedRegistryError, Registry
from scalelab.core.storage import atomic_write_bytes, registry_path, scalelab_home
from scalelab.rl.evaluation import EvalResult
from scalelab.rl.experiment import grade_by_part, load_eval_sets, make_task, run_experiment
from scalelab.rl.finished import load_finished_run
from scalelab.rl.warmstart_loop import StepMetrics

EXIT_USAGE = 2  # the command line asked for something impossible
EXIT_INTERRUPTED = 130  # stopped early with a checkpoint saved; run again to resume
METRIC = "greedy_pass@1"  # the only metric so far; sampled pass@k arrives with m2's sampling


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


def same_uncommitted_changes(recorded_env_json: str, env: EnvInfo) -> bool:
    """Was the run recorded with exactly the uncommitted changes we have now (or none)?"""
    recorded = EnvInfo.model_validate_json(recorded_env_json)
    return recorded.git.diff_sha256 == env.git.diff_sha256


def run(experiment: Path, overrides: Sequence[str]) -> int:
    """`scalelab run`: resolve the experiment, record it, run it (or resume it), finish it."""
    try:
        seed, runtime_values, config_overrides = split_overrides(overrides)
        config = load_config(experiment, config_overrides)
        runtime = RuntimeConfig.model_validate(runtime_values)
        if config.rl is not None:  # until m2s22, run_experiment would warm-start it instead
            raise ValueError("RL experiments can't run yet: `scalelab run` learns to in m2s22")
    except (ValueError, FileNotFoundError) as err:  # pydantic's ValidationError is a ValueError
        raise UsageError(str(err)) from None

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

    # Speed is measured between gradings, from the time and token count at the last one. A
    # resumed run's starting token count isn't known here, so its first grading shows none.
    last_tokens = 0 if existing is None else None

    def report(metrics: StepMetrics, results: list[EvalResult]) -> None:
        """Record each eval result in the registry, and print one line of progress: the
        grades, and the speed since the last grading, with the time it gives to the end."""
        nonlocal last_time, last_tokens
        for result in results:
            registry.record_eval(
                EvalRecord(
                    run_id=run_id,
                    step=metrics.step,
                    task=result.name,
                    metric=METRIC,
                    value=result.pass_rate,
                    n=result.n,
                    stderr=result.stderr,
                )
            )
        copy_home()
        rates = "  ".join(f"{result.name} {result.pass_rate:.1%}" for result in results)
        now, pace = time.monotonic(), ""
        if last_tokens is not None and now > last_time:
            read = metrics.tokens - last_tokens  # tokens read since the last grading
            pace = f"  ({speed(read, now - last_time, metrics.tokens, config.warm_start.tokens)})"
        last_time, last_tokens = now, metrics.tokens
        print(f"step {metrics.step:,}: loss {metrics.loss:.3f}  {rates}{pace}")

    policy = CheckpointPolicy()
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
            metrics = run_experiment(
                config,
                seed,
                home / "checkpoints" / run_id,
                home=home,
                policy=policy,
                deterministic=deterministic,
                report=report,
            )
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
    last = metrics[-1]
    print(
        f"done: ran {len(metrics)} steps in {duration(time.monotonic() - started)} "
        f"({last.tokens:,} tokens and {last.flops:.2e} FLOPs in all), last loss {last.loss!r}"
    )
    return 0


def grade(run_id: str, problems: int) -> int:
    """`scalelab grade`: greedy pass@1 of a finished run's model on the frozen eval sets of
    its task, whole and by part (arithmetic's operations). Prints the grades, records nothing."""
    if problems < 1:
        raise UsageError(f"--problems must be at least 1, got {problems}")
    home = scalelab_home()
    source = registry_path(home)
    if not source.is_file():
        raise UsageError(f"no registry at {source}")
    with tempfile.TemporaryDirectory() as folder:
        # Read a private copy, so the registry itself is never locked or changed: on Colab
        # it lives on Google Drive, where SQLite is unreliable.
        copy = Path(folder) / "registry.db"
        shutil.copyfile(source, copy)
        try:
            finished = load_finished_run(Registry.sqlite(copy), home, run_id)
        except (ValueError, OutdatedRegistryError) as err:
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
    args = parser.parse_args(argv)
    try:
        if args.command == "grade":
            return grade(args.run_id, args.problems)
        return run(args.experiment, args.overrides)
    except UsageError as err:
        print(f"scalelab: error: {err}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
