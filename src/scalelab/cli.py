"""The `scalelab` command."""

import argparse
import shutil
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scalelab.core import hashing
from scalelab.core.checkpoint import CheckpointPolicy
from scalelab.core.config import RuntimeConfig, load_config, parse_override
from scalelab.core.determinism import MAX_SEED, default_deterministic
from scalelab.core.envcapture import EnvInfo, capture_env
from scalelab.core.registry import EvalRecord, OutdatedRegistryError, Registry
from scalelab.core.storage import atomic_write_bytes, registry_path, scalelab_home
from scalelab.rl.evaluation import EvalResult
from scalelab.rl.experiment import run_experiment
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

    def report(metrics: StepMetrics, results: list[EvalResult]) -> None:
        """Record each eval result in the registry, and print one line of progress."""
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
        print(f"step {metrics.step:,}: loss {metrics.loss:.3f}  {rates}")

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
        f"done: ran {len(metrics)} steps ({last.tokens:,} tokens and {last.flops:.2e} FLOPs "
        f"in all), last loss {last.loss!r}"
    )
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
    args = parser.parse_args(argv)
    try:
        return run(args.experiment, args.overrides)
    except UsageError as err:
        print(f"scalelab: error: {err}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
