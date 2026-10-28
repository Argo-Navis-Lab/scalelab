"""The m0 exit criterion: one command records a complete run row."""

import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest
import torch

from scalelab.analysis.reports import gains
from scalelab.analysis.scaling import (
    ALL_OPS,
    NEAR,
    cumulative_compute,
    fit_run,
    pooled_rate,
    run_points,
    step_reaching,
)
from scalelab.analysis.stats import EFFECT, detection_power, seeds_for_power
from scalelab.cli import main
from scalelab.core.checkpoint import latest_checkpoint, load_checkpoint
from scalelab.core.config import RuntimeConfig, load_config
from scalelab.core.envcapture import EnvInfo
from scalelab.core.registry import Registry
from scalelab.costmodel.calibrate import Calibration, Errors, save_calibration
from scalelab.costmodel.roofline import Efficiency, Group
from scalelab.costmodel.specs import CPU
from scalelab.costmodel.validate import PARTS, Validation, check_step
from scalelab.rl.costing import measured_step, predict_rl_run, predict_run_step, predict_step
from scalelab.rl.gradings import load_gradings
from scalelab.telemetry.detectors import REGRESSION_STEPS, regression_limit
from scalelab.telemetry.sink import metrics_folder, read_metrics

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
RL = HELLO.parent / "rl.yaml"


@pytest.fixture
def home(monkeypatch, tmp_path):
    """Point SCALELAB_HOME at a fresh temporary folder for this test."""
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path))
    return tmp_path.resolve()


def last_loss(output: str) -> str:
    return output.strip().splitlines()[-1].split("last loss ")[1]


def test_one_command_records_a_complete_run_row(home, capsys):
    assert main(["run", str(HELLO), "seed=3"]) == 0

    registry = Registry.sqlite(home / "registry.db")
    (run,) = registry.list_runs()
    assert run.run_id.startswith("b38a7209a4d8bd8b-s3-")
    assert run.config_hash == "b38a7209a4d8bd8b"
    assert run.seed == 3
    assert run.status == "done"
    assert run.deterministic is True  # tests run on CPUs, where it is on by default
    assert run.executor == "local"
    assert run.hardware == "cpu"
    assert run.ended_at is not None
    assert run.started_at <= run.ended_at
    assert EnvInfo.model_validate_json(run.env_json).git.sha.startswith(run.run_id[-8:])
    assert registry.experiment_json(run.config_hash) is not None
    assert "done: ran 16 steps" in capsys.readouterr().out
    graded = [(e.step, e.task, e.metric, e.n) for e in registry.list_evals(run.run_id)]
    assert graded == [(16, f"arithmetic-{d}", "greedy_pass@1", 20) for d in (2, 3, 4, 5)]


def test_running_it_again_does_nothing(home, capsys):
    main(["run", str(HELLO)])
    assert main(["run", str(HELLO)]) == 0
    assert "already done" in capsys.readouterr().out
    assert len(Registry.sqlite(home / "registry.db").list_runs()) == 1


def test_runtime_settings_do_not_change_the_run(home, capsys):
    main(["run", str(HELLO)])
    main(["run", str(HELLO), "runtime.worker_name=colab-1"])  # same run_id: already done
    assert "already done" in capsys.readouterr().out


def test_overrides_make_a_different_experiment(home):
    main(["run", str(HELLO), "model=ladder_s/m2"])
    (run,) = Registry.sqlite(home / "registry.db").list_runs()
    assert run.config_hash != "b38a7209a4d8bd8b"


@pytest.mark.parametrize(
    "bad",
    [
        "model.n_layer=6",
        "nonsense",
        "model=ladder_s/m9",
        "seed=-1",
        "runtime.colour=red",
        "task.difficulties=[13]",
        "task.ops=*+",
        "algo=grpo",
    ],
)
def test_bad_input_is_one_clear_error_and_records_nothing(home, capsys, bad):
    assert main(["run", str(HELLO), bad]) == 2
    assert capsys.readouterr().err.startswith("scalelab: error: ")
    assert not (home / "registry.db").exists()


def test_old_registry_file_is_reported_clearly(home, capsys):
    registry = Registry.sqlite(home / "registry.db")
    with registry.engine.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE runs DROP COLUMN deterministic")
    assert main(["run", str(HELLO)]) == 2
    assert "created by an older version of ScaleLab" in capsys.readouterr().err


def test_interrupted_run_resumes_to_the_same_result(home, tmp_path, capsys, monkeypatch):
    longer = ["seed=4", "warm_start.tokens=50000"]  # a few seconds: time for the signal to land
    main(["run", str(HELLO), *longer])  # the uninterrupted reference, in `home`
    expected = last_loss(capsys.readouterr().out)

    other_home = tmp_path / "interrupted"
    job = subprocess.Popen(
        [sys.executable, "-u", "-m", "scalelab.cli", "run", str(HELLO), *longer],
        env={**os.environ, "SCALELAB_HOME": str(other_home)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert job.stdout is not None  # we asked for it with stdout=subprocess.PIPE
    for line in job.stdout:  # wait until the run is recorded and training
        if line.startswith("started"):
            break
    job.send_signal(signal.SIGTERM)
    output, _ = job.communicate(timeout=60)
    assert job.returncode == 130, output
    assert "stopped early, checkpoint saved" in output
    (run,) = Registry.sqlite(other_home / "registry.db").list_runs()
    assert run.status == "running"

    monkeypatch.setenv("SCALELAB_HOME", str(other_home))  # now resume it, in-process
    assert main(["run", str(HELLO), *longer]) == 0
    output = capsys.readouterr().out
    assert "resuming (it was running)" in output
    assert last_loss(output) == expected
    (run,) = Registry.sqlite(other_home / "registry.db").list_runs()
    assert run.status == "done"


# --- added in m1s22: a registry on the machine's own disk, copied home ----------------------


def on_disk_of(machine: Path) -> str:
    """The override that keeps the live registry on `machine`'s own disk."""
    return f"runtime.local_registry={machine / 'registry.db'}"


def test_a_local_registry_is_copied_home(home: Path, tmp_path: Path):
    vm = tmp_path / "vm"  # the VM's own disk, while `home` plays Google Drive
    assert main(["run", str(HELLO), on_disk_of(vm)]) == 0
    assert (home / "registry.db").read_bytes() == (vm / "registry.db").read_bytes()
    copy = Registry.sqlite(home / "registry.db")
    (run,) = copy.list_runs()
    assert run.status == "done"
    assert len(copy.list_evals(run.run_id)) == 4  # the final grade on each eval set


def test_a_new_machine_carries_on_from_the_copy_at_home(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    main(["run", str(HELLO), on_disk_of(tmp_path / "vm1")])
    assert main(["run", str(HELLO), on_disk_of(tmp_path / "vm2")]) == 0
    assert "already done" in capsys.readouterr().out  # vm2 began from vm1's copy at home


# --- added in m1s23: speed and time left ------------------------------------------------------


def test_each_grading_reports_the_speed_and_the_time_left(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    main(["run", str(HELLO), "eval.every=5"])
    output = capsys.readouterr().out
    gradings = [line for line in output.splitlines() if line.startswith("step ")]
    assert [line.split(":")[0] for line in gradings] == ["step 5", "step 10", "step 15", "step 16"]
    speed = re.compile(r"\([\d,]+ tokens/s, about \d+(\.\d)? (s|min|h) to go\)$")
    assert all(speed.search(line) for line in gradings), gradings
    assert gradings[-1].endswith("about 0 s to go)")  # the budget is spent
    assert re.search(r"done: ran 16 steps in \d+ s \(", output)


# --- added in m2s22: RL runs ----------------------------------------------------------------

# hello and rl.yaml made small enough for a test: 2-digit problems only, a 64-token context, 4
# problems x 2 answers an RL step, a budget of about 2 steps, graded at every step on the
# first 5 problems of arithmetic-2.
SMALL = ["model.context=64", "task.difficulties=[2]"]
SMALL_RL = [
    *SMALL,
    "rl.prompts_per_step=4",
    "rl.group_size=2",
    "rl.flops=1e10",
    "eval.every=1",
    "eval.problems=5",
]


def warm_started(home: Path) -> str:
    """Run hello, made small, and return its run_id: a finished warm start to begin RL from."""
    assert main(["run", str(HELLO), *SMALL]) == 0
    (run,) = Registry.sqlite(home / "registry.db").list_runs()
    return run.run_id


def test_an_rl_run_starts_from_a_warm_start_and_records_every_grading(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    assert main(["run", str(RL), *SMALL_RL, f"rl.from_run={warm_start}"]) == 0
    output = capsys.readouterr().out
    assert f"warm start:    {warm_start}" in output
    assert "step 0 (the warm start): arithmetic-2 " in output
    registry = Registry.sqlite(home / "registry.db")
    (rl,) = [run for run in registry.list_runs() if run.run_id != warm_start]
    assert rl.status == "done"
    steps = int(re.findall(r"done: ran (\d+) steps", output)[-1])
    graded = registry.list_evals(rl.run_id)
    assert sorted({e.step for e in graded}) == list(range(steps + 1))  # step 0, then every step
    metrics = {"greedy_pass@1", "sampled_pass@1", "sampled_pass@4"}
    for step in range(steps + 1):
        at_step = [e for e in graded if e.step == step and e.task == "arithmetic-2"]
        sampled = step in (0, steps)  # m4s3: sampled answers at the first and last gradings
        assert {e.metric for e in at_step} == (metrics if sampled else {"greedy_pass@1"})
        assert all(e.n == 5 for e in at_step)


def test_rl_from_a_run_the_registry_lacks_is_refused(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    missing = "rl.from_run=0123456789abcdef-s0-01234567"
    assert main(["run", str(RL), *SMALL_RL, missing]) == 2
    error = "rl.from_run: the registry has no run 0123456789abcdef-s0-01234567"
    assert error in capsys.readouterr().err
    assert Registry.sqlite(home / "registry.db").list_runs() == []  # nothing recorded


def test_rl_from_a_warm_start_of_another_model_is_refused(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    other = ["model=ladder_s/m2", f"rl.from_run={warm_start}"]
    assert main(["run", str(RL), *SMALL_RL, *other]) == 2
    error = f"rl.from_run {warm_start} trained model m1, not this experiment's model m2"
    assert error in capsys.readouterr().err
    assert len(Registry.sqlite(home / "registry.db").list_runs()) == 1  # only the warm start


# --- added in m2s28: telemetry --------------------------------------------------------------


def test_an_rl_run_records_what_the_collectors_measure_at_every_step(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    assert main(["run", str(RL), *SMALL_RL, f"rl.from_run={warm_start}"]) == 0
    output = capsys.readouterr().out
    (rl,) = [
        run for run in Registry.sqlite(home / "registry.db").list_runs() if run.run_id != warm_start
    ]
    assert f"metrics:       {home / 'metrics' / rl.run_id}" in output
    steps = int(re.findall(r"done: ran (\d+) steps", output)[-1])
    rows = read_metrics(home, rl.run_id)
    assert sorted({row.step for row in rows}) == list(range(1, steps + 1))
    for step in range(1, steps + 1):
        names = {row.metric for row in rows if row.step == step}
        # 6 rollout/, 9 policy/ (kl_ref: GRPO), 3 optim/, 11 system/, 8 numerics/, 4 mismatch/
        assert len(names) == 41
        assert {"rollout/reward_mean", "policy/kl_ref", "optim/loss", "system/step_s"} <= names


# --- added in m2s36: the telemetry off switch, and what telemetry costs ---------------------


def test_with_telemetry_off_an_rl_run_records_no_metrics(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    off = ["runtime.telemetry=false", f"rl.from_run={warm_start}"]
    assert main(["run", str(RL), *SMALL_RL, *off]) == 0
    assert "metrics:       off" in capsys.readouterr().out
    assert [folder.name for folder in (home / "metrics").iterdir()] == [warm_start]  # m3s11


def test_overhead_times_the_steps_without_and_with_telemetry_and_records_nothing(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    before = sorted(home.rglob("*"))
    registry = (home / "registry.db").read_bytes()
    timed = ["--steps", "2", "--warmup", "1"]
    code = main(["overhead", str(RL), *SMALL_RL, f"rl.from_run={warm_start}", *timed])
    output = capsys.readouterr().out
    assert f"warm start:    {warm_start}" in output
    assert "steps:         1 to warm up, then 2 timed; without, then with telemetry" in output
    assert re.search(r"without: +\d+\.\d{3} s a step", output)
    assert re.search(r"with: +\d+\.\d{3} s a step \(\d+ detector events\)", output)
    assert re.search(r"overhead: +[-+]\d+\.\d% of step time, (within|OVER) the 5% budget", output)
    assert code == (0 if "within the 5% budget" in output else 1)
    assert sorted(home.rglob("*")) == before  # no metrics, no checkpoints, no runs
    assert (home / "registry.db").read_bytes() == registry


# --- added in m2s38: gradings kept problem by problem ---------------------------------------


def test_an_rl_run_keeps_every_grading_problem_by_problem(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    assert main(["run", str(RL), *SMALL_RL, f"rl.from_run={warm_start}"]) == 0
    output = capsys.readouterr().out
    registry = Registry.sqlite(home / "registry.db")
    (rl,) = [run for run in registry.list_runs() if run.run_id != warm_start]
    assert f"gradings:      {home / 'gradings' / rl.run_id}" in output
    steps = int(re.findall(r"done: ran (\d+) steps", output)[-1])
    gradings = load_gradings(home, rl.run_id)
    assert [grading.step for grading in gradings] == list(range(steps + 1))
    graded = registry.list_evals(rl.run_id)
    for grading in gradings:  # each row of the evals table sums up one result in the files
        rows = {(e.task, e.metric): e.value for e in graded if e.step == grading.step}
        kept = {(result.name, "greedy_pass@1"): result.pass_rate for result in grading.greedy}
        for result in grading.sampled:
            kept[(result.name, "sampled_pass@1")] = result.pass_at(1)
            kept[(result.name, "sampled_pass@4")] = result.pass_at(4)
        assert kept == rows
        (whole,) = [result for result in grading.greedy if result.name == "arithmetic-2"]
        assert whole.n == 5


def test_report_gains_pairs_the_seeds_of_each_rl_experiment(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    assert main(["report", "gains"]) == 2  # only a warm start so far
    assert "no finished RL run kept its gradings" in capsys.readouterr().err
    for seed in (0, 1):
        assert main(["run", str(RL), *SMALL_RL, f"rl.from_run={warm_start}", f"seed={seed}"]) == 0
    registry = Registry.sqlite(home / "registry.db")
    runs = [run for run in registry.list_runs() if run.run_id != warm_start]
    gradings = [load_gradings(home, run.run_id) for run in runs]
    capsys.readouterr()
    assert main(["report", "gains"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"m1 grpo: experiment {runs[0].config_hash} at {runs[0].git_sha[:8]}" + (
        ", uncommitted changes" if runs[0].git_dirty else ""
    )
    steps = f"{gradings[0][-1].step}, {gradings[1][-1].step}"
    assert lines[1] == f"  seeds 0, 1; last steps {steps}"
    rows = [line.split() for line in lines[3:-1]]  # name, n, first, last, ..., verdict
    expected = gains(gradings)
    assert [(row[0], int(row[1]), row[-1]) for row in rows] == [
        (gain.name, gain.n, gain.verdict) for gain in expected
    ]
    assert lines[-1] == ""  # and nothing left out

    shutil.rmtree(home / "gradings" / runs[1].run_id)  # as if seed 1 ran before m2s38
    assert main(["report", "gains"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == f"  seeds 0; last steps {gradings[0][-1].step}"
    assert lines[-1] == "left out 1 of the RL runs: unfinished, or from before m2s38 (no gradings)"


# --- added in m3s10: the cost model's steps against the steps RL runs timed -----------------


def test_cost_validate_checks_each_run_that_timed_its_steps(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    rl = [*SMALL_RL, f"rl.from_run={warm_start}"]
    assert main(["run", str(RL), *rl]) == 0
    assert main(["run", str(RL), *rl, "seed=1", "runtime.telemetry=false"]) == 0
    capsys.readouterr()
    assert main(["cost", "validate", "--hardware", "t4"]) == 2
    assert "no run on t4 timed its steps" in capsys.readouterr().err
    assert main(["cost", "validate", "--hardware", "cpu"]) == 0
    out = capsys.readouterr().out
    (timed,) = [
        run.run_id
        for run in Registry.sqlite(home / "registry.db").list_runs()
        if run.run_id != warm_start and metrics_folder(home, run.run_id).is_dir()
    ]
    steps, measured = measured_step(read_metrics(home, timed))
    assert steps >= 1
    check = check_step(timed, predict_rl_run(load_config(RL, rl), CPU).step, steps, measured)
    saved = Validation.model_validate_json((home / "bench" / "cpu" / "validation.json").read_text())
    assert {each.run_id for each in saved.checks} == {warm_start, timed}  # m3s26: report cost
    assert "calibration:   none, so the datasheet's peaks" in out
    assert "runs:          2 runs timed their steps (left out 1 with no timed steps)\n" in out
    errors = "".join(f"{check.errors[part]:>+10.1%}" for part in PARTS)
    assert f"{timed:<30}{'fp32':<10}{steps:6d}{errors}  {check.predicted['step']:.3f} s vs " in out
    # m3s11: the warm start timed its steps too, and is checked on train and step alone
    steps, measured = measured_step(read_metrics(home, warm_start))
    warm = check_step(warm_start, predict_step(load_config(HELLO, SMALL), CPU), steps, measured)
    errors = " " * 20 + "".join(f"{warm.errors[part]:>+10.1%}" for part in ("train", "step"))
    assert (
        f"{warm_start:<30}{'fp32':<10}{steps:6d}{errors}  {warm.predicted['step']:.3f} s vs " in out
    )
    mape = f"{(abs(check.errors['step']) + abs(warm.errors['step'])) / 2:.1%}"
    assert f"verdict:       step MAPE {mape}, " in out
    validation = Validation.model_validate_json(
        (home / "bench" / "cpu" / "validation.json").read_text()
    )
    assert validation.hardware == "cpu" and "step" in validation.errors  # m3s15: for cost compare
    assert f"saved:         {home / 'bench' / 'cpu' / 'validation.json'}\n" in out


# --- added in m3s17: the padding cut, measured ----------------------------------------------


def test_cost_measure_predicts_the_cut_then_times_it_and_records_no_run(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    before = sorted(home.rglob("*"))
    registry = (home / "registry.db").read_bytes()
    timed = ["--hardware", "cpu", "--steps", "2", "--warmup", "1"]
    code = main(["cost", "measure", str(RL), *SMALL_RL, f"rl.from_run={warm_start}", *timed])
    output = capsys.readouterr().out
    assert code == 0
    (prediction,) = (home / "reports").glob("padding-cut-*-cpu.json")
    stem = prediction.with_suffix("")
    assert re.search(rf"predicted: +step [\d.]+ -> [\d.]+ s \([-+][\d.]+%\), {stem}\.md", output)
    assert "steps:         1 to warm up, then 2 timed; without, then with padding-cut" in output
    assert "# Measured: padding-cut, rl on cpu\n" in output
    seconds = r"(\d+\.\d{4} \| ){2}"
    assert re.search(rf"\| step \| {seconds}[-+]\d+\.\d% \| {seconds}[-+]\d+\.\d% \|", output)
    assert "- verdict: unjudged: cpu has no validation (cost validate saves one)" in output
    assert f"saved:         {stem}-measured.md and .json" in output
    names = sorted(path.name for path in (home / "reports").iterdir())
    assert names == [
        f"{stem.name}{end}" for end in ("-measured.json", "-measured.md", ".json", ".md")
    ]
    assert [path for path in sorted(home.rglob("*")) if "reports" not in path.parts] == before
    assert (home / "registry.db").read_bytes() == registry


def test_cost_measure_runs_on_the_device_it_predicts(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    other = "cpu" if torch.cuda.is_available() else "t4"
    assert main(["cost", "measure", str(HELLO), *SMALL, "--hardware", other]) == 2
    expected = f"--hardware {other}, but this machine runs steps on its {device}"
    assert expected in capsys.readouterr().err


# --- m3s21: fp16 -----------------------------------------------------------------------------


SCALER = ("optim/loss_scale", "optim/skipped")  # m3s22


def test_fp16_reaches_warm_starts_and_rl(home: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["run", str(HELLO), *SMALL, "runtime.precision=fp16"]) == 0
    assert "precision:     fp16" in capsys.readouterr().out
    (warm_start,) = Registry.sqlite(home / "registry.db").list_runs()
    fp16 = [f"rl.from_run={warm_start.run_id}", "runtime.precision=fp16"]  # m3s22: RL too
    assert main(["run", str(RL), *SMALL_RL, *fp16]) == 0
    steps = int(re.findall(r"done: ran (\d+) steps", capsys.readouterr().out)[-1])
    runs = Registry.sqlite(home / "registry.db").list_runs()
    (rl,) = [run for run in runs if run.run_id != warm_start.run_id]
    for run in (warm_start, rl):
        latest = latest_checkpoint(home / "checkpoints" / run.run_id)
        assert latest is not None
        assert load_checkpoint(latest, home)[1]["scaler"]["growth_interval"] == 2000  # it had one
    rows = read_metrics(home, rl.run_id)
    scaler = [(row.metric, row.value) for row in rows if row.metric in SCALER]
    assert sorted(set(scaler)) == [("optim/loss_scale", 2.0**16), ("optim/skipped", 0.0)]
    assert len(scaler) == 2 * steps  # both, at every step
    assert all(row.value == 0 for row in rows if row.metric.startswith("numerics/"))


# --- added in m3s23: a run's runtime -------------------------------------------------------------


def test_a_run_records_its_runtime_and_resumes_only_with_it(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["run", str(HELLO), *SMALL, "runtime.padding_cut=true"]) == 0
    registry = Registry.sqlite(home / "registry.db")
    (run,) = registry.list_runs()
    assert registry.get_runtime(run.run_id) == RuntimeConfig(padding_cut=True)
    with registry.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE runs SET status = 'failed'")  # as if it had crashed
    capsys.readouterr()
    assert main(["run", str(HELLO), *SMALL, "runtime.precision=fp16"]) == 2
    expected = (
        "started with runtime.precision=fp32 runtime.padding_cut=True: resume it the same way"
    )
    assert f"run {run.run_id} {expected}" in capsys.readouterr().err


def test_cost_validate_predicts_each_run_as_it_ran(home: Path, capsys: pytest.CaptureFixture[str]):
    fp16 = ["seed=1", "runtime.precision=fp16", "runtime.padding_cut=true"]
    assert main(["run", str(HELLO), *SMALL]) == 0
    assert main(["run", str(HELLO), *SMALL, *fp16]) == 0
    capsys.readouterr()
    assert main(["cost", "validate", "--hardware", "cpu"]) == 0
    out = capsys.readouterr().out
    config = load_config(HELLO, SMALL)
    registry = Registry.sqlite(home / "registry.db")
    for run in registry.list_runs():
        runtime = registry.get_runtime(run.run_id)
        assert runtime is not None
        label = "fp16 cut" if run.seed == 1 else "fp32"
        steps, measured = measured_step(read_metrics(home, run.run_id))
        check = check_step(run.run_id, predict_run_step(config, runtime, CPU), steps, measured)
        errors = " " * 20 + "".join(f"{check.errors[part]:>+10.1%}" for part in ("train", "step"))
        assert f"{run.run_id:<30}{label:<10}{steps:6d}{errors}  " in out
    validation = Validation.model_validate_json(
        (home / "bench" / "cpu" / "validation.json").read_text()
    )
    assert validation.errors["step, fp16 cut"].samples == 1  # each runtime on its own too
    assert validation.errors["step, fp32"].samples == 1
    fitted = Efficiency(compute=0.5, memory=0.5, launch_s=1e-5)
    fp32 = {name: fitted for name in get_args(Group) if not name.endswith(".fp16")}
    path = home / "bench" / "cpu" / "calibration.json"  # fitted before any fp16 bench
    save_calibration(path, Calibration(hardware="cpu", versions={}, efficiencies=fp32, errors={}))
    assert main(["cost", "validate", "--hardware", "cpu"]) == 0
    expected = "runs:          1 runs timed their steps (left out 1 in fp16, with no fp16 fit)\n"
    assert expected in capsys.readouterr().out


# --- added in m3s24: fp16, measured --------------------------------------------------------------


def test_cost_measure_times_fp16_beside_fp32(home: Path, capsys: pytest.CaptureFixture[str]):
    timed = ["--hardware", "cpu", "--change", "fp16", "--steps", "2", "--warmup", "1"]
    assert main(["cost", "measure", str(HELLO), *SMALL, *timed]) == 0
    output = capsys.readouterr().out
    assert "steps:         1 to warm up, then 2 timed; without, then with fp16" in output
    assert "# Measured: fp16, hello on cpu\n" in output
    (prediction,) = (home / "reports").glob("fp16-*-cpu.json")
    assert prediction.with_name(f"{prediction.stem}-measured.json").is_file()


# --- added in m3s25: the throughput detector ------------------------------------------------


def test_an_rl_run_is_watched_against_its_predicted_step_once_validated(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    rl = [*SMALL_RL, f"rl.from_run={warm_start}"]
    assert main(["run", str(RL), *rl]) == 0
    out = capsys.readouterr().out
    assert "throughput:    not watched (run cost validate --hardware cpu first)\n" in out
    # like the Mac's healthy runs (m3s10)
    step = Errors(samples=14, mape=0.3, bias=-0.3, worst=-0.86, median=-0.28, spread=0.05)
    validation = Validation(hardware="cpu", errors={"step": step})
    (home / "bench" / "cpu").mkdir(parents=True)
    (home / "bench" / "cpu" / "validation.json").write_text(validation.model_dump_json())
    assert main(["run", str(RL), *rl, "seed=1"]) == 0
    predicted = predict_run_step(load_config(RL, rl), RuntimeConfig(), CPU).seconds
    limit = regression_limit(predicted, -0.28, 0.05)
    line = (
        f"throughput:    {predicted:.3f} s a step predicted; {REGRESSION_STEPS} in a row above "
        f"{limit:.3f} s are a regression (step: median -28.0%, spread 5.0%)\n"
    )
    assert line in capsys.readouterr().out


# --- added in m4s8: each RL run's pass rate against its compute, and the curve through it ---


def test_fit_prints_each_grading_and_the_curve_through_them(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    assert main(["fit", warm_start]) == 2
    assert "is a warm start: only RL runs have a curve" in capsys.readouterr().err
    rl = [*SMALL_RL, "rl.flops=3e10", f"rl.from_run={warm_start}"]  # 3 steps or more
    assert main(["run", str(RL), *rl]) == 0
    registry = Registry.sqlite(home / "registry.db")
    (run_id,) = [run.run_id for run in registry.list_runs() if run.run_id != warm_start]
    capsys.readouterr()
    assert main(["fit", run_id]) == 0
    lines = capsys.readouterr().out.splitlines()
    points = run_points(load_gradings(home, run_id), read_metrics(home, run_id), ALL_OPS)
    assert points[-1].compute >= 3e10 > points[-2].compute  # C is what rl.flops spends
    curve = fit_run(points).curve
    assert lines[0] == f"run:           {run_id}"
    pooled = f"pooled ({points[0].n} of them)"
    assert lines[1] == f"R:             greedy pass@1 on its +-* problems, {pooled}"
    table = [line.split() for line in lines[3 : 3 + len(points)]]
    assert [int(row[0]) for row in table] == [point.step for point in points]
    assert [float(row[1]) for row in table] == pytest.approx([p.compute for p in points], rel=1e-2)
    assert lines[3 + len(points)].startswith(f"fit:           A {curve.a:.1%}, C_mid ")
    (near,) = [line for line in lines if line.startswith("near A:")]
    step = step_reaching(cumulative_compute(read_metrics(home, run_id)), curve.compute_at(NEAR))
    assert near.endswith(f"(step {step})" if step else f"(not reached by step {points[-1].step})")

    shutil.rmtree(metrics_folder(home, run_id))  # as if it ran with runtime.telemetry=false
    assert main(["fit", run_id]) == 2
    assert f"{run_id}: the metrics stop before step 1, a graded one" in capsys.readouterr().err


# --- added in m4s10: the seed study's spread, and the power it leaves a comparison ---


def test_power_prints_each_seeds_final_r_and_the_power_their_spread_gives(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    warm_start = warm_started(home)
    assert main(["power", warm_start, warm_start]) == 2
    assert "a run is given twice" in capsys.readouterr().err
    rl = [*SMALL_RL, "rl.flops=3e10", f"rl.from_run={warm_start}"]
    assert main(["run", str(RL), *rl]) == 0
    registry = Registry.sqlite(home / "registry.db")
    (run_id,) = [run.run_id for run in registry.list_runs() if run.run_id != warm_start]
    capsys.readouterr()
    assert main(["power", run_id]) == 2
    assert "has one run: a spread needs two seeds or more" in capsys.readouterr().err
    assert main(["run", str(RL), *rl, "seed=1"]) == 0
    other = run_id.replace("-s0-", "-s1-")
    capsys.readouterr()
    assert main(["power", run_id, other]) == 0
    lines = capsys.readouterr().out.splitlines()
    finals = [pooled_rate(load_gradings(home, run)[-1], ALL_OPS)[0] for run in (run_id, other)]
    sd = statistics.stdev(finals)
    assert lines[0] == f"config:        {run_id.split('-')[0]}, 2 runs"
    assert [line.split()[0] for line in lines[2:4]] == [run_id, other]
    assert lines[4].startswith(f"final R:       mean {statistics.fmean(finals):.1%}, sd ")
    power = f"{detection_power(EFFECT, sd, 5):.0%} to see 3 pp with 5 seeds a side"
    assert lines[5] == f"power:         {power} (two-sided, alpha 0.05)"
    assert lines[6] == f"seeds:         {seeds_for_power(EFFECT, sd)} a side for 80%"
    assert main(["power", run_id, other, "--seeds", "1"]) == 2
