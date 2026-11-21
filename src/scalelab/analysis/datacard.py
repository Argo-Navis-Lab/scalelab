"""What an export holds, for the people who download it: runs.csv and runs.parquet, one line
a run, and the README.md the Hugging Face Hub shows as the dataset's page (its datasheet)."""

import csv
import io
import json
from collections import Counter
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scalelab.core.registry import Registry
from scalelab.core.storage import atomic_write_bytes, atomic_write_text
from scalelab.rl.gradings import gradings_folder

RUNS_CSV = "runs.csv"
# m4s20: the Hub's viewer reads every config with ONE builder, picked from the default config's
# files: a CSV default made it read metrics/*.parquet as CSV (UnicodeDecodeError). Every config
# the card names is Parquet; runs.csv stays for spreadsheets.
RUNS_PARQUET = "runs.parquet"
CARD = "README.md"
CODE = "https://github.com/Argo-Navis-Lab/scalelab"


@dataclass(frozen=True)
class RunRow:
    run_id: str
    kind: str  # "warmstart" or "rl"
    model: str  # "m1" .. "m5"
    algo: str  # "grpo", "rloo", "dapo"; empty for a warm start
    seed: int
    from_run: str  # the warm start an RL run starts from; empty for a warm start
    git_sha: str
    hardware: str
    last_step: int  # the last step graded
    gradings: int  # grading files (RL runs only)


def run_rows(home: Path, registry: Registry) -> list[RunRow]:
    """One RunRow for each run in `registry`, sorted by kind, model, algo, seed."""
    rows: list[RunRow] = []
    for run in registry.list_runs():
        config_json = registry.experiment_json(run.config_hash)
        assert config_json is not None  # every run row points at an experiment row
        config = json.loads(config_json)
        rl = config.get("rl") or {}
        rows.append(
            RunRow(
                run_id=run.run_id,
                kind="rl" if rl else "warmstart",
                model=config["model"]["name"],
                algo=(config.get("algo") or {}).get("name", ""),
                seed=run.seed,
                from_run=rl.get("from_run", ""),
                git_sha=run.git_sha,
                hardware=run.hardware,
                last_step=max((e.step for e in registry.list_evals(run.run_id)), default=0),
                gradings=len(list(gradings_folder(home, run.run_id).glob("grading-*.json"))),
            )
        )
    return sorted(rows, key=lambda r: (r.kind != "warmstart", r.model, r.algo, r.seed))


def runs_csv(rows: list[RunRow]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(field.name for field in fields(RunRow))
    writer.writerows(astuple(row) for row in rows)
    return out.getvalue()


def runs_parquet(rows: list[RunRow]) -> bytes:
    """The same table as runs_csv, as Parquet bytes (seed, last_step, gradings as int64)."""
    columns = {field.name: [getattr(row, field.name) for row in rows] for field in fields(RunRow)}
    types = {
        field.name: pa.int64() if field.type is int else pa.string() for field in fields(RunRow)
    }
    table = pa.table({name: pa.array(values, type=types[name]) for name, values in columns.items()})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def counts_table(rows: list[RunRow]) -> str:
    """Runs by kind and algorithm (lines) and model (columns), as a Markdown table."""
    models = sorted({row.model for row in rows})
    counts = Counter((row.algo or "warm start", row.model) for row in rows)
    lines = ["| runs | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for algo in sorted({algo for algo, _ in counts}, key=lambda a: (a != "warm start", a)):
        lines.append(f"| {algo} | " + " | ".join(str(counts[algo, m]) for m in models) + " |")
    return "\n".join(lines)


def dataset_card(rows: list[RunRow]) -> str:
    """The README.md: the Hub's YAML header (licence, tags, the two tables its viewer shows),
    then a datasheet: what is in it, how it was made, what it is for and not for."""
    shas = sorted({row.git_sha[:8] for row in rows})
    return f"""---
license: cc-by-4.0
pretty_name: ScaleLab RL runs
tags:
- reinforcement-learning
- scaling-laws
- grpo
- telemetry
size_categories:
- n<1K
configs:
- config_name: runs
  data_files: {RUNS_PARQUET}
  default: true
- config_name: metrics
  data_files: metrics/*/*.parquet
---

# ScaleLab RL runs

{len(rows)} finished training runs of small transformers (m1-m5) on arithmetic, made with
[ScaleLab]({CODE}): warm starts (supervised pretraining) and the RL runs started from them,
GRPO, RLOO and a DAPO-style variant at equal FLOPs, 5 seeds each (10 for GRPO at m2 and m4).
The algo `dapo` is that variant, not DAPO as published: it drops groups whose answers all
score the same but never refills the batch with more samples. Every run
keeps its config, its gradings (pass@1 on frozen eval sets, problem by problem) and its
per-step telemetry.

{counts_table(rows)}

## What is in it

- `{RUNS_CSV}` and `{RUNS_PARQUET}`: one line a run (kind, model, algo, seed, its warm start,
  commit, last step); the Parquet one is what the viewer shows.
- `registry.db`: ScaleLab's SQLite registry of these runs only: each config exactly as
  hashed, run rows, eval rows, anomaly findings, runtime settings, the sweeps and jobs.
- `gradings/<run_id>/grading-<step>.json`: one grading a file, every problem's outcome.
- `metrics/<run_id>/*.parquet`: telemetry, one row per (step, metric, layer): reward, loss,
  KL, entropy, gradient and update norms, Adam state, activations, numerics, timings.
- `manifest.json`: every file's sha256.

The folder is a ScaleLab home: with `SCALELAB_HOME` set to it, `scalelab fit`, `joint`,
`backtest`, `versus` and `report scaling` read it as they read the runs where they ran.

## How it was made

On one NVIDIA T4 (Colab, then vast.ai), fp16, from commits {", ".join(shas)}. Warm starts
learn from 20M tokens; RL runs train on multiplication at 4 and 5 digits for a fixed FLOPs
budget. The cost model, the study and its limits are ScaleLab's write-ups
([cost model]({CODE}/blob/main/docs/writeups/01-cost-model.md),
[scaling]({CODE}/blob/main/docs/writeups/02-scaling.md)).
`scalelab export` left out what names a machine (local paths, host names) and refused to
write the folder if any remained.

## Uses and limits

For studying RL training dynamics at small scale: seed variance, algorithm comparisons at
equal compute, fitting and backtesting scaling curves, testing run-health detectors on real
telemetry. One task, one learning rate, one warm start a size, models of 0.8M to 59.5M
non-embedding parameters: results here say nothing yet about large language models.
No model weights.

## Licence and citation

Data CC BY 4.0; the code that made it Apache-2.0. Cite the code repository.
"""


def write_card(home: Path, registry: Registry) -> None:
    """runs.csv, runs.parquet and README.md into `home` (an export), from its own registry."""
    rows = run_rows(home, registry)
    atomic_write_text(home / RUNS_CSV, runs_csv(rows))
    atomic_write_bytes(home / RUNS_PARQUET, runs_parquet(rows))
    atomic_write_text(home / CARD, dataset_card(rows))
