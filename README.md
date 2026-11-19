[![CI](https://github.com/Argo-Navis-Lab/scalelab/actions/workflows/ci.yml/badge.svg)](https://github.com/Argo-Navis-Lab/scalelab/actions/workflows/ci.yml)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/argo-navis-research-laboratory/scalelab-runs)

# ScaleLab

**ScaleLab is a measurement lab for reinforcement learning (RL) on small language models.**
It asks one question: when you train a model with RL, how much better does it get as you
spend more compute, and how does that change with model size? It runs on a laptop and one
rented T4 GPU, and it measures carefully: every run is priced before it starts, watched while
it runs, and compared with confidence intervals after.

It is not an RL training library like TRL or verl. Its models are small (0.8M to 59M
non-embedding parameters, trained from scratch) and its task is one (multi-digit arithmetic),
so its results are about method at small scale, not claims about frontier models.

## What it found

- [Write-up 1: the cost model, and what it says about RL's bottlenecks](docs/writeups/01-cost-model.md).
  A calibrated FLOP and memory model predicts held-out RL steps to 4.1% on a T4 (13.2% on an
  A100). At this scale RL's bottleneck is generation's kernel launches, not arithmetic.
- [Write-up 2: how RL gains scale with model size, and how far one fit predicts](docs/writeups/02-scaling.md).
  One family of sigmoid curves fits 50 runs (five sizes, five seeds, GRPO and DAPO) to about
  1.5 pp. Fitted on the three smallest sizes it predicts the fourth to 1.5 pp; fitted on four
  it over-predicts the fifth by about 3 pp. DAPO beats GRPO at equal FLOPs, not at equal samples.

## The data

Every finished run behind Write-up 2 (90 runs: 85 RL runs and their 5 warm starts) is a public
dataset, [scalelab-runs](https://huggingface.co/datasets/argo-navis-research-laboratory/scalelab-runs)
(CC BY 4.0, 312 MB): the run registry, every grading, and every per-step metric. The download
is a ScaleLab home, so the commands below read it directly:

    uvx --from huggingface_hub hf download argo-navis-research-laboratory/scalelab-runs --repo-type dataset --local-dir scalelab-runs
    SCALELAB_HOME=scalelab-runs uv run scalelab report gains

## Try it

Python 3.12 and [uv](https://docs.astral.sh/uv/). On a Mac or Linux CPU:

    git clone https://github.com/Argo-Navis-Lab/scalelab && cd scalelab
    uv sync
    uv run scalelab run configs/experiments/hello.yaml

`hello` is a 16-step warm start of the smallest model (about 10 s). Runs and their records go
to `runs/` (or `SCALELAB_HOME`). `uv run scalelab --help` lists every command.

## What is inside

| Part | What it does | Commands |
| --- | --- | --- |
| Run records | Config hashing, run ids tied to the commit, an SQLite registry, checkpoints and resume | `run`, `grade` |
| Models and task | A small transformer ladder (m1 to m5), arithmetic with verifiers, frozen eval sets | `bands` |
| RL | GRPO, Dr. GRPO, RLOO and DAPO on one loop, on a FLOP budget | `run` |
| Telemetry | Per-step metrics, per-layer statistics, numerics checks, anomaly detectors (spikes, divergence, collapse) | `overhead` |
| Cost model | FLOPs, memory and time per op, benchmarked and calibrated per GPU | `bench`, `cost` |
| Sweeps | Planned sweeps priced in GPU hours, a job queue and a worker | `sweep`, `worker` |
| Analysis | Bootstrap intervals over seeds and problems, power, scaling fits, backtests, compute-matched comparisons | `report`, `fit`, `joint`, `backtest`, `versus`, `power` |
| Export | Runs as a dataset with machine details left out | `export` |

## Checks

    uv run ruff check .
    uv run ruff format --check .
    uv run pyright
    uv run pytest

CI runs the same four on every push.

## Citing

To cite ScaleLab, use [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository"
button reads it). What changed in each version: [CHANGELOG.md](CHANGELOG.md).

## License

Code: Apache 2.0 ([LICENSE](LICENSE)). Dataset: CC BY 4.0.

---

Maintained by **SOHEIL SANATI MOUTABAN** · [Argo Navis Lab](https://github.com/Argo-Navis-Lab)