# Changelog

Every release of ScaleLab, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Write-up 2 now calls its DAPO what it is, a DAPO-style variant that drops flat groups without
  refilling the batch; says a dropped group's generation is billed; reports the m4 backtest's
  predicted values (it misses at the last grading too); reads equal samples as "no difference
  detected"; states how the power estimate was made; and its figures name the algorithms.
- The README and the walkthrough say the same.

- Write-up 2 gives sizes as non-embedding parameters (0.80M to 59.5M), heads its backtest by
  what it found (close at m4, over-predicts m5), computes power over all 252 seed splits and
  calls it approximate, and clarifies the 100M-token warm starts' numbers.


## [1.0.0] - 2026-11-19

The first release: the measurement lab behind Write-ups 1 and 2 and the
[scalelab-runs](https://huggingface.co/datasets/argo-navis-research-laboratory/scalelab-runs) dataset.

### Added

- Run records: composed YAML configs with a stable hash, run ids tied to the config, the seed
  and the commit, an SQLite registry, content-addressed checkpoints, resume after a stop or a
  crash (`scalelab run`, `scalelab grade`).
- Models and task: a small transformer ladder (m1 to m5, RMSNorm, RoPE, SwiGLU, a KV cache),
  multi-digit arithmetic and Countdown with verifiers, eval sets frozen by hash, the bands an
  RL task should use (`scalelab bands`).
- RL: GRPO, Dr. GRPO, RLOO and DAPO on one loop, run on a FLOP budget so algorithms compare
  at equal compute; greedy and sampled gradings on a geometric schedule.
- Telemetry: per-step and per-layer metrics in Parquet, numerics and sampler-mismatch checks,
  detectors for spikes, divergence and collapse that warn, snapshot or halt
  (`scalelab overhead` measures what it costs).
- Cost model: exact parameter, FLOP and memory counts, GPU benchmarks, a calibration per GPU
  and precision, predictions checked against measured steps (`scalelab bench`, `scalelab cost`).
- Sweeps: plans priced in GPU hours, a job queue in the registry, a worker that leases jobs
  and fits them to a session (`scalelab sweep`, `scalelab worker`).
- Analysis: bootstrap intervals over seeds and problems, power, per-run and joint scaling fits,
  backtests, compute-matched comparisons, figures (`scalelab report`, `fit`, `joint`,
  `backtest`, `versus`, `power`).
- Export: runs as a dataset with a card and machine details left out (`scalelab export`).
- Write-up 1 (the cost model) and Write-up 2 (how RL gains scale with model size).

[1.0.0]: https://github.com/Argo-Navis-Lab/scalelab/releases/tag/v1.0.0
