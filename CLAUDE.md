# ScaleLab: notes for AI coding sessions

ScaleLab is a small-scale RL scaling lab built from a private blueprint PDF, "Small-Scale RL
Scaling Lab — Architecture & Implementation Blueprint" (the user gives its path). We follow its
milestones M0 → M7. Package `scalelab`, src layout, uv, Python 3.12, PyTorch only.

## How we work (default mode)

- The maintainer is learning as they build. Act as a step-by-step tutor: one small step at a
  time, the what and why, full file contents, one copy-pasteable command per `bash` block, code
  explained piece by piece, each step tied to the blueprint section it implements.
- Don't ask questions. Make sensible decisions, state them, continue.
- The user types every change themselves. Don't edit their repo. Before writing a lesson, build
  and run the whole step on a scratch copy (`rsync` the repo without `.venv`, `__pycache__`,
  `.pytest_cache`, `runs`) so every command and output in the lesson is real.
- At the start of each step, check the user's repo: `git status`, `git log`, the four checks
  below, the latest CI run (`gh run list --limit 3`), and diff their files against what was
  taught. Report anything that doesn't match (missing files have happened).
- Lesson format: what and why (with blueprint links) → decisions → sub-steps → "Try it" → run
  the checks → "Prove the tests can fail" (one deliberate sabotage, its exact failure, then the
  revert) → save (`git add .`, `git status --short`, `git commit -m`, `git push`) → "✅ How to
  know it worked" with exact expected outputs: test counts, `git status --short` lines, demo
  output.
- End each step by updating the Progress section of this file.

## The four checks (run all of them before every push; CI runs the same)

    uv run ruff check .
    uv run ruff format --check .
    uv run pyright
    uv run pytest

## Progress

- [x] M0, skeleton and core (finished 2026-10-18, 201 tests): config + composition, hashing,
  pytest, envcapture, SQLite registry, storage, determinism, artifacts + checkpoints + resume,
  `scalelab run` CLI, CI. Exit criterion: `scalelab run configs/experiments/hello.yaml` records
  a complete run row; CPU resume and determinism tests run in CI on every push.
- [ ] M1, models, tasks and cost model v0 (blueprint Section 12): the Track S transformer with
  MHA and the attention-variant interface, KV-cache decode, the arithmetic and Countdown tasks
  and verifiers, warm-start training, and subsystem A's analytic FLOP and memory counts.
  Exit: m1–m4 warm-start to useful pass rates; A's parameter and FLOP counts pass golden tests.

## Decisions already made (don't relitigate)

- Track S (scratch models) is primary. Track P (pretrained Qwen) is the M7 reality check.
- PyTorch only; no TPUs. Postponed: Postgres, parallel workers, vLLM, Triton, `fsspec` (until
  the first non-local backend), pre-commit hooks.
- Run identity: `run_id = f"{config_hash}-s{seed}-{git_sha[:8]}"`. The seed is not part of
  `ExperimentConfig`. `RuntimeConfig` and `EnvInfo` are recorded, never hashed.
- Config composition: `defaults:` picks one file per group (`GROUP_FOLDERS` in config.py, now
  `model` and `task`), the experiment file goes on top section by section, CLI overrides last.
  CLI words: `seed=N`, `runtime.<field>=…`, `group=choice` swaps a file, anything else sets a
  value. Values are parsed as YAML.
- `scalelab run`: done → "already done"; running or failed → resume from
  `<home>/checkpoints/<run_id>`; recorded with different uncommitted changes → refuse.
  Exit codes 0 ok, 2 bad input (nothing recorded), 130 stopped early (checkpoint saved),
  1 crash (row marked failed). The job is still `core/dummy_job.py` (50 steps).
- The registry stores and never decides. `SCALELAB_HOME` beats `RuntimeConfig.output_dir`
  (default `runs`). Registry at `<home>/registry.db`. On Colab the live SQLite file stays on
  the VM's local disk and is copied to Drive atomically; SQLite over the Drive mount is
  unreliable.
- Checkpoints: one safetensors file of all tensors in the content-addressed store
  (`<home>/artifacts/<sha[:2]>/<sha>`) plus a JSON manifest (`step-%09d.json`). Policy: save
  every N steps or every 15 minutes, whichever comes first, keep the last 2, Ctrl+C and SIGTERM
  mean "save and stop". Saving never consumes random numbers.
- Seed streams `STREAMS = ("init", "data", "sampling")` from `SeedSequence.spawn`.
  Deterministic mode on for CPU, off for GPU, recorded per run (`runs.deterministic`).
- Quality: ruff (`line-length = 100`, rules E W F I UP B), pyright `strict` on all of `src`
  except three rules switched off (`reportUnknownMemberType`, `reportUnknownArgumentType`,
  `reportUnknownVariableType`) because they mostly flag gaps in PyTorch/NumPy typing.
- CI installs the same `uv.lock` as every other machine, including PyTorch's CUDA wheels on
  Linux (a few GB, slower installs) — one lockfile everywhere beats a faster CI. Revisit only
  if CI time becomes a real problem.

## Invariants: changing these breaks past runs, so only change them deliberately

- `canonical_json` settings, `config_hash = sha256[:16]`, the `run_id` format.
- Adding a field with a default to `ExperimentConfig` changes every config's hash. hello is
  pinned at `3f7ffa68d30146bd` by `test_hello_hash_is_pinned`: update it on purpose in M1 and
  say so.
- `STREAMS` order (only append). `test_child_seeds_are_pinned` holds golden values.
- Checkpoint `FORMAT_VERSION = 1`. Write tensors before the manifest; delete manifests before
  tensors; only prune folders inside `SCALELAB_HOME`; a tensor file is deleted only when no
  manifest under home still uses it.
- All Pydantic models inherit `StrictModel` (`extra="forbid"`, `frozen=True`). Plain files only
  through `storage.atomic_write_*`; SQL only through SQLAlchemy Core in `registry.py`.
- `metadata.create_all` never alters a table. New columns make old registry files raise
  `OutdatedRegistryError`; delete demo registries or migrate real ones.

## Code conventions

- Module docstrings don't cite the blueprint (keep section references in lesson text).
- Tests: `tests/unit/` and `tests/integration/` (`--import-mode=importlib`, `strict = true`).
  Shared fixtures `raw` and `config` (hello values) in `tests/conftest.py`. Use `tmp_path`,
  `monkeypatch`, `capsys`; subprocess tests for real kills and signals.
- `__main__` demos remain in envcapture, determinism and dummy_job. `scalelab.cli:main` is the
  `scalelab` command; `src/scalelab/__init__.py` stays import-light (no torch).

## Known open items for M1 and later

- `ExperimentConfig` will gain algo, rollout, budget and hardware sections (Section 4.1); add
  `algo` and `hardware` to `GROUP_FOLDERS` with their config folders.
- Replace the dummy job with the real trainer; the CLI's `DUMMY_STEPS` becomes a budget.
- The registry's `artifacts`, `evals`, `jobs` and `sweeps` tables (Section 4.2) don't exist yet.
- `env_json` records only the first session of a resumed run.
- Before the first Colab GPU run: a notebook bootstrap that installs at a pinned SHA, sets
  `SCALELAB_HOME` on Drive, keeps the registry on local disk and copies it to Drive.
