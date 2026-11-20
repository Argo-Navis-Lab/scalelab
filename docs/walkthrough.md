# A walkthrough on a laptop

This page runs ScaleLab end to end on a CPU in about two minutes of compute: a first run, a
warm start, a cost prediction, ten RL steps with telemetry, a graded comparison, and then the
same analysis on the 85 real RL runs behind [Write-up 2](writeups/02-scaling.md). Every output
below is real, from an M2 MacBook Air; another CPU may differ in the last digits.

You need Python 3.12 and [uv](https://docs.astral.sh/uv/), on macOS or Linux.

## 1. Install, and choose a home

```bash
git clone https://github.com/Argo-Navis-Lab/scalelab && cd scalelab
uv sync
export SCALELAB_HOME=~/scalelab-walk
```

A home is one folder for everything ScaleLab records: the run registry (`registry.db`, SQLite),
checkpoints, gradings and per-step metrics. Without `SCALELAB_HOME` it is `runs/` in the repo.

## 2. A first run

```bash
uv run scalelab run configs/experiments/hello.yaml
```

```
experiment:    hello (config b38a7209a4d8bd8b), seed 0
run:           b38a7209a4d8bd8b-s0-688766a3
...
step 16: loss 2.955  arithmetic-2 0.0%  arithmetic-3 0.0%  arithmetic-4 0.0%  arithmetic-5 0.0%
done: ran 16 steps in 5 s (20,625 tokens and 2.98e+11 FLOPs in all), last loss 2.954692840576172
```

A run's id is its config's hash, its seed and the commit it ran at, so the same experiment at
the same code always has the same id. Run the command again and it says the run is already
done; stop a run with Ctrl+C and the same command resumes it from its last checkpoint.

## 3. A warm start

RL needs a model that already answers some problems. This one trains the smallest model (m1)
on one million tokens of 2-digit arithmetic:

```bash
uv run scalelab run configs/experiments/warmstart.yaml warm_start.tokens=1000000 'task.difficulties=[2]'
```

```
experiment:    warmstart (config e2d937b08c9be697), seed 0
...
done: ran 402 steps in 30 s (1,000,227 tokens and 6.92e+12 FLOPs in all), last loss 0.05601158365607262
```

Its id ends in your checkout's commit. Keep it, and grade the model on the frozen eval set:

```bash
WS=e2d937b08c9be697-s0-$(git rev-parse --short=8 HEAD)
uv run scalelab grade $WS
```

```
model: m1, step 402, trained on arithmetic at difficulties 2
arithmetic-2    64.5% ± 1.5%  (645 of 1000)
arithmetic-2/*   6.3% ± 1.4%  (20 of 319)
arithmetic-2/+  95.0% ± 1.2%  (320 of 337)
arithmetic-2/-  88.7% ± 1.7%  (305 of 344)
```

Addition and subtraction are nearly solved; multiplication (`*`) is the frontier, so the RL
below trains on `*` only.

## 4. Price the RL run before it runs

```bash
uv run scalelab cost predict --hardware cpu configs/experiments/rl.yaml rl.from_run=$WS 'task.ops=*' 'task.difficulties=[2]' rl.max_steps=10 eval.every=5
```

```
calibration:   none, so the datasheet's peaks and free launches (a lower bound)
phase           seconds  share  bound    slowest op class
generate          0.069    27%  memory   attention 0.032 s
logprobs          0.076    30%  compute  gemm 0.045 s
train             0.108    43%  compute  gemm 0.067 s
step              0.253         2.78e+11 FLOPs
run:           10 steps, 3 s (rl.max_steps: the FLOPs budget buys 3593)
```

Uncalibrated, the cost model is only a lower bound: the run below takes 33 s, not 3. On a GPU,
`scalelab bench` and `scalelab cost calibrate` fit it to the machine; on a T4 it then predicts
held-out RL steps to 4.1% ([Write-up 1](writeups/01-cost-model.md)).

## 5. Ten steps of RL

```bash
uv run scalelab run configs/experiments/rl.yaml rl.from_run=$WS 'task.ops=*' 'task.difficulties=[2]' rl.max_steps=10 eval.every=5
```

```
experiment:    rl (config deb1f96192f77694), seed 0
...
warm start:    e2d937b08c9be697-s0-688766a3
step 0 (the warm start): arithmetic-2 64.5% (sampled 50.8%)
step 1: reward 0.014  arithmetic-2 64.0%  (8.22e+10 FLOPs/s, about 31 s to go)
step 2: reward 0.043  arithmetic-2 63.6%  (1.12e+11 FLOPs/s, about 21 s to go)
step 3: reward 0.029  arithmetic-2 63.7%  (1.35e+11 FLOPs/s, about 15 s to go)
step 5: reward 0.027  arithmetic-2 64.2%  (1.76e+11 FLOPs/s, about 8 s to go)
step 6: reward 0.041  arithmetic-2 64.0%  (1.26e+11 FLOPs/s, about 9 s to go)
step 10: reward 0.020  arithmetic-2 64.8% (sampled 51.4%)  (9.39e+10 FLOPs/s, about 0 s to go)
done: ran 10 steps in 33 s (2.86e+12 FLOPs in all), last reward 0.020
```

This is GRPO, 64 prompts a step, 8 answers each. The run is graded at step 0 (the warm start
itself) and on a geometric schedule after. Telemetry is on: every step records 53 metrics, and
anomaly detectors (loss and gradient spikes, divergence, entropy collapse) would print a
warning here if one fired. Read a few of the last step's:

```bash
uv run python -c "
import os; from pathlib import Path
from scalelab.telemetry.sink import read_metrics
home = Path(os.environ['SCALELAB_HOME'])
run = next((home / 'gradings').iterdir()).name
for row in read_metrics(home, run):
    if row.step == 10 and row.layer is None and row.metric.split('/')[0] in ('policy', 'optim'):
        print(f'{row.metric:24} {row.value:.4g}')
"
```

```
policy/entropy           0.1219
policy/flat_groups       0.875
policy/ratio_mean        1
policy/ratio_max         1
policy/clip_fraction     0
policy/kl_old            0
policy/advantage_mean    1.572e-09
policy/advantage_std     0.3307
policy/kl_ref            0.00202
optim/loss               8.103e-05
optim/grad_norm          0.1002
optim/grad_clipped       0
```

`policy/flat_groups` 0.875 says 7 groups in 8 got the same reward on every answer, so they teach
nothing: m1 rarely gets a 2-digit product right yet.

## 6. Did RL help?

```bash
uv run scalelab report gains
```

```
m1 grpo: experiment deb1f96192f77694 at 688766a3
  seeds 0; last steps 10
  eval set            n   first    last  gain, mean [95% CI]       gain, IQM [95% CI]        verdict
  arithmetic-2     1000   64.5%   64.8%  +0.3 [-0.8, +1.4] pp      +0.3 [-0.8, +1.4] pp      unclear
  arithmetic-2/*    319    6.3%    7.5%  +1.3 [-1.3, +3.8] pp      +1.3 [-1.3, +3.8] pp      unclear
```

Not yet: ten steps and one seed are too few, and the report says so rather than calling
+1.3 pp a gain. The intervals bootstrap over seeds and problems together.

## 7. The real runs

The 85 RL runs (and their 5 warm starts) behind Write-up 2 are a public dataset, and the
download is itself a home (312 MB):

```bash
uvx --from huggingface_hub hf download argo-navis-research-laboratory/scalelab-runs --repo-type dataset --local-dir ~/scalelab-runs
export SCALELAB_HOME=~/scalelab-runs
```

`scalelab report gains` now prints every configuration (about 30 s). Its largest size, m5:

```
m5 grpo: experiment 5b7f83e07935b6fa at 0413a34f
  arithmetic-4/*    316   13.6%   22.9%  +9.3 [+5.4, +13.3] pp     +9.5 [+5.4, +13.4] pp     improved
m5 dapo: experiment bc652cd7a72d1ce9 at 0413a34f
  arithmetic-4/*    316   13.6%   31.3%  +17.7 [+12.5, +23.2] pp   +17.3 [+12.0, +23.6] pp   improved
```

DAPO looks twice as good. Both had the same FLOPs budget, but DAPO (here a variant that drops
groups whose answers all score the same without refilling the batch) skips their log-prob and
training passes, so it ran 3.4 times the steps. Compare the five seeds pair by pair,
first at equal FLOPs, then at equal samples:

```bash
uv run scalelab versus 5b7f83e07935b6fa-s{0..4}-0413a34f --against bc652cd7a72d1ce9-s{0..4}-0413a34f --at flops
uv run scalelab versus 5b7f83e07935b6fa-s{0..4}-0413a34f --against bc652cd7a72d1ce9-s{0..4}-0413a34f --at samples
```

```
m5: other - base +4.69 pp [+2.30, +7.11], 5 seeds x 653 problems, 95%
m5: other - base +0.43 pp [-1.29, +2.17], 5 seeds x 653 problems, 95%
```

At equal FLOPs DAPO wins by 4.7 pp; given the same number of sampled answers, no difference is
detected. Most of its win comes with running more steps on the same compute.

## Where next

- [Write-up 2](writeups/02-scaling.md) fits these runs across five model sizes and tests how
  far the fit predicts; its last section gives the commands.
- `uv run scalelab --help` lists every command; `uv run scalelab <command> --help` its options.
- The real runs used `configs/sweeps/` on a rented T4: `scalelab sweep plan` prices a sweep,
  `sweep submit` queues it, and `scalelab worker` runs the queue.
