# Write-up 2: how RL gains scale with model size, and how far one fit predicts

ScaleLab M4, 2026-11-17. Figures: `scalelab report scaling` (see Reproduce).

## In short

- We ran 50 RL runs from scratch-trained models of 0.80M to 59.5M non-embedding parameters
  (five sizes x five seeds x two algorithms, GRPO and DAPO-style), each on the same FLOP budget
  per size, and fitted **one family of sigmoid curves in compute** whose three parameters
  move linearly in log size. It fits all 50 runs to **1.4 pp (GRPO) and 1.6 pp (DAPO-style) rms**.
- **The fit pins R where it was measured, not where it is going.** A seed bootstrap gives
  R at each size's last compute to within about 1 pp, but the ceiling A spans about 45 to
  55 pp from m3 up (GRPO m3-m5 roughly 26-82%).
- **Backtest: close at m4, over-predicts m5.** Fitted on m1-m3, the family predicts
  m4 from its starting accuracy alone to 1.5 pp (GRPO) and 1.2 pp (DAPO-style) rms over its
  trajectory, but under-predicts its last grading (GRPO 19.2% v 21.5% measured, DAPO-style 24.3%
  v 25.1%). Fitted on m1-m4, it over-predicts m5 by about 3 pp at the end of the run (GRPO
  15.7% v 12.2%, DAPO-style 19.1% v 16.9%). All four measured values fall outside the 95%
  bootstrap interval for the predicted mean, which resamples seeds only: it is not a
  predictive interval, and it leaves out the form's own uncertainty.
- **DAPO-style beats GRPO at equal FLOPs (+3.6 to +7.0 pp at every size); at equal samples no
  difference is detected** (-0.6 to +1.6 pp, every interval spans zero, but they allow up
  to +3.3 pp). Equal FLOPs buys DAPO-style 2.2-3.4x the steps. It is a variant of DAPO (it never
  refills the batch it filters: see Setup), and it differs from GRPO in five ways at once,
  so this doesn't isolate any one of them.
- **RLOO, run alongside (25 runs), shows no consistent difference from GRPO** at equal FLOPs
  or equal samples, despite 1.21x the steps; DAPO-style beats it at every size at equal FLOPs
  (+4.4 to +5.3 pp).
- None of this was pre-registered. The plan was to commit a prediction for two larger
  sizes before running them; those runs were cut, and the backtests stand in for them. Some
  decision rules (the task bands, the RL budget, the prediction's algorithm) were written in
  a private plan before the data they decide; no public, timestamped record shows that.

## Setup

| model | layers | d_model | heads | N (non-embedding) | R_0 |
|---|---:|---:|---:|---:|---:|
| m1 | 4 | 128 | 2 | 0.80M | 16.4% |
| m2 | 6 | 256 | 4 | 4.82M | 11.3% |
| m3 | 8 | 384 | 6 | 14.2M | 12.9% |
| m4 | 10 | 512 | 8 | 31.6M | 12.4% |
| m5 | 12 | 640 | 10 | 59.5M | 6.7% |

- **Warm starts.** Each size is first trained by next-token prediction on 20M tokens of
  worked arithmetic solutions (one seed-0 warm start a size; every RL seed starts from it).
  We first trained them on 100M tokens, but that left m5 at 66% or more on multiplication
  at every band up to 5 digits (greedy pass@1 on the * problems of the same frozen eval sets;
  90.2% and 66.5% at 4 and 5 digits), with little room for RL; we dropped to 20M after seeing it.
- **Task.** Multiplication at 4 and 5 digits, chosen by a rule fixed before looking: the
  bands where the smallest model solves at least 5% and the largest at most 50%.
  **R** = greedy pass@1 over the two frozen eval sets pooled (653 problems). **R_0** = R
  before RL. At 20M tokens R_0 *falls* with size: a fixed token budget undertrains the
  bigger models.
  
- **The eval sets were seen while designing.** The 100M warm starts' results on them led to
  20M tokens, the bands were picked on them by the rule above, the pilot that set the budget
  was graded on them, and every RL result is reported on them. No eval problem is trained on,
  but these numbers are not from untouched held-out data.

- **RL.** GRPO (clip 0.2, KL 0.04 to the warm start, each answer's loss averaged, then
  averaged over answers) and a DAPO-style variant (clip 0.2/0.28, token-mean loss, no KL, a
  soft penalty for overlong answers, groups whose 8 answers all score the same dropped). The
  DAPO paper samples more until the batch is full again; ours does not, so each update
  learns from the groups that survive (about half at m1, a quarter at m5). "DAPO-style" below
  always means this variant. 64 prompts x 8 answers a step, constant lr 1e-5, fp16, on a T4
  (one GPU at a time: Colab's, then two rented machines).
  Rewards: 1 for the right answer, 0 otherwise.
- **Budget.** Each size gets the FLOPs of 1000 GRPO steps as the cost model prices them
  (`rl.flops`: m1 1.3e15 to m5 8.7e16). A pilot at m1, m3 and m5 showed that none reached
  90% of its fitted ceiling by step 1000. GRPO runs about 1010-1020 steps on that budget;
  DAPO-style 2233-3446: a dropped group's generation is billed, but it skips log-probs and
  training, and DAPO-style has no KL reference pass (at m5 a DAPO-style step costs 2.5e13 FLOPs, a
  GRPO step 8.5e13).
- **Seeds.** A 10-seed study at m2 and m4 measured the seed spread of final R (sd 0.57 and
  1.02 pp). Power, approximate: for all 252 assignments of the 10 seeds to two labelled arms
  of five (126 distinct partitions, each under both labellings), the arms' seeds matched in
  order as `versus` matches seed s with seed s, and the seeds-and-problems
  bootstrap run on the per-problem differences. Mean 95% half-width 1.69 pp (m2), 1.90 pp
  (m4); with SE = half-width / 1.96, power at 3 pp = Phi(3 / SE - 1.96): 93% and 87%. This
  assumes the DAPO-style runs vary across seeds as GRPO's do and that seed s of two algorithms pairs no more
  closely than two GRPO seeds. The real equal-samples intervals came out wider (half-widths
  1.7-2.7 pp: power at 3 pp roughly 60-93%), so five seeds resolve about 3 pp at best.
- **C** = the RL FLOPs a run has spent (sampling, log-probs, training), summed over steps.
  Gradings run at steps 0, 1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000, 1778, 3162
  (four per decade), every 1000 steps, and at the run's last step.

## One family of curves

Each run follows a sigmoid in log compute (the form of Khatri et al., below):

```
R(C) = R_0 + (A - R_0) / (1 + (C_mid / C)^B)
```

R_0 is measured, never fitted. Across sizes, with x = log10(N / 1e7):

```
logit A = a0 + a1 x      ln C_mid = c0 + c1 x      ln B = b0 + b1 x
```

Six numbers fit every grading after step 0 of every run, seeds pooled, by least squares
([scaling.py](../../src/scalelab/analysis/scaling.py), `scalelab joint`).

![R against RL compute, each run's gradings and the family's curve per size](figs/scaling-curves.png)

*Dots: every grading of every seed. Lines: the fitted family at each size's mean R_0.*

| | m1 | m2 | m3 | m4 | m5 |
|---|---:|---:|---:|---:|---:|
| GRPO final R, measured | 41.2% | 26.9% | 23.1% | 21.5% | 12.2% |
| GRPO, fitted | 41.0% | 27.3% | 23.0% | 20.0% | 14.6% |
| GRPO A | 49.3% | 41.3% | 36.7% | 33.4% | 30.9% |
| GRPO A, 95% interval | 46-76% | 38-71% | 33-78% | 29-82% | 26-69% |
| DAPO-style final R, measured | 46.8% | 34.0% | 29.2% | 25.1% | 16.9% |
| DAPO-style, fitted | 46.8% | 35.1% | 28.5% | 24.1% | 16.8% |
| DAPO-style A | 60.3% | 84.4% | 92.1% | 95.3% | 97.0% |
| DAPO-style A, 95% interval | 56-69% | 69-91% | 58-97% | 48-98% | 40-99% |

*Intervals: 4000 seed bootstraps (resample each size's seeds, refit). Residuals: GRPO 1.39
pp rms over 350 gradings, DAPO-style 1.56 pp over 410.*

- **Gains fall with size**, at this budget: GRPO adds 24.8 pp at m1 and 5.5 pp at m5.
- **Only m1 bends toward a ceiling.** From m2 up the runs are still on the lower half of
  the sigmoid, so a low ceiling reached soon and a high one reached far beyond the data
  fit the gradings about equally well. The DAPO-style fit takes the second reading (A near 100%,
  C_mid at m5 near 1e20 FLOPs, about 1000x past the last grading), GRPO's the first. The
  intervals are lopsided for the same reason: A and C_mid are not measured here.
- **GRPO misses at m4 and m5** by more than seed noise: m5's measured 12.2% lies below the
  fitted interval (13.7-14.8%). The family's straight lines in log size are not the shape
  of these data at the largest size.

## Backtest: predict the largest size before its RL

Fit the family without the largest size, then predict that size's whole curve from its
measured R_0 alone (`scalelab backtest`), and compare with its five seeds.

![m5 predicted from m1-m4, against its measured gradings](figs/scaling-backtest.png)

| fitted on, predicted | algorithm | fit rms | held-out rms | bias | last R predicted [95%] | measured |
|---|---|---:|---:|---:|---:|---:|
| m1-m3, m4 | GRPO | 1.39 pp | 1.46 pp | -0.85 pp | 19.2% [18.9, 19.6] | 21.5% |
| m1-m3, m4 | DAPO-style | 1.66 pp | 1.23 pp | -0.02 pp | 24.3% [23.7, 24.9] | 25.1% |
| m1-m4, m5 | GRPO | 1.32 pp | 2.22 pp | +1.92 pp | 15.7% [15.4, 16.0] | 12.2% |
| m1-m4, m5 | DAPO-style | 1.56 pp | 2.37 pp | +1.84 pp | 19.1% [17.9, 19.4] | 16.9% |

*Held-out rms over every grading of the predicted size's five seeds; bias = predicted
minus measured, averaged. The interval resamples the fitted sizes' seeds only, not the
form, so it is narrow.*

Over its whole trajectory the m4 prediction is as good as the fit itself, but at the last
grading it falls short by 2.3 pp (GRPO) and 0.8 pp (DAPO-style), outside the seed interval. The
m5 prediction starts close (+0.6 pp at step 1 for GRPO) and drifts high as compute grows
(+2.3 pp at step 100, +3.5 pp at the end). m4 to m5 is a smaller step in size (1.9x) than
m3 to m4 (2.2x), so distance is not the cause. Our hypothesis: m5 is the most undertrained
warm start (lowest R_0, highest N per token), and nothing in the family knows about
warm-start tokens. It is not tested here: one warm start a size, no token budget varied.

## DAPO-style against GRPO: what "equal" means

`scalelab versus` pairs runs by size and seed and reads DAPO-style at its recorded grading
nearest to GRPO's final measure (no interpolation), then bootstraps the per-problem
differences over seeds and problems. At equal FLOPs that is each run's last grading
(cumulative FLOPs within 1%); at equal samples, the DAPO-style run's grading at step 1000
against GRPO's last (step 1011-1018), so GRPO has generated 1.1-1.8% more answers.

![DAPO-style minus GRPO at equal samples and at equal FLOPs, per size](figs/scaling-versus.png)

| DAPO-style - GRPO, pp [95%] | m1 | m2 | m3 | m4 | m5 |
|---|---|---|---|---|---|
| equal FLOPs | +5.6 [2.5, 8.6] | +7.0 [4.8, 9.3] | +6.1 [4.0, 8.2] | +3.6 [1.3, 6.2] | +4.7 [2.3, 7.1] |
| equal samples | +0.7 [-2.1, 3.3] | +0.8 [-1.4, 2.9] | +1.6 [-0.1, 3.3] | -0.6 [-2.5, 1.4] | +0.4 [-1.3, 2.2] |

At equal FLOPs DAPO-style wins at every size. Our FLOPs count what a step computes: the DAPO-style
variant's dropped groups are generated and billed, then skip log-probs and training, and DAPO-style has no
KL reference pass either. Generation is a small share of the FLOPs (15% of a GRPO step at
m5) but dominates the time at this scale: in these fp16 runs' own timings, 49% (m5) to 83%
(m1) of a GRPO step and 71-86% of a DAPO-style step ([Write-up 1](01-cost-model.md) explains
why). So equal FLOPs buys DAPO-style 2.2-3.4x the steps. In FLOP bookkeeping at m5, a GRPO step
without its reference pass would cost about 17% less (7.05e13 v 8.52e13 FLOPs), and the passes DAPO-style skips cover the
remaining 2.8x; RLOO (no reference, no filtering) ran 1.21x GRPO's steps at every size, as that predicts.
That splits what the counter charged; it is not an ablation. Read at equal samples (GRPO's
~1017 steps against the DAPO-style runs' 1000; samples count every generated answer), no difference is
detected at any size, though the intervals allow up to +3.3 pp. Clipping, KL, loss
averaging and overlong shaping differ too, so isolating the filtering needs an ablation
(GRPO plus filtering only) that was not run.

Equal wall-clock time is the comparison a practitioner wants, and these runs cannot answer
it. Per step DAPO-style is cheaper (m5: 1000 steps took 2.9-3.7 h, GRPO's 4.7-5.1 h), and its
gradings after step 1000 are too sparse to read it at GRPO's hours. The answer is open,
not null.

## RLOO, the third algorithm

RLOO ran in the same sweep (five sizes, five seeds, the same FLOP budget): each answer's
advantage is its reward minus the mean of the other seven in its group, no KL penalty (so
no reference pass), GRPO's clipping and loss averaging, no filtering. An m5 step costs
7.05e13 FLOPs against GRPO's 8.52e13, so the budget buys about 1230 steps to GRPO's 1017
(1.21x). Read the same way as above, GRPO as the base:

| pp [95%] | m1 | m2 | m3 | m4 | m5 |
|---|---|---|---|---|---|
| RLOO - GRPO, equal FLOPs | +0.3 [-2.4, 3.1] | +2.3 [0.2, 4.6] | +0.8 [-1.1, 2.8] | -1.0 [-3.0, 0.9] | +0.3 [-1.8, 2.2] |
| RLOO - GRPO, equal samples | -0.8 [-3.2, 1.7] | +1.0 [-1.4, 3.2] | +0.6 [-1.1, 2.3] | -1.3 [-3.3, 0.7] | -0.4 [-2.3, 1.3] |
| DAPO-style - RLOO, equal FLOPs | +5.3 [1.6, 9.0] | +4.7 [2.5, 7.0] | +5.3 [3.2, 7.3] | +4.7 [2.1, 7.1] | +4.4 [1.9, 7.1] |

One of five equal-FLOPs intervals excludes zero (m2), with no trend in size; at equal
samples none does (matched as above). At equal wall-clock hours the match is loose: past
step 562 RLOO is graded only at 1000 and at its end, so its nearest grading lands 9% short
of GRPO's hours to 13% past them; no interval excludes zero there either (-1.3 to +1.0 pp).
RLOO's 21% more steps came with no detectable difference, where DAPO-style's 2.2-3.4x came
with +3.6 to +7.0 pp. That contrasts two recipes (RLOO also changes the baseline and drops
the KL penalty); it doesn't show what causes either result. At the same
step RLOO's policy entropy tracks DAPO-style's (m3, step 1000: 0.0163 and 0.0164 nats a
token, GRPO 0.0201): GRPO, the one with a KL penalty, stays higher.

The family fits RLOO's 25 runs to 1.50 pp rms over 350 gradings; with 4000 seed bootstraps
its ceiling A spans 28-53% at m3, 23-41% at m4, 20-32% at m5: narrower than GRPO's, but a
run of 400 bootstraps had put the upper ends at 92-98%, so a few resamples still jump to a
far-off ceiling and the top edge depends on how many are drawn. Backtests: m1-m3 -> m4 1.27 pp
held-out rms, bias -0.30 pp, last grading 20.4% [19.9, 20.9] v 20.5% measured (inside, the
only one of the three algorithms); m1-m4 -> m5 2.40 pp, +1.96 pp, 16.1% [15.9, 16.4] v 12.5%
(+3.7 pp). All three algorithms over-predict m5's last grading, by 2.2-3.7 pp.

## Against the published scaling laws

- **Khatri et al., "The Art of Scaling Reinforcement Learning Compute for LLMs"**
  ([arXiv 2510.13786](https://arxiv.org/abs/2510.13786)) fit the same sigmoid to one run's
  early compute and extrapolate that run to far more compute, with pretrained models and
  very large budgets. We fit it across five sizes at once and extrapolate to a size never
  trained. Our data agree that the sigmoid fits a run; they add
  that from small runs on the lower half of the curve, its ceiling cannot be read.
- **Tan et al.** ([arXiv 2509.25300](https://arxiv.org/abs/2509.25300)) fit a power law in
  compute for test loss, with an exponent that depends on size, on pretrained Qwen2.5
  models from 0.5B to 32B, and predict a held-out 72B model. We did not fit their form;
  that comparison is left open. Their extrapolation across size works where ours fails at
  m5. One plausible difference: their models share one pretraining recipe scaled with
  size, ours share one fixed token budget, which leaves the larger models further from
  trained.

Both studies are far larger than this one. This write-up adds a small, fully reproducible
case with seeds, intervals and a held-out size, not a new law.

## Limits

- **Not pre-registered.** The family's form, its fitting choices and the backtests were
  all chosen after the sweep's data existed. The rules fixed beforehand are only in a private
  plan.
- **One warm start a size.** The size axis has one draw per size; RL seeds share it. R_0
  falls with size, so size and undertraining move together.
- **A DAPO-style variant, five changes at once.** No refill, so the variant's batch shrinks; and it
  differs from GRPO in clipping, KL, loss averaging, overlong shaping and filtering.
- **Pointwise intervals.** Five sizes compared, no correction for multiple comparisons; the
  intervals cover RL seeds and eval problems, not warm-start variation.
- **One task, one learning rate, one budget rule.** Multiplication at 4-5 digits (653
  problems), lr 1e-5 at every size, 1000 GRPO steps' FLOPs. A skill with unlimited
  supervised data is an odd target for RL; it was chosen because it is cheap to verify.
- **Small.** 0.8M to 59M parameters, character-level, trained from scratch. Nothing here
  says how a 1B model behaves.
- **One form.** Only the sigmoid family was fitted; no alternative form was tried.
- **Compute.** About 350 T4 hours in all for M4 (warm starts, pilot, seed study, the 75-run
  main sweep with RLOO).

## Reproduce

The runs: config hash per size and algorithm, seeds 0-4, code at commit 0413a34f, so each
run id is `<hash>-s<seed>-0413a34f`.

| | m1 | m2 | m3 | m4 | m5 |
|---|---|---|---|---|---|
| GRPO | 04568b9187309a00 | a2f03ca3586f3bd7 | f18c79f8cdc97d83 | c885be542cd1b57b | 5b7f83e07935b6fa |
| DAPO-style | 9e52d8444af3da4a | fda59e83fa576a05 | d6fcb97580964ec2 | 9f1a8171ddc6e4f5 | bc652cd7a72d1ce9 |
| RLOO | f2f88fb691139ca0 | c39204b198630d96 | 97142389552b13c6 | e0614bfb878c6916 | 5edf25932bb11f3a |

With their registry, gradings and metrics under `$SCALELAB_HOME` (the public dataset is such a
home) and the 25 run ids of each algorithm in `$GRPO`, `$DAPO` and `$RLOO`:

```bash
uv run scalelab report scaling $GRPO $DAPO --out docs/writeups/figs
uv run scalelab joint $GRPO --bootstrap 4000
uv run scalelab backtest $GRPO --bootstrap 4000
uv run scalelab versus $GRPO --against $DAPO --at flops
uv run scalelab versus $GRPO --against $DAPO --at samples
uv run scalelab power $GRPO
uv run scalelab versus $GRPO --against $RLOO --at flops
uv run scalelab joint $RLOO --bootstrap 4000
uv run scalelab backtest $RLOO --bootstrap 4000
```

The sweep files are [configs/sweeps](../../configs/sweeps) (`main-grpo.yaml`,
`main-dapo.yaml`, `main-rloo.yaml`). The code: [scaling.py](../../src/scalelab/analysis/scaling.py),
[versus.py](../../src/scalelab/analysis/versus.py),
[scalingreport.py](../../src/scalelab/analysis/scalingreport.py).
