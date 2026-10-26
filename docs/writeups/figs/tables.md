# The cost model's record

Made by `scalelab report cost` from each device's validation.json (cost validate) and
the measured changes in reports/ (cost measure). Errors are predicted / measured - 1.

## Whole steps and phases, predicted vs measured

| device | part | runs | MAPE | bias | worst | median | spread |
|---|---|---:|---:|---:|---:|---:|---:|
| a100-40g | generate | 9 | 7.9% | -6.9% | -18.4% | -6.6% | 6.7% |
| a100-40g | logprobs | 9 | 10.1% | -7.8% | -27.5% | -4.0% | 13.2% |
| a100-40g | train | 13 | 14.5% | -14.2% | -38.7% | -8.1% | 14.3% |
| a100-40g | step | 13 | 13.2% | -12.9% | -41.5% | -12.4% | 12.2% |
| t4 | generate | 24 | 11.8% | +9.0% | +17.9% | +15.2% | 3.5% |
| t4 | logprobs | 24 | 4.2% | +1.3% | +10.4% | +0.3% | 5.3% |
| t4 | train | 28 | 8.8% | -5.3% | -21.7% | -4.7% | 8.0% |
| t4 | step | 28 | 4.1% | +0.9% | -10.1% | +1.0% | 6.1% |

## Where an RL step's time goes (predicted fp32, without the cut; measured laps)

| device | FLOPs a step | phase | predicted s | compute | memory | launch | measured s |
|---|---:|---|---:|---:|---:|---:|---:|
| a100-40g | 8.7e+11 | generate | 1.2556 | 0.0% | 1.6% | 98.4% | 1.4828 |
| a100-40g | 8.7e+11 | logprobs | 0.0336 | 62.2% | 37.8% | 0.0% | 0.0397 |
| a100-40g | 8.7e+11 | train | 0.1076 | 37.0% | 4.3% | 58.7% | 0.1766 |
| a100-40g | 4.95e+12 | generate | 1.9460 | 0.2% | 6.5% | 93.3% | 2.1700 |
| a100-40g | 4.95e+12 | logprobs | 0.1415 | 77.2% | 22.8% | 0.0% | 0.1643 |
| a100-40g | 4.95e+12 | train | 0.2956 | 61.5% | 7.9% | 30.6% | 0.3789 |
| a100-40g | 1.43e+13 | generate | 2.7181 | 0.4% | 10.5% | 89.1% | 2.8917 |
| a100-40g | 1.43e+13 | logprobs | 0.3685 | 83.7% | 16.3% | 0.0% | 0.3581 |
| a100-40g | 1.43e+13 | train | 0.6612 | 73.3% | 6.4% | 20.4% | 0.7010 |
| t4 | 8.7e+11 | generate | 1.3893 | 0.1% | 19.5% | 80.4% | 1.5229 |
| t4 | 8.7e+11 | logprobs | 0.1840 | 42.2% | 57.8% | 0.0% | 0.1667 |
| t4 | 8.7e+11 | train | 0.3402 | 36.2% | 55.4% | 8.4% | 0.3634 |
| t4 | 4.95e+12 | generate | 2.4894 | 0.6% | 30.2% | 69.2% | 2.1784 |
| t4 | 4.95e+12 | logprobs | 0.7276 | 67.7% | 32.3% | 0.0% | 0.7192 |
| t4 | 4.95e+12 | train | 1.1784 | 63.9% | 36.1% | 0.0% | 1.2316 |
| t4 | 1.43e+13 | generate | 3.9272 | 1.0% | 37.2% | 61.8% | 3.4231 |
| t4 | 1.43e+13 | logprobs | 1.8113 | 74.6% | 25.4% | 0.0% | 1.9865 |
| t4 | 1.43e+13 | train | 2.8193 | 72.2% | 27.8% | 0.0% | 3.1792 |

## Changes, predicted vs measured speedups (held: within the device's validated MAPE)

| device | change | kind | FLOPs a step | predicted | measured | error | held |
|---|---|---|---:|---:|---:|---:|---|
| a100-40g | padding-cut | RL | 8.7e+11 | 0.98x | 0.98x | +0.2% | yes |
| a100-40g | padding-cut | RL | 4.95e+12 | 1.00x | 1.02x | -1.9% | yes |
| a100-40g | padding-cut | RL | 1.43e+13 | 1.02x | 1.04x | -1.7% | yes |
| a100-40g | padding-cut | warm start | 8.23e+10 | 0.45x | 0.38x | +19.1% | no |
| a100-40g | padding-cut | warm start | 4.59e+11 | 0.75x | 0.63x | +19.1% | no |
| a100-40g | padding-cut | warm start | 1.32e+12 | 1.05x | 0.93x | +12.3% | yes |
| a100-40g | padding-cut | warm start | 2.91e+12 | 1.29x | 1.30x | -0.2% | yes |
| t4 | fp16 | RL | 8.7e+11 | 1.05x | 0.97x | +7.9% | no |
| t4 | fp16 | RL | 4.95e+12 | 1.37x | 1.25x | +9.5% | no |
| t4 | fp16 | RL | 1.43e+13 | 1.69x | 1.65x | +2.4% | yes |
| t4 | fp16 | warm start | 8.23e+10 | 1.81x | 1.44x | +25.6% | no |
| t4 | fp16 | warm start | 4.59e+11 | 2.44x | 2.09x | +16.7% | no |
| t4 | fp16 | warm start | 1.32e+12 | 2.84x | 2.41x | +17.7% | no |
| t4 | fp16 | warm start | 2.91e+12 | 3.14x | 2.82x | +11.4% | no |
| t4 | padding-cut | RL | 8.7e+11 | 1.03x | 1.04x | -1.0% | yes |
| t4 | padding-cut | RL | 4.95e+12 | 1.07x | 1.08x | -1.2% | yes |
| t4 | padding-cut | RL | 1.43e+13 | 1.10x | 1.11x | -0.7% | yes |
| t4 | padding-cut | warm start | 8.23e+10 | 1.01x | 0.82x | +23.6% | no |
| t4 | padding-cut | warm start | 4.59e+11 | 1.48x | 1.57x | -5.3% | no |
| t4 | padding-cut | warm start | 1.32e+12 | 1.73x | 1.82x | -5.2% | no |
| t4 | padding-cut | warm start | 2.91e+12 | 1.87x | 1.94x | -3.6% | yes |
