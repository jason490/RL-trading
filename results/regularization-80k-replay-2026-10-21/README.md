# Retained80k SAC replay — diagnostic, strict score tolerance failed

The interrupted candidate snapshot has80k weights and curve, but its legacy
validation file is stale50k-selected output. This fixed replay uses only the
retained80k model and original26 development episodes to recover diagnostic
paths. It does not substitute for a100k final or complete/promote the study.

Replayed mean5.10354112% versus recorded80k curve5.10354333%. Absolute return
difference2.21088e-8 exceeds the declared1e-10 replay tolerance: **failure remains
failure**. The mean dollar difference at$100k initial equity is−$0.00221. No
threshold was relaxed and no exact parity/acceptance is claimed. Both scores
round to5.10%; the original unassessable screen stays unassessable.

| Diagnostic replay | Policy return | Matched holding | Excess |
| --- | ---: | ---: | ---: |
|22 stocks | 6.92% | 32.63% | −25.71pp |
|4 ETFs | −4.87% | 19.47% | −24.33pp |

Separate2021–2023 development episodes, not annualized or a pooled investable
portfolio. Median stock excess−15.92pp. Mean long exposure72.64% stocks/72.34%
ETFs; short exposure/contribution0 in both groups. Total returns on AMZN−41.82%,
DIS−56.66%, NEE−25.75%, all with zero short P&L:0/3 joint successes. Future
full-period decline labels are reporting-only, never inputs or selection labels.

Saved-path incremental cost drag8.02pp stocks/7.04pp ETFs, non-cost excess
−17.69/−17.30pp. Cost add-back is descriptive, not a causal zero-cost replay;
this policy's gap is not solely transaction costs. No coefficient sweep,
more steps, confirmation or automatic allocation follows.

Weights, alpha, optimizer settings/update counts, source hashes and normalization
remain unchanged. Saved-path accounting passes;20 report/participation/direction
checks pass without fitting. Complete diagnostic paths/charts remain under
`artifacts/regularization-80k-replay-2026-10-21`. Original stale files and partial
checkpoint remain untouched. No local optimizer updates, retired-test access,
provider calls or Colab. Conservative ledger37.787213/150 and stop flag unchanged.

[Exact diagnostic results, tolerance failure and source hashes](summary.json),
[replay script](../../scripts/replay_partial_regularization.py).
Use new result/artifact paths when reproducing. CPU flush mode is explicitly off
in this isolated replay, and no numerical training setting changes. The lack of
original80k action paths prevents attribution of tiny cross-runtime differences.
