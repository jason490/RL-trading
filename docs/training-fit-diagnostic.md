# Fitted-period policy diagnosis

The fixed TD3 state screen failed alongside earlier PPO/SAC state experiments.
Before another model change, distinguish a failure to approach holding on fitted
history from a training/development contrast. Neither observation alone proves
underfitting or overfitting: market regimes, durations, initialization, source
revisions and optimization objectives differ.

[Protocol](../configs/experiments/training_fit_diagnostic.json) fixes final weights
from the existing 50k seed-11 full-state native PPO, SAC and TD3 runs. PPO used
wide initialization; the others standard. This is descriptive within-policy
analysis, not a controlled algorithm ranking. No best-checkpoint selection,
model fitting, optimizer call, normalization update, provider request or Colab
allocation is allowed.

## Window and accounting

Use only existing training bars/features in immutable dataset 672a2162….
December 29, 2017 is the decision anchor; executions run January 2, 2018 through
December 2, 2020. The latter is the actual existing training cutoff after split
purging; never reconstruct missing December bars. All 26 symbols must have the
same endpoints. Preserve aligned bars/features/ATR and explicit `train` labels.
The ordinary evaluator continues to reject `train`; a separately named training
diagnostic invokes the same simulator loop without mislabeling fitted data as
validation. Preprocessing remains the original training-fitted normalization,
not a new fit or an independent holdout.

Use unchanged costs, corporate actions, next-session fills, actions and holding
limits, with matched holding on each actual episode. Compare per-group excess
and median stock excess, rather than raw returns across different periods.
Training comparability flags require stock/ETF excess and median stock excess
at least -2pp. Flags diagnose only: they cannot override failed development
gates or authorize promotion. A training-pass/development-fail contrast is
consistent with a generalization gap, not proof of its cause. Failure on both
periods is weak benchmarked fitted-period performance, not a mathematical test
of model capacity or optimizer correctness.

## Reproducibility

Source states must be complete with verified owned cleanup. Latest weights,
metadata and saved final development paths are checksummed; existing development
accounting is validated against immutable bars. Policy tensors, update counts,
normalization and source files must remain unchanged. Full derived training
paths/charts remain under ignored artifacts; compact group/symbol results are
tracked. Retired test data stays unopened, and future declining-stock cohorts
remain reporting-only.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/diagnose_training_fit.py \
  --output artifacts/training-fit/summary.json \
  --artifacts artifacts/training-fit/paths
```

Use new paths; the command refuses to overwrite evidence. Current status:
protocol and explicit training-window implementation prepared; 20 slicing,
split-isolation, report-contract and direction-report checks pass without
fitting. Actual saved-policy replay is not complete yet. Project usage remains
36.149107/150; all owned Colab sessions are stopped.

## Actual inference-only outcome

The committed replay completed:26 episodes/19,136 execution rows per model.
SAC fitted stocks67.06% versus holding58.15%, ETFs53.86% versus47.15%, median
stock excess-0.46pp: training comparability passes. Its same final policy
development mean7.93% versus30.60% and group excess-23.94/-15.70pp fail. This
contrast is consistent with a generalization gap, not proof of a single cause.
PPO/TD3 fail the training flags; do not describe all learners as only overfit.

All13 source hashes, policy tensors including SAC alpha, optimizer update counts
and normalization are unchanged. No fitting, checkpoint selection, retired data
or Colab. Three published assets verify; duplicate release copies removed while
full derived paths remain. Project36.149107/150; all owned runtimes stopped.
[Actual evidence](../results/training-fit-2026-10-20/README.md).

Next audit existing expanded survivor data for a fixed SAC training-diversity
hypothesis, keeping original normalization and validation unchanged with a fresh
matched original-universe control. Failed prior PPO expansion remains negative
evidence. No allocation before coverage/source/sampling checks and a separately
committed bounded protocol. Preserve all absolute/multi-seed/prospective gates.

## Fitted decline/rebound decisions

A saved-path20-session/10% audit finds older50k SAC short requests40.63% before
continued declines versus31.44% before rebounds, conditional on accepted
protective decisions. Mean forward return after past declines is positive;
blind decline oversampling need not teach persistent shorting. Future buckets
are reporting-only and overlapping, not independent trades or profit evidence.
PPO/TD3 differ in source/initialization; no causal algorithm ranking.
[Results, caveats and hashes](../results/training-regimes-2026-10-20/README.md).
No fitting or new inference/Colab was needed. Next audit actual100k diversity
final policies on the same fixed fitted window before another hypothesis.
