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
