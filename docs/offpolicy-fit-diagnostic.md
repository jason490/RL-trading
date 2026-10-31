# Recurrent off-policy fitted-period diagnosis

This inference-only audit follows the failed100k
[GRU-SAC/SAC comparison](../results/recurrent-offpolicy-2026-10-31/README.md).
It asks whether the already-selected and final policies also underperform matched
holding on eligible fitted history. It neither reopens the family nor promotes
a model. No optimization, checkpoint selection, new data/provider call, retired
test access or Colab allocation is permitted.

## Fixed inputs and scope

The [protocol](../configs/experiments/recurrent_offpolicy_fit_diagnostic.json)
has canonical SHA256
`cfa2cac907c5292762b8a2b12dbdd97254d01b71ad988188049d216a76ff4fe2`.
It binds the accepted screen receipt, immutable cdb93a33 dataset and original
normalizer. Replay exactly four states: GRU-SAC selected10k/final100k and native
SAC selected10k/final100k, seed11. Verify the complete accepted file inventories
before loading. RecurrentPPO final is excluded because its exact final weights
and complete Adam state already match the historical fitted-period study.

Use the original22 stocks and SPY/QQQ/IWM/DIA, with2017-12-29 as decision anchor
and2018-01-02 through2020-12-02 as executions. All source training episodes must
remain aligned, chronological and before2021, including the85 training-only
symbols excluded from the common replay panel. Keep explicit `train` labels.
Standard evaluation still rejects training splits; a dedicated diagnostic calls
the shared simulator. This does not weaken validation/test contracts.

Both models use full21 observations, fixed unit-sign evaluation exposure,
default costs/accounting, learned protections and the20-session holding limit.
GRU carries inference state within each episode and resets it per symbol.
Normalization stays at its original training-fitted values. This is eligible
fitted history, not an independent holdout or proof that every row was sampled.

## Guards and evidence

A scoped runtime guard rejects SAC `learn/train`, recurrent learner `update`,
backward/autograd, normalization fitting and Adam steps. Loaders validate saved configuration, lineage,
checksums, finite moments and exact counters. Fingerprints before/after inference
cover policy/target tensors, all optimizer moments/groups, raw entropy parameter,
counters and the copied frozen GRU actor. Source file hashes and normalization
must stay unchanged. The guard restores library methods even after failure.

Full derived paths and interactive equity/drawdown/price/protection/fill charts
stay in ignored artifacts. Compact results preserve source hashes, unchanged-state
fingerprints, group/stock excess, matched holding identity and reporting-only
decliner diagnostics. Saved-path participation/cost diagnosis uses an explicit
training-window scope; its default still requires development dates.

The descriptive−2pp stock/ETF and median-stock comparability flags never select
weights or override the failed development gate. A training/development contrast
does not prove overfitting, capacity failure or an optimizer cause: periods,
regimes, duration and execution hardware differ. Source data also retains the
documented free-data/survivorship limitations.

## Execution

Commit verified protocol and implementation before invoking. Use fresh paths:

```bash
MPLCONFIGDIR="$PWD/.runtime/matplotlib" PYTHONPATH=src \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
 scripts/diagnose_offpolicy_fit.py \
 --output results/recurrent-offpolicy-fit-2026-10-31/summary.json \
 --artifacts artifacts/recurrent-offpolicy-fit-2026-10-31
```

Implementation checks passed34 focused tests and Ruff before actual replay.
No model fitting occurred in those checks. Actual saved-policy replay is pending
at this implementation milestone. Project usage remains39.92585919/150; all owned
Colab sessions are stopped. See [fitted-period methodology](training-fit-diagnostic.md)
and [saved-path diagnosis](participation-diagnostics.md) for definitions.
