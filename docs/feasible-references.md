# Fixed feasible-reference diagnosis

Before another learned model, characterize three unconditional reference requests
on the same fitted history as the accepted GRU/SAC diagnosis. This is a bounded
simulation audit, not learning, an optimizer/capacity test, a promotion experiment
or an upper bound on attainable returns.

## Fixed contract

[Protocol](../configs/experiments/feasible_references.json), canonical SHA256
`67248a9cfa7c99852bdf4a88fe6d7c7584ab14a1ef3b6537b20df20d19d21051`.
It pins the immutable cdb93a33 dataset, original normalization, prior fitted
summary and exact matched holding-path identity.

| Reference | Requested action: exposure, stop, near, width |
| --- | --- |
| Long | `[1, 1, 1, 1]` |
| Short | `[-1, 1, 1, 1]` |
| Cash | `[0, 1, 1, 1]` |

Every symbol/session gets the same request for each reference, independent of
observations or outcomes. There is no sweep, threshold fitting or future-label
direction choice. The three references are all retained, including losses.
They are manually specified diagnostic references, not replacements for learned
take-profit zones in any trained policy.

The existing decoder maps the protection components to5 ATR stop,8 ATR near
target and4 ATR width. These are active boundary values, not disabled protective
orders. Entries can still be rejected/clipped; partial targets, gap ordering,
fees, adverse impact, borrow, whole shares,100% cap and20-session holding exits
remain unchanged. A long/short request does not guarantee continuous inventory.
Cash must have no fills, costs or net return.

## Window and verification

Use only the original22 stocks and SPY/QQQ/IWM/DIA, decision anchor2017-12-29,
executions2018-01-02–2020-12-02. Preserve explicit training labels and full21
observation schema. Normalization remains training-fitted; there is no new fit.
The unchanged standard evaluator still rejects training scope. Development and
retired-test data are not replayed.

Verify source/configuration/data/normalizer, unchanged closure and idle shared
ledger; identical holding paths to the fitted-policy audit; constant proposals
on every row; and cost/long+short/equity reconciliation. The existing runtime
guard rejects fitting, gradients, Adam steps and normalization fitting. No learned
model is loaded, no provider is requested and no Colab is allocated. Source and
ledger hashes must remain unchanged.

Keep full paths and charts in ignored artifacts and compact group/symbol,
participation/cost and reporting-only decline results in tracked evidence.
Observed gaps do not prove other feasible policies cannot outperform holding.
Neither direction is selected from the negative-holding cohort or promoted from
these fitted results. The failed recurrent family stays closed.

## Command

Commit verified implementation and protocol before actual replay; use fresh paths:

```bash
MPLCONFIGDIR="$PWD/.runtime/matplotlib" PYTHONPATH=src \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
 scripts/diagnose_feasible_references.py \
 --output results/feasible-references-2026-10-31/summary.json \
 --artifacts artifacts/feasible-references-2026-10-31
```

35 focused checks and Ruff pass. They cover unconditional fresh action arrays, long/short ATR
geometry, explicit cash accounting, retained train dates and actual20-session
holding exits, with the existing no-fitting/split/participation regressions.
Actual replay completed after committed source; no new learning or allocation.

## Actual outcome

All78 episodes/57,408 rows complete. Long mean43.28%, short−45.60%, cash0%,
holding56.45%. Long stock/ETF returns43.85/40.14% versus holding58.15/47.15%.
No source, ledger, normalization, closure or rule changed; every request remained
unconditional and matched holding paths are identical to the fitted-policy audit.
Fixed short profits on CVX but loses on XOM despite both negative holding returns;
the cohort remains reporting-only. The observed gaps do not establish an optimal
feasibility bound or justify relaxing constraints. Read the
[complete evidence](../results/feasible-references-2026-10-31/README.md).
