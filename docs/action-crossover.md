# Read-only recurrent action-component crossover

Status: fixed protocol and verified implementation prepared; actual replay pending.
This is diagnostic inference, not a trained/deployable hybrid or rescue of the
closed L2 family. No Colab allocation, fitting, orders or retired-test access.

## Question and fixed controls

The [saved L2 audit](../results/l2-actions-2026-10-23/README.md) shows stock return
34.00→19.27% despite similar91.3% closing long exposure and lower costs. Aggregate
participation does not reveal protective timing or conditional direction effects.
Replay all four component sources from the same preexisting selected50k/final100k
L2 policy weights:

| Case | Exposure output source | Three protective output source |
| --- | --- | --- |
| selected_selected |Selected|Selected|
| final_final |Final|Final|
| selected_final |Selected|Final|
| final_selected |Final|Selected|

[Fixed numeric protocol](../configs/experiments/l2_crossover.json), canonical
SHA-256 `4b33f924813e9dd0fe337477b9ca80925ea4afe4e6cf53ba3f0adf6b1ece3e1f`.
Same original26 development symbols, unchanged immutable dataset and normalizer,
21 masked observations, unit-sign exposure and default simulation/costs/20-session
hold. No new checkpoint selection, hybrid selection/promotion, coefficient/threshold
grid or failed-family extension. Report every case, including failures.

## State and source integrity

`inference/crossover.py` owns component combination and independent carried
hidden/cell states. Both models receive copies of the same current hybrid-account
observation; source-specific states reset per symbol. Never exchange states
between weights or reuse a prior symbol's memory. The wrapper is deterministic
and exposes no training method. It does not alter raw outputs before the existing
unit-exposure filter; all three protective outputs come from the declared source.

Both native diagonals must reproduce accepted full saved paths before mixed
cases run. Exact keys/labels/dates/reasons/integer quantities and row/event counts;
finite numeric floats use abs1e-5 plus rel1e-9 tolerance. This includes actions,
protective prices, daily equity/costs, metrics, events and holding paths. Failure
rejects interpretation; do not widen tolerances after observing actual results.

The thin `scripts/diagnose_l2_crossover.py` requires committed source/protocol,
accepted retained metadata/models/report hashes, exact loaded optimizer/model
records and unchanged normalizer. All four holding paths must match. Full tensor/
Adam/source fingerprints remain unchanged after inference. Original raw data,
models and prior reports retained; derived paths outside Git, compact proof tracked.

## Interpretation and execution

This is a counterfactual policy-component experiment, not additive attribution.
Changing stops/targets changes account state and subsequent outputs from both
models. Different memory trajectories remain part of each counterfactual. A
mixed case's return cannot establish independent evidence or revive the L2 model.
Future declining-stock labels are reporting-only; no cohort-based action routing.

40 focused crossover/fit/action tests pass, including all four source contracts,
independent states, reset and caller-input immutability, invalid observations/
states/stochastic or unbounded outputs, exact-count/label and numeric-tolerance
rejection. Source compiles. Commit before actual replay:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python \
 scripts/diagnose_l2_crossover.py \
 --output results/l2-crossover-2026-10-23/summary.json \
 --artifacts artifacts/l2-crossover-2026-10-23
```

Project38.74617914/150 unchanged; last official account349.23 zero rate/assignments.
All prior owned sessions stopped. No live trading or new training from this recipe.
