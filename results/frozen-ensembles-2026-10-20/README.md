# Frozen seed ensembles — prepared

The committed diagnostic fixes PPO seeds 11/23/37, SAC seeds 11/23/37 and uniform
PPO-only, SAC-only and combined action averages. All source members received
100k transitions; no new models or Colab allocations are needed. All policies
receive the same evolving account state; all four bounded outputs are averaged.

Preflight verifies six frozen input contracts and exact source validation dates/
closing prices against the development-only evaluation snapshot. The older
source dataset is not opened because its request extends into retired dates.
Different data/provenance IDs remain explicit. Fresh local individual references
must satisfy the predeclared financial drift caps before ensemble evaluation.
No performance result is claimed yet.

See [protocol](../../configs/experiments/frozen_ensembles.json),
[methodology](../../docs/ensembles.md) and [preflight](preflight.json).

The original run stopped after twelve verified cases on final SAC seed11
reference cost drift: $3.24 versus the $1 cap. Equity/return/drawdown caps pass.
Selected ensemble means were 19.27% PPO, 21.38% SAC and 21.42% combined, below
30.60% holding. A diagnostic-only continuation will retain the original failed
reference gate, reuse verified cases, rerun the failed case and complete the
remaining final cases. It cannot promote a model; all members, weights,
evaluation data and performance gates remain unchanged.
