# Frozen seed ensembles

This diagnostic tests uniform averaging of existing stateless PPO/SAC seed
predictions while retaining account inputs. It does not train new weights or
allocate Colab. The prior market-only exposure architecture failed its controls;
seed averaging tests model variance without removing cost/position information.

## Fixed members and one account

The protocol fixes PPO seeds 11/23/37, SAC seeds 11/23/37 and three candidates:
PPO-only, SAC-only and all six combined. Each source model received 100k training
transitions. The six-member ensemble therefore reuses 600k historical training
transitions; it is not one 100k model. All weights are uniform and fixed before
replay. Selected and final source weights are evaluated separately.

`inference/ensemble.py` averages all four deterministic bounded action outputs:
exposure, stop distance, near-target distance and zone width. Every member sees
the **same** normalized market/account observation from one evolving simulator.
The averaged exposure becomes a desired total exposure under the usual
whole-share, affordability, cost and next-session gates. Averaging independent
returns would describe a different strategy and is not used.

Input contracts bind feature schema/order, frozen normalization, provider feed,
adjustments, observation profile/size and action schema. Stateless members must
have the normalized four-component action space. Duplicate members, nonfinite or
out-of-range actions, incompatible inputs and stochastic inference are rejected.
Recipes record ordered member identities, weight-file checksums, equal weights
and the complete input contract; restoration rejects any change. No weights or
normalization are updated. Recipes reference existing source models rather than
copying or retraining them.

## Safe evaluation-data bridge

The older source dataset's manifest request extends through 2026. That dataset
is **not opened**. Only source frozen manifests, metadata and saved 2021–2023
validation reports are inspected. Evaluation uses the existing development-only
snapshot ending in 2023, with no test episodes. Every source validation session
and closing price must match; feature schema/names, normalization, feed and state
contract must also match.

Different dataset/provenance identities remain explicit. This does not claim
identical raw training snapshots or silently relabel the old dataset. All
individual reference policies are freshly replayed on the same evaluation data
and local CPU backend used for ensembles. Original cloud-path drift is measured
and reported. Before ensemble performance is evaluated, individual references
must have matching sessions and stay within the declared $50 daily equity,
0.0005 return/drawdown fraction (0.05 percentage point), and $1 cost caps.
Quantity, sign and action differences are reported, not claimed bitwise exact.
These caps are fixed before results and are small relative to the two-point
improvement gate. Any larger discrepancy stops the diagnostic for review.

## Bounded protocol and evidence

[The protocol](../configs/experiments/frozen_ensembles.json) fixes six source
members, three candidates, selected/final phases and 18 labeled replays including
individual references. Identical phase model fingerprints can share an actual
deterministic replay, but labels and original weight hashes remain separate.
All unsuccessful candidates remain in the report. Future negative-holding stock
labels never choose a member, weight, trade or checkpoint.

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/replay_ensembles.py --preflight-only \
  --output .runtime/ensemble-preflight.json

# Commit source/protocol/preflight before the full replay.
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/replay_ensembles.py \
  --output artifacts/frozen-ensembles-2026-10-20
```

The command refuses an existing full-output directory and requires committed
source/protocol. It writes atomic progress, complete compressed paths, source
member catalog, recipes, per-symbol CSV and interactive charts. Source files,
model tensors and normalization are checked unchanged. Candidate comparisons
use mean freshly replayed individual metrics as a statistical reference, not a
claim that independent mean returns are an executable strategy. Diagnostic gates
require two-point selected/final gains, no final stock/ETF deterioration and at
most two points of extra drawdown. Absolute holding comparisons remain separate.

One ensemble built from three seeds is **one** ensemble, not three independent
ensemble trials. A promising development result would still require separately
preregistered robustness repetitions and untouched prospective evaluation. This
diagnostic cannot establish the project goal by itself. No broker orders or
retired-test evaluation are authorized by it.

## Diagnostic continuation after an original reference failure

The original run stopped on final SAC seed11: $3.24 source cost drift exceeded
the $1 cap, while its equity, return and drawdown differences stayed inside
their original caps. That original run remains failed. A separately committed
continuation preserves every member, weight, dataset, performance gate and
original drift limit. It allows only this cost-only discrepancy up to $5 for
descriptive matched-local analysis; the original reference gate remains failed
and promotion remains disabled. No result is relabeled cloud-exact.

`--reuse-verified` reuses only completed cases whose raw report checksums,
recipes, core-package hash and input preflight still match. The failed case is
evaluated again and must satisfy every other original cap. This avoids repeating
verified simulations and does not resume training or Colab sessions. The original
failed report and the continuation are retained separately.

## Completed evidence

[The completed diagnostic](../results/frozen-ensembles-2026-10-20/README.md) reports every candidate and phase. Combined final return is 22.16% versus 30.60% holding; all absolute gates fail. Combined has zero of three declining stocks with both positive total return and positive net short P&L. PPO-only and SAC-only each pass that joint condition on NEE alone. The original reference gate remains failed and nothing is promoted. No additional Colab units were used.
