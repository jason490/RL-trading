# Fixed-policy direction replay

A positive short contribution on an unrestricted trading path does not prove
that a short-only strategy would make that profit. Removing longs changes cash,
equity, subsequent whole-share sizing, holding age, protections and the inputs
seen by the frozen policy. This diagnostic replays those paths instead of
subtracting accounting contributions.

## Universal filters

`evaluate(..., exposure_mode='unrestricted'|'long_only'|'short_only')` starts a
fresh simulator for each validation episode. The default preserves the existing
path. Long-only replaces negative target exposure with zero; short-only replaces
positive target exposure with zero. The original stop, near-target and zone-width
outputs remain unchanged. A zero target closes the position under the existing
next-session execution rules; it does not invert a trade or promise an immediate
fill before active protection. Costs, borrow, dividends, gaps, stops, partial
exits, corporate actions and the holding limit keep their usual semantics.

Every new observation is constructed from the restricted path's actual account
and execution state. The model's normalization and parameters are frozen.
Filtered rows record `policy_action` before restriction and `raw_action` after
restriction. Summary statistics separate requested long/short fractions,
suppressed signals and executed end-of-session positions. These fractions are
session-weighted, while group returns use equal-weight symbol episodes.

Filters apply to **every symbol and session**. Future negative-holding labels
never select a filter, stock, checkpoint or reward. The declining-stock cohort
is reported afterward using the same positive-total-return plus positive-short-
P&L requirement as [short-side.md](short-side.md).

## Bounded protocol

[The fixed protocol](../configs/experiments/direction_replay.json) includes all
three 50k seed-11 PPO observation profiles and their selected/final weights,
three universal modes and the original 26-symbol 2021–2023 development panel:
18 labeled cells. No model is fitted, no checkpoint reselected and no Colab
runtime allocated. Local CPU inference and artifact checks are permitted;
research training continues to require Colab.

Before any replay, the command requires committed source/protocol, complete
original training with verified cleanup, matching data/configuration, valid
frozen checksums and original selected weights. Unrestricted paths must match
the original report's sessions/quantities exactly, daily equity within $0.10,
costs within $0.01, action components within 0.000002 and return/drawdown within
0.000001. Exact report equality is recorded separately. These small declared CPU
backend tolerances are not permissions to substitute materially different paths.

Selected and final evaluations may be reused **within the same run only** when
full policy tensors, policy class and input/output spaces have identical
fingerprints. Phase and original weight-file hashes remain explicit; both
unrestricted references are independently checked. This avoids repeating an
identical deterministic simulation while preserving selected/final provenance.

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/replay_directions.py \
  --state .runtime/observation-screen-state.json \
  --output artifacts/direction-replay
```

The command refuses an existing output directory. It writes incremental status,
compact summary JSON, one compressed complete path per unique replay,
per-symbol CSV and an interactive comparison. Original bundles, weight files,
normalization, source reports and dataset are checked unchanged. Model tensors
are also checked after inference. A failure retains its partial evidence and
requires review; it never silently starts training or restarts a Colab session.

## Interpretation

All modes, profiles, phases and unsuccessful stocks remain in the report. A
short-only mode may help falling stocks while losing on stocks or ETFs overall.
That is evidence about specialization, not an all-market replacement policy.
A long-only improvement does not establish profitable shorting. A causal gate or
separate direction specialist would need a new bounded training protocol and
multi-seed development checks, followed by untouched prospective evaluation.
Repeated 2021–2023 diagnostics do not create independent evidence; the retired
2024–September 2026 interval remains isolated.

## Existing confirmation seeds

After the seed-11 diagnostic, separate committed seed-23/37 protocols repeat all
three universal modes on the already-trained wide-initialization **legacy**
models. These protocols use a one-profile, six-cell subset with no new training.
Only the declared initialization arm/seed is taken from the complete original
confirmation state; other preexisting study cells are not inputs. Old bundles
without observation metadata retain the legacy contract, and pre-sampler source
metadata means the original legacy-uniform starts. Original unrestricted paths
must independently pass parity for each phase. Seed 11's exact training-tensor
bridge to the old source is retained in the prior study results.

Report all seeds and both phases, including failures. These replays confirm
diagnostic consistency, not a newly trained or prospectively validated strategy.

## Reference drift inspection

`--reference-only` runs just the unrestricted selected/final paths and records
maximum action, equity, cost and position differences against saved reports.
It writes status `calibrated`, never `complete`, and never runs filtered cases or
relaxes parity thresholds. Whole-share boundary differences can be diagnosed
without silently accepting them. Failed ordinary replay retains the complete
compressed failed path and discrepancy summary for review. A changed acceptance
protocol requires a separate explicit declaration before filtered evaluation.
