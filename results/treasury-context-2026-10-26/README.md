# Treasury readiness-matched payloads — accepted, training disabled

Masked/full datasets are built, immutably stored, reloaded and exactly reconstructed.
Independent row/normalizer verification passes. No model is constructed or trained;
this is preprocessing/eligibility acceptance, not a return result. A separate
fixed training experiment remains required. The generic trainer explicitly rejects
Treasury inputs until that registration is implemented.

## Why the training history changes

The [accepted source audit](../treasury-source-2026-10-25/README.md) identifies one
unavailable decision date,2010-10-13. The latest eligible source observation has
null yields; no fallback or imputation is permitted. Fixed
[payload rules](../../configs/experiments/treasury_payload_rules.json) precede actual
preprocessing: remove unready rows, purge the preceding20 rows, partition into
contiguous segments, discard segments below513 bars and require exactly one
remaining segment per original symbol. No segment is selected by price/return.

The existing sampler requires one episode per symbol and a complete512-transition
window. Both payloads preserve all111 training symbols and all26 development
episodes with the same eligibility, episode keys and original symbol order.

| Training row accounting | Rows |
| --- | ---: |
| Original source | 298,946 |
| Unavailable | 110 |
| Preceding20-row purge | 2,200 |
| Short contiguous segments | 13,970 |
| Retained | 282,666 |

The exclusions total16,280 rows and affect2010 only:2010 rows decrease22,330→6,050.
Every2011–2020 row remains. This history change must not be credited to Treasury
information; a future comparison needs both an original legacy control and a
readiness-matched masked control. Both Treasury payloads have225,834 eligible
512-step starts, identical bounds and all four ETF instruments. Group sampling
remains unchanged when a future experiment is registered.

## Immutable datasets and preprocessing

| Arm | Dataset ID | New-column fit rows |
| --- | --- | ---: |
| Masked | `dc6bb7c5cda99dd6fa19c6e8c0bb6e24b27cfc2e18f946ad63bd13c2c6f452ff` | 0 |
| Full | `c6ca10068f8d0607224c3e8252568e4de965663fce5db30ce5d33198c2f5e5d3` | 282,666 |

The original source dataset is unchanged:
`cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b`.
Both payloads contain nine market fields. The original seven raw fields and
names/means/scales remain exact. Only full's two appended Treasury columns fit
retained training observations; masked raw values/means are0 and scales1.
No validation data, clipping or reward/price transformation enters fitting.

All retained bars and ATR values match original half-open row slices exactly.
Every original development key, date, bar, ATR and seven-field vector is preserved:
19,058 development rows. No retired test is reconstructed. Source-rate availability,
maximum ages and current-vintage caveats remain identical to the source audit.
No VIX field or prior learned feature model is reused.

## Acceptance

[Payload record](payloads.json) retains dataset IDs, schemas, complete normalizer
states, source/alignment/eligibility identities, all137 original episode bounds,
exclusion counts and sampler fingerprints. Both datasets reload by content hash
and reconstruct exactly from accepted original data/source. All transformed inputs
are finite float32-compatible; masked extras remain exactly zero. The largest
normalized magnitude is54.2753 in both arms, inherited from the original fields;
no clipping adjustment is introduced.

[Independent verification](payload-verification.json) uses NumPy prefix-sum masks
and segment edges rather than the builder's set/loop implementation. All137
source bounds, every retained original row, all development episodes, exclusions
and eligible-start counts agree. Extra values match original decimal source facts
exactly. Independent NumPy population means/scales agree within1.96e−13 on retained
training rows only. Original seven statistics match exactly. Nine protected
source/rules/code/data/ledger files remain unchanged.

46 focused payload/guard and prior GAE/VIX dispatch tests pass; new modules/script/
tests pass Ruff. Tests include future validation-rate perturbations that cannot
alter training statistics, all-validation readiness, duplicate eligible segments,
source immutability and trainer rejection before model/environment construction.
No local optimizer step or policy initialization occurs.

## Runtime and retention

The local preprocessing worker completed in68.11seconds, within600s; its handle
exited0 and terminal journal is `.runtime/treasury-payload-progress.json`.
No temporary progress file remains. Source originals, receipts, original dataset,
new datasets, rules and original project ledger are retained. Large datasets stay
under ignored `data/datasets/`; provider responses stay under `data/context/treasury/`.
No provider request, Colab runtime, broker order or retired-test access occurred.
Cumulative project usage remains39.351004/150 units; no new model return exists.

## Next acceptance boundary

Register one legacy/masked/full Treasury information comparison with fixed
algorithm/reward/actions/costs, shared original tensors/RNG and zero new input
columns. Require original-control reproduction, exact masked/full sampling,
actual Colab smoke restoration/frozen parity and budget-backed deadline projection
before100k screens. Only passing selected/final stock/ETF/median/drawdown gates
permit seeds11/23/37 and prospective confirmation. Model-research alternatives
such as CrossQ stay separate; this preprocessing result does not register them.


The [three-arm model protocol](../../configs/experiments/treasury_context_preregistration.json)
and [binding](../../configs/experiments/treasury_context_binding.json) are now fixed.
Original/reusable zero-column initializers still need dedicated validation and
runtime integration. This registration does not enable generic training or
allocate Colab; no policy performance result exists.
