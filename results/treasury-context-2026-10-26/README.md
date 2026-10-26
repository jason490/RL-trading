# Treasury context screen — failed

The fixed seed11 three-arm100k RecurrentPPO screen completed in Colab CPU.
Treasury inputs improved the selected checkpoint but deteriorated with further
training and failed the preregistered selected/final gates. This closes the fixed
Treasury feature family; no confirmation, extra steps or maturity/lag/model/reward
rescue is authorized by this result. Reused2021–2023 development data is not an
independent test; retired2024+ data stays unopened.

| Arm | Selected return | Final return |
| --- | ---: | ---: |
| Original legacy | 19.62% | 18.48% |
| Readiness-matched masked | 22.75% | 6.92% |
| Treasury full | 26.51% | −4.56% |
| Matched holding | 30.60% | 30.60% |

Full selects10k; masked20k; legacy80k. Full's selected stock/ETF means are30.13%/
6.58%, versus holding32.63%/19.47%. Final means fall to−3.72%/−9.19%. Final full
lags legacy23.04pp and masked11.48pp, with worse mean drawdown. Two of three
reporting-only declining stocks have both positive policy return and short P&L
at selection; zero of three do at final weights. This is not robust shorting.

## Verification and evidence

- [Fixed summary](summary.json): all checkpoint curves, group/short-side metrics,
  source identities, model restoration and exact gate decisions.
- [Independent verification](verification.json):156 selected/final symbol reports
  reaggregated with maximum error5.56e−17;179 protected files unchanged.
- [Learned smokes](smokes.json): three512-transition runs;234 exact frozen
  predictions. Full screens add234, totaling468 across six accepted runs.
- Legacy reproduces all11 historical checkpoint scores and both26-symbol reports
  exactly. Full/masked initial tensors/RNG, observed date/window paths, holding
  prices, normalized contracts and execution assumptions pass.
- [First failure](smoke-failure.json) is retained: post-training audit snapshot
  included an inference reset; reviewed fix precedes all accepted runs.

Clean training source55044a5; default Colab Python3.13.16 verified before setup.
All seven owned allocations, including the first failure, are stopped. Conservative
project usage39.47256169564856/150, remaining110.52743830435145; complete Treasury
study0.12155722796917 units including setup, idle, downloads, failure and retry.
Official account348.64, rate0, no active assignments. Original project150/130,
account150/170 and two-unit study baseline remain intact.

All raw/source/dataset and retrieved model evidence is retained. The standalone
interactive comparison is prepared at
`artifacts/public/treasury-context-2026-10-26/treasury-comparison.html`.
[Research prerelease](https://github.com/jason490/RL-trading/releases/tag/treasury-context-2026-10-26)
is published; all14 asset sizes and server SHA256 digests match local copies.
See [release verification](release-verification.json). Cleanup removed13 exact
duplicates/393.94MiB while retaining six differing snapshots and the failed
attempt. [Cleanup inventory](cleanup.json) rehashes938 protected files unchanged;
[post-cleanup verification](post-cleanup-verification.json) reruns strict reports
and restoration with an identical canonical summary.
The next distinct model proposal is CrossQ feasibility against a matched SAC
control, separately preregistered; it has not been enabled or trained here.
No robust/prospective winner or paper eligibility is claimed. Historical Treasury
publication vintages and historical universe/delisting coverage remain unverified.

## Preprocessing evidence completed before training

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
information; the completed comparison therefore includes both an original legacy control and a
readiness-matched masked control. Both Treasury payloads have225,834 eligible
512-step starts, identical bounds and all four ETF instruments. Group sampling
remains unchanged in the registered model experiment.

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

## Preprocessing runtime and retention

The local preprocessing worker completed in68.11seconds, within600s; its handle
exited0 and terminal journal is `.runtime/treasury-payload-progress.json`.
No temporary progress file remains. Source originals, receipts, original dataset,
new datasets, rules and original project ledger are retained. Large datasets stay
under ignored `data/datasets/`; provider responses stay under `data/context/treasury/`.
This preprocessing stage made no provider request, Colab allocation, broker order
or retired-test access; subsequent model usage and results are recorded above.


[Asset/data contracts](contracts.json), [payload record](payloads.json) and
[payload verification](payload-verification.json) preserve the independent
preprocessing and untrained21/23-input initializer acceptance. Manual Adam
serialization fixtures alone were not learned acceptance; actual Colab smokes
and full saved-model verification above supply that evidence.
