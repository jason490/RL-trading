# Dataset construction and split isolation

`generate(snapshot, output, weekly=False, purge=20, allow_unverified=False)` reads a verified immutable snapshot, computes features, builds chronological episodes, fits a training-only normalizer and writes a content-addressed dataset. No training-time provider calls occur.

## Quality gate

The default gate requires verified exchange-calendar coverage, no missing sessions, and verified dividend/action treatment. Synthetic weekday data is intentionally unverified. Alpaca bar-only snapshots are intentionally missing dividend verification. A user may make an explicit research override with `--allow-unverified`; the manifest records it, and it does not turn the data into audited market history.

Calendar coverage includes sessions on or after the requested start and strictly
before the end. Holiday and weekend endpoints are valid; they do not need to be
exchange sessions themselves.

Every symbol must have training, validation and test episodes with at least two retained feature rows. Missing verified sessions break the series and restart feature warmup; no one-day return is inferred across an unobserved session. Unknown calendar quality is visible in provenance. Mixed providers/feeds/adjustment modes are rejected before generation.

## Chronological partitions

| Partition | Included session dates | Purpose |
| --- | --- | --- |
| Training | Through 2020-12-31 | Policy fitting and preprocessing fit |
| Validation | 2021-01-01 through 2023-12-31 | Checkpoint selection and development |
| Test | 2024-01-01 through requested September 2026 endpoint | Audit after freezing |

The last **20 observed sessions** of each training/validation partition are purged, matching the maximum supported holding horizon. Values below 20 are rejected. No position is carried from one partition into another. Dataset episodes are separate contiguous feature segments; training may sample a start within a training segment, but it cannot shuffle rows across partition boundaries.

Indicators at the first validation/test sessions may use already available historical warmup bars from the preceding partition. This is causal history, not a fitted validation/test statistic. Each row retains the source bar's public-availability assumption; the simulator also checks that a decision's completed bar is available before the next execution open.

## Normalizer

`Normalizer` is a frozen dataclass containing ordered feature names, training means and population standard deviations. Near-zero scales have a floor of 1e-8. Fit rejects nonfinite/ragged matrices. Transform rejects name/order/length mismatch and nonfinite values; it has no update or partial-fit path. The same saved normalizer is used during validation, test and inference.

Market features are normalized. Portfolio-state components remain raw scaled ratios and do not affect normalizer fitting. Observation clipping is not used to hide invalid or extreme data.

## Dataset manifest

`dataset.json` stores feature schema/name order, normalizer, symbol/split episodes, ATR in executable units and a manifest with snapshot identity, source provenance/coverage, exact split boundaries, purge length, universe, weekly flag, override status and survivorship caveat. Loading checks the directory hash against exact canonical bytes.

The fixed ETF universe was selected retrospectively. Evaluations on these four liquid ETFs are not evidence about arbitrary current stock constituents or historical universe survival. Retain unsuccessful experiments and freeze research choices before opening the final test. Changing a feature schema or provider/feed requires a new dataset identity and new inference bundle.
