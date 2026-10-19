# Dataset construction and split isolation

`generate(...)` reads a verified immutable snapshot, computes features, builds chronological episodes, fits a training-only normalizer and writes a content-addressed dataset. No training-time provider calls occur. Optional `evaluation_symbols`, `include_test` and `coverage_scope` arguments control the declared evaluation panel and coverage gate without changing the defaults.

## Quality gate

The default gate requires verified exchange-calendar coverage, no missing sessions, and verified dividend/action treatment. Synthetic weekday data is intentionally unverified. Alpaca bar-only snapshots are intentionally missing dividend verification. A user may make an explicit research override with `--allow-unverified`; the manifest records it, and it does not turn the data into audited market history.

The default `coverage_scope='snapshot'` checks every source coverage record,
including failed requests outside a smaller configured universe. An explicit
`coverage_scope='universe'` (CLI `--coverage-scope universe`) instead checks
**every configured tradable and context symbol**. It requires an explicit
universe, rejects absent bars/coverage records and retains dividend verification.
It does not enable `allow_unverified`, impute gaps or merge feeds. The manifest
records checked symbols and excluded source failures in `coverage_audit` while
preserving the original complete provenance. This allows an explicitly declared
complete-case subset without pretending the original wider request succeeded.

Calendar coverage includes sessions on or after the requested start and strictly
before the end. Holiday and weekend endpoints are valid; they do not need to be
exchange sessions themselves.

Every configured tradable symbol must have training, validation and test episodes with at least two retained feature rows. Missing verified sessions break the series and restart feature warmup; no one-day return is inferred across an unobserved session. Unknown calendar quality is visible in provenance. Mixed providers/feeds/adjustment modes are rejected before generation.

## Chronological partitions

| Partition | Included session dates | Purpose |
| --- | --- | --- |
| Training | Through 2020-12-31 | Policy fitting and preprocessing fit |
| Validation | 2021-01-01 through 2023-12-31 | Checkpoint selection and development |
| Test | 2024-01-01 through requested September 2026 endpoint | Audit after freezing |

The last **20 observed sessions** of each training/validation partition are purged, matching the maximum supported holding horizon. Values below 20 are rejected. No position is carried from one partition into another. Dataset episodes are separate contiguous feature segments; training may sample a start within a training segment, but it cannot shuffle rows across partition boundaries.

Indicators at the first validation/test sessions may use already available historical warmup bars from the preceding partition. This is causal history, not a fitted validation/test statistic. Each row retains the source bar's public-availability assumption; the simulator also checks that a decision's completed bar is available before the next execution open.

## Normalizer

For a declared training-panel comparison, `evaluation_symbols` restricts validation
and test episodes to a nonempty unique subset of configured tradables.
`include_test=False` creates only training and validation episodes. These choices
are recorded in the manifest and leave the default all-symbol, three-split behavior
unchanged. All training symbols still need valid training episodes; every scored
symbol needs each requested evaluation split. See the
[universe protocol](universe-expansion.md) for the current use of these options.
The CLI exposes these as `--evaluation-symbols SYMBOL ...` and
`--development-only`. The latter excludes test episodes entirely. No omitted
test partition can later be treated as a fresh holdout after it was retired.

`Normalizer` is a frozen dataclass containing ordered feature names, training means and population standard deviations. Continuous near-zero scales have a floor of 1e-8. Optional readiness/presence/freshness masks and issuer one-hot fields use fixed mean zero/scale one to preserve their binary contract. Fit rejects nonfinite/ragged matrices. Transform rejects name/order/length mismatch and nonfinite values; it has no update or partial-fit path. The same saved normalizer is used during validation, test and inference.

Market features are normalized. Portfolio-state components remain raw scaled ratios and do not affect normalizer fitting. Observation clipping is not used to hide invalid or extreme data.

## Dataset manifest

`dataset.json` stores feature schema/name order, normalizer, symbol/split episodes, ATR in executable units and a manifest with snapshot identity, source provenance/coverage, exact split boundaries, purge length, universe, weekly flag, override status and survivorship caveat. Loading checks the directory hash against exact canonical bytes.

[Configured universes](sectors-universe.md) distinguish tradable symbols from context ETFs and retain dated sector metadata in the manifest. The default fixed ETF universe was selected retrospectively. Evaluations on these four liquid ETFs are not evidence about arbitrary current stock constituents or historical universe survival. Retain unsuccessful experiments and freeze research choices before opening the final test. Changing a feature schema or provider/feed requires a new dataset identity and new inference bundle.

## Historical-universe evidence is a separate gate

Use [historical import and audit](historical-data.md) to verify supplied stable
security IDs, known-by-date membership, missing/dead-stock prices and terminal
outcomes. Its eligible-session CSV is a data-request panel, not a training dataset.
The existing generator still creates fixed-universe episodes and requires each
symbol in all three splits; it must not be used unchanged for IPO/delisted
cohorts. `training_ready` intentionally remains false pending historical episode,
membership-exit/suspension and terminal-aware policy/benchmark accounting. No
future-failure or recovery label may select samples or weights. Keep previous
models and the retired final-test interval separate from a new expansion study.
