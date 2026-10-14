# Shared contracts and schema reference

`core/contracts.py` owns frozen data contracts, typed provider capability declarations, canonical JSON and SHA-256 identities. Schema/name order travels with datasets and model bundles; incompatible changes require a new version and retraining.

## Provider contract

`BarProvider` is a protocol with `capabilities: Capabilities` and `bars(request: Request) -> Sequence[Bar]`. SDK-specific objects never pass into environments. Network operations occur only when `bars()` is called. `Request` contains an immutable symbol tuple and inclusive start/exclusive end date; duplicate/empty symbols and reversed dates are rejected.

`Capabilities` declares provider name, supported `RecordType` values, supported intervals, earliest declared date, feed, adjustment mode and limitations. `require(record, interval)` fails explicitly if unsupported. Record types enumerate bars, trades, quotes, sectors, filings, options and news independently. Current adapters implement bars only; an enum declaration is not a promise of implemented retrieval for the other families.

## Bar

| Field | Type | Contract |
| --- | --- | --- |
| `symbol` | str | Declared instrument identity |
| `session` | str | ISO exchange-session date |
| `open_time` | str | UTC execution-open timestamp |
| `event_time` | str | UTC completed-bar timestamp |
| `available_time` | str | UTC earliest allowed decision availability |
| `retrieved_time` | str | UTC snapshot acquisition time |
| `provider`, `feed` | str | Source/feed identity; mixed sources require explicit harmonization |
| `revision` | str | Snapshot/revision identity supplied by the adapter |
| `open`, `high`, `low`, `close` | float | Positive finite executable prices in one declared adjustment convention |
| `volume` | float | Finite nonnegative source volume |
| `dividend` | float | Effective cash dividend per share; zero when none, source verification required |
| `split` | float | Positive effective new-shares/old-shares ratio; 1 for already split-adjusted prices |
| `adjustments` | str | `raw_with_actions` or `split_adjusted_with_dividends` |

Validation requires timezone-aware UTC, open < event <= availability, retrieval >= event, unique increasing sessions per symbol and valid OHLC geometry. Availability can be reconstructed by a provider; its uncertainty is recorded in capability/provenance limitations. The record does not certify historical vintages by itself.

## Features and preprocessing

Default schema: `daily-indicators-v1`. Optional weekly schema: `daily-indicators-v1-weekly`. Ordered default names: `return_1`, `volatility_20`, `ema_20_ratio`, `ema_50_ratio`, `rsi_14`, `atr_14_ratio`, `relative_volume_20`. Feature rows retain source index, finite numeric values, raw-price-unit ATR and availability timestamp.

`Normalizer` has ordered names, means and positive scales as immutable tuples. Fitting uses retained training rows only. Transform accepts precisely the same names/order/count and finite values. [Feature formulas](features.md) and [dataset construction](datasets.md) own the computation/timing details.

## Decision

Action schema: `target-zone-v1`. `Decision` contains schema, symbol, decision/expiry timestamps, reference price, raw/clipped four-component vectors, clipped exposure, signed desired whole-share quantity, stop/near/far, direction integer (-1/0/+1), intent, clipping flag, absolute notional, ordered zone lower/upper and side (`short`/`flat`/`long`).

Intent is `hold`, `close`, `reverse`, `add` or `reduce`, based on desired versus current signed quantity. It is a research proposal. The next-open gate separately reports approved target quantity and any risk clipping/rejection; actual quantity after protective exits can differ. [Action/risk decoding](actions-risk.md) owns the formulas and constraints.

## Environment diagnostics

Step info includes marked equity, cash, quantity, age, exposure, cumulative costs/turnover, step borrow cost, previous equity, raw net return, raw action and decoded decision (null if protective geometry is invalid), active protection tuple, voluntary approved target quantity, risk clipping flag, rejection reason, session/price and current-step fill events. Fill events include delta/current quantity, reference/fill price, fee/impact, closed quantity, realized exit P&L, holding sessions and marked equity.

`reset` returns `(observation, info)`; `step` returns `(observation, reward, terminated, truncated, info)`. [Environment documentation](environment.md) defines observation order and boundary semantics. The SB3 VecEnv interface remains separate.

## Serialized artifacts

Snapshots and datasets use sorted compact canonical JSON, finite numbers only, and content-addressed directories. Runs use unique timestamp/algorithm/seed/random-ID directories. Bundles use manifest identities plus per-file checksums. Paths, secrets and authenticated HTTP outputs are not model features or metadata. [Storage](storage.md), [training](training.md) and [inference](inference.md) document lifecycle and recovery separately.
