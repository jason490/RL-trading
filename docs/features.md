# Causal feature families

The default schema is `daily-indicators-v1`; adding completed weekly context produces `daily-indicators-v1-weekly`. The ordered names are part of the bundle contract. Features use the completed daily bar, available under its source's recorded availability assumption. Neither centered windows nor future bar values are used.

| Feature | Implementation | Formula / scaling | Warmup |
| --- | --- | --- | --- |
| `return_1` | `features/price/indicators.py` | log(close_t / close_t-1) | One earlier close |
| `volatility_20` | Same | Population standard deviation of the latest 20 log returns | 20 returns |
| `ema_20_ratio` | `features/trend/ema.py` | close / EMA20 - 1; alpha=2/21 | 20 bars |
| `ema_50_ratio` | Same | close / EMA50 - 1; alpha=2/51 | 50 bars |
| `rsi_14` | `features/momentum/rsi.py` | Wilder gain/loss smoothing; RSI / 100 | 14 price differences |
| `atr_14_ratio` | `features/price/indicators.py` | Wilder true range / close | 14 bars |
| `relative_volume_20` | `features/volume/relative.py` | current completed volume / mean of **prior** 20 volumes | 20 prior volumes |
| Optional `completed_weekly_ema10_ratio` | `features/timeframes/weekly.py` | completed weekly close / weekly EMA10 - 1 | Ten released completed weeks |

EMA starts from the first causal close and masks values until its stated window; it is not seeded from a future-window mean. RSI initializes from the first 14 gains/losses, then applies Wilder smoothing. A flat price series has RSI 50; positive gains with zero losses give 100. ATR uses max(high-low, abs(high-previous close), abs(low-previous close)), initializes with the first 14 ranges and applies Wilder smoothing thereafter. These explicit conventions can differ from library implementations; they are versioned assumptions.

## Splits and price units

For raw prices, a cumulative split factor is built forward using only effective splits observed by that row. OHLC are multiplied by that factor and volume divided by it for split-continuous indicators. No future split factor is needed. Ratios/returns remain dimensionless. ATR is divided back by the current factor before the simulator decodes price levels. Already split-adjusted Yahoo/Alpaca series carry split=1 so the adjustment is not applied twice.

Dividends remain cash accounting events; price-return features are explicitly price returns rather than total-return labels. Corporate-action source quality remains a data gate.

## Weekly availability

Weekly grouping uses ISO calendar weeks and the last observed session close in each group. A week is released at the first observed session of the next week. This is deliberately conservative: Friday's completed week is not yet visible in Friday's daily observation. Holidays do not require inventing a Friday bar. The final incomplete group is never released. No weekly feature is backfilled into earlier days.

Feature rows with missing warmup/nonfinite results are excluded, not arbitrarily filled. Verified calendar gaps restart all feature-family warmup before another episode can begin.

## Optional research families

[Multi-timeframe supply/demand zones](supply-demand.md) are an implemented opt-in
market-structure family based on causal base/impulse formation and as-of
lifecycle state. Study D1 and completed W1 inputs first, then M1 separately.
The current seven-indicator model has no supply/demand inputs; enabling this
family requires a new feature schema, dataset, ablation and retrained bundle.

[Sector rotation](sectors-universe.md) is implemented as opt-in historical ETF context; issuer labels require audited training-era classifications. Filings, options/OI, sentiment and order flow remain disabled. Add each family independently with event/public-availability/retrieval/revision metadata and backward as-of joins. A current snapshot is not historical evidence. Estimated signed trades or bar-derived volume profiles must be labelled as proxies. Match data windows, costs, seeds and budgets when evaluating each feature ablation.

Offline tests compare full-series outputs with causal prefixes and altered future suffixes, verify current-volume exclusion, weekly prefix parity and forward split continuity. Incremental production feature state is a future shadow-inference extension; current inference should recompute the same batch feature functions on available history.

[Completed-session decomposition](session-context.md) describes optional causal
overnight/intraday fields, same-dimension masks and pending research preflight.

[Compact broad-market context](market-context.md) implements an opt-in feature
reader for synchronized SPY20-session price return/volatility and own-relative
return. Cached111/26 source coverage passes for every row; dataset integration,
training-only extra-field normalization and matched model ablation remain pending.
This coverage audit is not evidence of predictive or trading improvement.

[Financial filing snapshots](filing-data.md) now have an optional source-only
arithmetic builder for coherent-accession revenue growth, net margin and
liabilities/assets. Exact fiscal intervals, duplicate/concept agreement and
original comparatives are required. It remains outside the default registry:
arithmetic readiness is not publication/identity/price-link or dataset readiness.
No financial family has yet trained or passed model acceptance.
