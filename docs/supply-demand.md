# Multi-timeframe supply/demand inputs

**Status: D1/W1 implemented and tested; 36-run study and frozen held-out audit complete. See [verification](verification.md) for outcomes and mixed-hardware limitations.**
The requested feature is generic causal supply/demand zones across timeframes.
Daily and completed weekly zones are available as opt-in dataset inputs; completed monthly zones
follow as a separate experiment. Intraday zones require actual intraday history.

## Reference and ownership

The design reference is [tradingAlgo supply_demand.py at b78d578](https://github.com/jason490/tradingAlgo/blob/b78d578c0b03433e7ddcb4f0fe64658cf9eaebae/screener/indicators/supply_demand.py).
That revision was read from the repository. Its detection uses a short sequence
of small-bodied base candles followed by a directional impulse and sufficient
departure from the base. Bounds come from the base's highs/lows. Zone state
changes after formation as later bars revisit or close beyond those bounds.

Implementation ownership is
`src/rl_stocks/features/market_structure/supply_demand.py`; completed-period
aggregation lives in `features/timeframes/aggregation.py`. Default parameters are
recorded in `configs/features/supply_demand.json`; `--zone-config` can load reviewed
parameter overrides. The explicit `--supply-demand d1 w1` flag chooses timeframes. The feature remains provider independent
and consumes the same versioned OHLCV used by the baseline.

## Detection specification

For each symbol and timeframe, scan completed bars chronologically:

1. Compute the previous completed bar's ATR and examine the current completed
   impulse candle. A bullish qualifying impulse proposes demand; a bearish one
   proposes supply.
2. Examine up to three immediately preceding base candles, stopping at the first
   candle that fails the configured body/range test. Require at least one base
   candle and positive zone width.
3. Set `bottom` to the minimum base low and `top` to the maximum base high.
   Demand's proximal edge is `top`, distal edge `bottom`; supply reverses them.
4. Require the impulse close to depart past the proximal edge by the configured
   ATR amount. Activate only after the impulse is publicly available.
5. Update already active zones using completed bars of that same timeframe.
   A distal-edge close invalidates the zone; a proximal-edge revisit marks it
   mitigated. A break takes priority when both conditions occur in one bar.

Reference defaults are hypotheses to record, not proven trading optima:

| Parameter | Reference default |
| --- | --- |
| ATR period | 14 |
| Impulse body / previous ATR threshold | 1.2 |
| Maximum consecutive base candles | 3 |
| Base body / full candle range ceiling | 0.6 |
| Minimum departure / previous ATR | 0.8 |
| Volume spike multiplier | 1.5 |
| Volume-based impulse-threshold multiplier | 0.8 |
| Bullish / bearish RSI thresholds | 60 / 40 |
| RSI-based impulse-threshold multiplier | 0.9 |
| Maximum selected zones per side/timeframe | 10 |
| Overlap merging | Reference enables it; first causal study disables it |

Start with static price-action thresholds, then test the reference volume/RSI
relaxations separately. The reference volume mean includes the current completed
bar; the baseline relative-volume feature uses the prior 20 bars. Both are
causal after close, but they are different definitions. Record the selected
convention rather than silently substituting one for the other.

Use local indicator functions where practical. The reference imports TA-Lib;
adding this family must not automatically make TA-Lib a baseline dependency.
Audit seed/warmup differences against fixed reference fixtures and version any
deliberate differences. Optional reference-parity tooling can use its own extra.

## Timeframes and availability

| Timeframe | Source and aggregation | First eligible observation |
| --- | --- | --- |
| D1 | Canonical daily OHLCV | Daily impulse close plus recorded availability lag |
| W1 | First open, maximum high, minimum low, last close, summed volume across a completed exchange week | First observed daily decision of the next week, matching the existing conservative weekly convention |
| M1 | The equivalent aggregation across a completed calendar month | First observed daily decision of the next month |
| H1 / H4, later | Verified intraday bars with explicit exchange-session boundaries | After each complete source period and its publication lag |

Do not derive intraday candles from daily bars or use a developing week/month.
Handle holidays and early closes through the exchange calendar. Preserve feed,
adjustment, interval, source/revision and retrieval identity. Resampling must use
consistent split-continuous units; convert stored bounds as needed using only
corporate actions already effective at the decision.

Run a separate zone engine for each symbol/timeframe. Weekly/monthly lifecycle
state updates on completed bars of that timeframe; daily-price distance and
inside-zone inputs can change at every daily decision. A finer-timeframe touch
or invalidation rule is a separate, explicitly versioned variant.

## Preventing historical leakage

The reference's full-history zone methods resolve lifecycle through the end of
their input. Applying one final zone list to earlier observations would leak
future invalidations, mitigation and freshness. Instead, produce each historical
snapshot from incremental chronological updates, with emitted snapshot copies compared
against independently processed causal prefixes in the tests.

Keep `formation_event_time` separate from `available_time`. A weekly impulse
ending Friday cannot appear in Thursday's input, and the project's conservative
release convention publishes it at the following week's first daily decision.
Later changes never rewrite previously emitted feature rows. Retain invalidation
events for replay even when a broken zone is absent from the current live set.

Compute freshness, strength, rankings and the top-zone limit from state known at
that time. Resolve ties deterministically. If overlap merging is introduced,
recompute proximal/distal geometry from the merged bounds, retain component IDs
and publish the union only when every contributing zone is available. Never
retroactively widen an earlier zone using a later formation.

## Fixed observation contract

Use prefixes such as `sd_d1_*`, `sd_w1_*` and `sd_m1_*`. The initial compact
proposal includes a readiness mask per timeframe and these fields for the nearest
known live demand and supply zone:

| Input | Definition |
| --- | --- |
| `present` | An eligible zone exists as of the decision |
| `proximal_distance_atr` | Demand: `(close - top) / daily_ATR`; supply: `(bottom - close) / daily_ATR` |
| `width_atr` | `(top - bottom) / daily_ATR` |
| `fresh` | Freshness as of the decision, distinct from absence through `present` |
| `strength_log` | A bounded log transform of currently known formation strength and freshness |
| `age_sessions` | Time since public availability, expressed in daily sessions and explicitly scaled/clipped |

Select nearest by absolute proximal-edge distance; break ties using current
strength, most recent availability and a stable zone ID. Preserve signed distance
when price crosses a higher-timeframe zone before its next lifecycle update.
Optional follow-up inputs include inside-zone flags, mitigated state, zone counts,
retest counts and cross-timeframe overlap/confluence. Introduce them separately
rather than expanding the first experiment indiscriminately.

Represent absence with zero placeholders plus masks; distinguish an unwarmed
timeframe from a ready timeframe with no zone. Freeze clipping, feature order,
timeframe definitions and training-only preprocessing. Normalization must not
change during validation, testing or inference. Do not silently fill gaps or
use later periods to replace unavailable values.

These are OHLCV-derived price-action proxies. They do not identify resting orders
or verified institutional inventory. They are model inputs: the existing learned
stop and take-profit action components remain policy decisions. This addition
does not automatically snap orders to indicator boundaries or introduce broker
execution.

## Acceptance and experiments

Before training, test prefix/future-suffix invariance, completed-period release,
formation/mitigation/break timing, long/short geometry, missing bars, corporate
actions, masked absence, deterministic ranking and batch/incremental parity.
Compare fixed prefix fixtures with the reference under declared seed conventions.
Test save/load schema rejection and frozen prediction parity after changing inputs.

Use separate studies: baseline; D1 zones; W1 zones; D1+W1; then optional M1 and
confluence. Keep the same declared universe, costs, action/reward contract, 50,000-transition
budget, seeds 11/23/37 and validation selection rule identical. Match dates and
warmup coverage; monthly lookbacks must not receive credit for simply shortening
the comparison window. Parameters may be selected only on training/validation.
The 2024-onward test remains reserved until the feature decision is frozen.

Persist zone snapshots/events, schema/parameter hashes and charts showing only
zones actually available at each decision. Include negative studies and extra
runtime/data cost in the cumulative Colab ledger. Adding this family requires a
new dataset and newly trained frozen bundle; the existing baseline bundle is
retained for comparison. Expanded CPU smoke and the full four-input-set study
are complete; D1-versus-W1 isolation remains a separate experiment. The existing baseline
bundle still expects its original seven inputs. Dataset manifests record the
local ATR seed (first 14 true ranges, then Wilder recursion), static thresholds,
clipping and timeframe release convention. This deliberately differs from
TA-Lib's initial seed and is not a claim of exact library parity.

## Building a dataset

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local dataset data/raw/SNAPSHOT_HASH \
  --supply-demand d1 w1 --zone-config configs/features/supply_demand.json
```

D1/W1 add 26 market values to the original seven (33 market inputs, plus five
portfolio values in the environment). Each episode saves as-of selected zone
snapshots and formation/mitigation/break events. Schema identity includes the
parameter hash. Boolean masks/freshness use identity preprocessing; continuous
fields fit retained training rows only. Missing-bar segments reset engines,
and the first week after each history boundary is discarded conservatively.
An engine queried before its latest release rejects the call; replay a prefix
instead of asking current lifecycle state to reconstruct the past.

Use [the universe/sector guide](sectors-universe.md) to combine these inputs
with the expanded stock universe and rotation context. Monthly, intraday,
overlap merging and volume/RSI threshold relaxations remain separate studies.
