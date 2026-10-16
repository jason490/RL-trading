# Historical universe coverage

## Why this is separate from adding tickers

The current 26-tradable sample was selected retrospectively from surviving
companies and broad ETFs. Training contains declining price regimes, but that
does not establish historical market representation. Adding today's weakest
stocks or stocks known to never recover introduces another hindsight choice.
Keep the matched indicator data unchanged; a universe change requires a new
manifest and separately declared experiment.

## Required evidence

A historical security record needs a stable issuer/security identifier, dated
ticker mappings, exchange, instrument type, effective listing/delisting dates,
source, retrieval/revision identity, and the timestamp at which each decision
could know membership. Current sector labels cannot be backfilled as historical
classifications. Record universe selection rules and information cutoffs before
observing later returns.

Price data must include each selected member for its full eligible period,
corporate actions and verified delisting/settlement returns. A historical ticker
list alone is insufficient. Missing last prices must not silently become cash,
zero-return exits or zero-value positions. Exclude unavailable securities only
under a declared coverage rule, report every exclusion, and measure how the
exclusions change the sample. Never silently drop failed downloads.

## Provider audit, checked 2026-10-16

[Alpha Vantage's listing utility](https://www.alphavantage.co/documentation/#listing-status)
documents active/delisted US stock and ETF lists for specified dates after
2010-01-01. It requires an API key. This is a candidate source for dated listing
coverage, not evidence that complete delisting-price histories or publication
vintages are available. Audit those separately before using the data.

[Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars) supply
price aggregates; its [assets API](https://alpaca.markets/sdks/python/api_reference/trading/assets.html)
describes available assets. Neither capability alone proves a point-in-time
membership and terminal-return dataset. Verify coverage, entitlement and
corporate-action behavior per endpoint. Current/inactive asset status is not a
replacement for dated membership.

The current Yahoo adapter offers bars and cash dividends, with reconstructed
availability and revised-history limitations. It does not verify historical
membership or delisting settlement returns. No provider is silently substituted
for missing records. File imports remain a valid route for independently verified
historical datasets. Credential environment variables were absent from the
research controller's environment during this audit; do not put keys in Git,
command arguments or chat. Configure provider secrets privately when needed.

## Training-only diagnostics and a later experiment

Use `scripts/audit_training_regimes.py` to inspect trailing training-price
coverage. The current audit found 1,851/63,648 eligible overlapping symbol-sessions
below −20% on the trailing-year measure, and 16/26 symbols with negative trailing
quarter returns while in that decline bucket. These are descriptive price
statistics, not independent events or total-return strategy performance.

Once data pass the evidence gates, declare a moderate expansion before hundreds
of stocks: fixed liquidity/price eligibility using prior information, sector
coverage where historical classifications exist, and a documented missing-data
policy. Match the reward, algorithms/seeds, budget and execution costs to a
control. Report results by trailing regime and symbol, not only one aggregate.
Only training data may determine sampling weights; validation stays chronologically
separate and the inspected test interval remains retired. More samples from the
same episodes do not create independent market evidence.
