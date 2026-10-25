# Historical implied-volatility context

This optional information family is **disabled in model inputs**. The first
source/coverage audit is prepared; no download, trained-model return or feature
benefit is established by implementation tests. It is distinct from the failed
SPY realized-price market-context and SAC/TQC method studies.

## Source and method

[Cboe historical-data documentation](https://www.cboe.com/tradable_products/vix/vix_historical_data)
publishes daily VIX closing history and distinguishes the SPX-based2003
methodology from the older OEX/VXO series. VIX is broad option-implied volatility,
not an individual stock's option surface, OI/dealer inventory or realized price
volatility. This audit starts2009, after the2003 change; later methodology and
revision/publication limitations must remain recorded.

The installed optional yfinance adapter requests only `^VIX`, inclusive2009-01-01
and exclusive2024-01-01. Raw current-vintage close/date records are stored in a
content-addressed immutable snapshot under `data/context/vix/`, separately from
executable equity bars. No trades, dividends, quantities, stock returns or broker
orders derive from these index records. A single top-level history call has a
30-second request timeout, with180-second worker bound, no feed fallback and
no automatic restart. Library metadata/session HTTP calls are not counted as
additional research-history requests. Original provider values stay local.

FRED's [official observations API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
has documented observation bounds and requires a registered key. Its public
graph CSV filtering is not treated as a documented date-bound guarantee here.
The full current Cboe CSV is not downloaded, preserving retired-price isolation.
No new key, dependency installation or Colab allocation is required for this
Yahoo research audit; unofficial personal-use/revised-history limitations apply.

## Lag and coverage contract

Only two fixed coordinates are prepared: log previous-session VIX close and
log(previous close /21st-prior-session close), a20-XNYS-session change. Dates
refer to exact XNYS sessions; missing exact endpoints yield uncovered decisions,
never stale filling. An assumed availability timestamp is the following XNYS
session's close plus15minutes. Actual UTC decision timestamps must reach that
availability before context is exposed. Same/future-session source values cannot
affect a prior decision. Early closes/calendar dates use exchange-calendars.

This is a deliberately conservative **availability reconstruction**, not proof
of historical publication timestamps or original vintages. A lag cannot repair
a later historical revision. The source snapshot records retrieval/revision
identity and flags historical publication vintages as unverified. Any later
research must explicitly retain that limitation and require prospective evidence.

The fixed protocol `configs/experiments/vix_source_audit.json` compares readiness
with the immutable original26-symbol train/development-only dataset. Coverage
must reach99.5% overall and in2010–14/2015–20/2021–23 eras, and99% each calendar
year. No source2024+ observations, stock-return scoring, normalization fitting,
training or data-family activation. Passing coverage is only feasibility under
the stated assumption; a separate matched experiment/protocol and acceptance are
needed before model inputs change. Failure does not loosen these thresholds.

Four offline tests validate exact lag/lookback indices, no stale filling,
same-day future perturbation isolation, UTC availability gating, invalid source
schemas and pre-network request bounds. The default pipeline remains unchanged.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
.venv/bin/python scripts/audit_vix_source.py
```

The journal `.runtime/vix-source-progress.json` preserves complete/failure status
and source recovery identity. An existing journal requires review; no implicit
retry or provider substitution. Actual audit evidence belongs in
`results/vix-source-2026-10-25/`.
