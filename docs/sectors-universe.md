# Configurable stock universe and sector rotation

## Wider data collection: 110 stocks

[stock_research_110.json](../configs/universes/stock_research_110.json) retains
the prior 55 stocks and adds 55, with ten names per declared current sector,
four broad ETFs and eleven sector ETFs. This configuration is for data collection;
it does not replace any previous experiment's universe or historical membership.

The October 19 pull acquired 2010–2023 Yahoo histories for 107 stocks plus the
15 ETFs, and separate 2016–2023 Alpaca SIP histories for EA, AVB and EQR, whose
Yahoo requests failed. All requested names remain in coverage reports. The
source snapshots are not harmonized or training-ready; Alpaca cash dividends
remain unverified. See [counts, snapshot identities, commands and tests](../results/stock-pull-2026-10-19/README.md).

Use ingestion's `--start` and exclusive `--end` to fix the history interval.
Opt-in `--continue-on-error` preserves provider failures while collecting other
symbols; it does not bypass data-quality gates. Defaults still fail immediately.
No new training or evaluation is triggered by downloading a wider universe.

## Scope and status

Universe selection, dated sector snapshots, sector-ETF rotation features and
expanded dataset generation are implemented. They are opt-in: the original four
ETF baseline and its trained bundle remain intact. Expanded-input CPU smoke acceptance has passed; all 36 research runs and the frozen held-out audit
are complete; see [verification](verification.md) for the benchmark shortfall
and CPU/CUDA comparison limitations. The environment still trades one
symbol per episode; it does not allocate capital across a simultaneous portfolio.

The checked-in starting configuration is
[sector_research.json](../configs/universes/sector_research.json). It contains
26 tradable instruments (the original four ETFs plus 22 stocks) and 11 context
ETFs. These are a retrospective research sample, not recommendations or a
historical constituent universe. Edit or copy this JSON to choose another set;
the simulator and model do not contain a hard-coded stock list.

## Starting set

| Sector | Stocks | Context ETF |
| --- | --- | --- |
| Communication services | GOOGL, DIS | XLC |
| Consumer discretionary | AMZN, HD | XLY |
| Consumer staples | PG, KO | XLP |
| Energy | XOM, CVX | XLE |
| Financials | JPM, BAC | XLF |
| Health care | JNJ, UNH | XLV |
| Industrials | CAT, HON | XLI |
| Materials | LIN, APD | XLB |
| Real estate | AMT, PLD | XLRE |
| Technology | AAPL, MSFT | XLK |
| Utilities | NEE, DUK | XLU |

SPY, QQQ, IWM and DIA have the category `diversified`; assigning a single issuer
sector to a broad ETF would be misleading. Context instruments supply features
and are excluded from training episodes unless their configured role is changed
to `tradable`. Sector proxy mappings follow the
[issuer's sector ETF list](https://www.ssga.com/us/en/individual/capabilities/equities/sector-investing/select-sector-etfs).
Their historical holdings and sector definitions can change; ETF price returns
are not a reconstruction of today's constituent basket in earlier years.

## Configuration and ownership

- `data/universe/config.py` validates symbols, roles, sector names and history boundaries.
- `data/providers/yahoo/sectors.py` retrieves current issuer-sector metadata, separately from bars.
- `data/universe/sectors.py` validates and stores immutable dated classification snapshots.
- `features/sectors/rotation.py` computes causal cross-sector context.
- `data/generation/datasets.py` creates episodes for the configured tradable set.

`sector` in the universe JSON is a declared current annotation. It is not
automatically inserted into historical feature rows. Real Yahoo snapshots were
retrieved for every starting stock and matched these annotations. Source labels
such as Yahoo's `Consumer Cyclical` are mapped to an explicit fixed vocabulary;
unknown provider labels fail instead of guessing. No API credentials are needed.

Every classification record contains symbol, sector, effective-from/to dates,
public availability, retrieval time, source and revision. A current Yahoo snapshot
is available only when retrieved, and its effective-from date cannot be backdated.
Historical file records require genuinely audited historical availability and
effective intervals. `put_sectors(records, output)` validates and stores those
records; `read_sectors(path)` verifies the content-addressed identity.

## Observation families

Default `--sector-rotation` adds **55 values**: five for each of 11 sector ETFs.

| Value per sector | Definition |
| --- | --- |
| `ready` | Same-session sector and SPY bars, full 20/60-session lookbacks and public availability are verified |
| `return_20` | Trailing 20-session log total return, including cash dividends |
| `relative_20` | Sector 20-session return minus SPY's |
| `relative_60` | Sector 60-session return minus SPY's |
| `rank_20` | Relative 20-session momentum percentile among currently ready sectors; ties take their midpoint |

All channels are observable historical ETF returns; they do not depend on a
stock's current classification. A missing sector or unavailable lookback gives
zero placeholders plus a readiness mask. A developing period, delayed source
bar or stale earlier-session price cannot substitute for the requested session.
Gaps invalidate overlapping windows; no arbitrary filling is performed.

The optional `--sector-labels` adds **13 issuer-specific values**: a known mask,
11 one-hot classifications and the stock's 20-session return minus its sector
ETF's. This flag requires `--sector-rotation`, dated records and at least one
known training-era classification for every stock. Today's snapshots cannot
satisfy that gate for training ending in 2020. It rejects the dataset instead
of training all-zero categories that first appear during inference. Without
historical classifications, current labels remain metadata and these 13 values
are excluded from observations.

Boolean masks and one-hot fields use identity preprocessing (mean zero, scale
one); continuous fields fit only retained training rows. Feature names/order,
proxy mapping, return conventions and classification policy are stored with
the dataset and frozen bundle. Adding issuer labels changes the feature schema.

## History and reproducibility

Historical requests still cover 2010 through September 2026. XLC and XLRE did
not trade over that entire interval. The configuration declares first exchange
trading dates of 2018-06-19 and 2015-10-08 respectively; the issuer distinguishes
fund inception from exchange listing. See the
[XLC fund page](https://www.ssga.com/us/en/individual/etfs/state-street-communication-services-select-sector-spdr-etf-xlc)
and [XLRE fund page](https://www.ssga.com/us/en/individual/etfs/state-street-real-estate-select-sector-spdr-etf-xlre).
Coverage excludes only the declared pre-history interval, records that boundary
and checks every later expected exchange session. Do not move a history boundary
to hide unexplained gaps. Earlier rotation rows mask unavailable proxies rather
than substituting another ETF or recreating pre-inception returns.

Each symbol's history can be fetched as one resumable immutable chunk using
`--full-range`; the default accumulator still supports year chunks. This avoids
hundreds of unnecessary requests for the initial 37-symbol set. Transient bar
failures use bounded retries; nontransient failures remain visible. Reusing a
cursor verifies cached content instead of silently refreshing it.

## Commands

Run with the reviewed Python 3.13 environment from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local ingest-sectors \
  --universe configs/universes/sector_research.json

PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local ingest --provider yahoo \
  --universe configs/universes/sector_research.json --full-range

# Replace both hashes with the directories printed above.
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local dataset data/raw/BAR_HASH \
  --universe configs/universes/sector_research.json --sector-rotation \
  --sectors data/sectors/SECTOR_HASH --supply-demand d1 w1
```

Omit `--supply-demand` or `--sector-rotation` to create matched ablation datasets.
Use `--sectors` to retain metadata even in a non-rotation dataset. Leave
`--sector-labels` off until historical classification records are audited.
Training still uses the shared algorithm factory and the Colab budget runner.
See [verification](verification.md) for actual retrieval/build/check results.

## Scaling to hundreds of stocks

The configurable path supports larger symbol lists now, but expansion should
follow acceptance of this smaller study. First measure retrieval/feature-build
time, dataset size, training throughput and validation behavior. Record that
the single-symbol environment samples separate episodes; widening a list does
not implement portfolio rotation or increase a fixed transition budget.

Before broad-universe performance claims, add point-in-time membership and
historical classifications, delisted stocks, listing/delisting coverage and
corporate-action audits. The initial gate requires every tradable symbol to
have train/validation/test episodes; IPOs after training or delistings before
testing need an explicitly designed membership/eligibility protocol. Do not
bypass this gate silently or select only today's survivors as representative.

Next optimize storage/joins with profiled evidence, compare matched stock counts
and sector balance, set sampling/compute budgets and persist manifests. Hundred-
stock experiments still share the existing 150-unit Colab ledger; they do not
receive a fresh budget. Simultaneous sector allocation requires a separate
portfolio environment/action contract and acceptance milestone.

## Explicit 107-stock complete-case configuration

`stock_research_110.json` remains the original 110-stock acquisition request.
`stock_research_107_complete.json` is a separately named Yahoo-only research
subset with 107 stocks, four tradable broad ETFs and eleven context sector ETFs.
It records its parent configuration hash and exclusions EA/AVB/EQR because
Yahoo returned no bars. Separate partial Alpaca histories are not merged or
treated as complete substitutes. Sector annotations remain present for every
stock; this subset is not equally balanced after those exclusions.

Use explicit universe-scoped coverage checking and development-only episodes;
every retained symbol still requires verified coverage and corporate actions.
For a matched expansion, pass the original panel with `--evaluation-symbols`
so additional names contribute training only. Every failed source request is
retained in the dataset's original provenance and coverage audit. This is
retrospective survivor research, not solved historical membership/delisting
coverage or representative market performance.
