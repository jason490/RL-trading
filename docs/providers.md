# Provider and dependency audit — 2026-10-14

The implemented request is SPY/QQQ/IWM/DIA, 2010-01-01 through September 2026. A Yahoo download on 2026-10-14 returned 4,211 daily bars per symbol, from 2010-01-04 through 2026-09-30, with no missing XNYS sessions. Split/dividend fields are preserved under the adapter's documented semantics. Alpaca endpoint coverage remains unverified. Historical publication vintages and profitable performance are not established by coverage checks.

Yahoo is an unofficial personal-use research adapter, not a production service guarantee. It requests original daily OHLC with cash dividends and records split-adjusted-price semantics. Its reconstructed close+15-minute availability and revised historical data require explicit interpretation; it does not supply historical public-availability vintages. Alpaca requires account credentials and explicit entitlement; IEX and SIP volumes differ. Its earliest capability is declared as 2016; the requested pre-2016 range may be unsupported/empty and must be reported rather than substituted. Bar-only Alpaca snapshots are not accepted by the default dataset gate until dividends are independently verified.

Current sectors, financials, options/OI, sentiment, trades/quotes and order-book data remain disabled. The typed capability registry rejects unsupported families independently. No current snapshot is inserted into historical observations.

Dependency pins target Python 3.13, torch 2.8.0, Gymnasium 1.2.0, SB3 2.7.0 and optional alpaca-py 0.42.0/yfinance 0.2.65. SB3 release metadata permits Gymnasium >=0.29.1,<1.3, NumPy >=1.20,<3 and torch >=2.3,<3, so the direct pins satisfy those declared bounds. `requirements.lock` now contains hash-pinned transitive dependencies. See [verification.md](verification.md) for actual native imports and training acceptance.

Sources checked during implementation:

- [SB3 2.7.0 release metadata](https://pypi.org/pypi/stable-baselines3/2.7.0/json)
- [SB3 2.7.0 installation](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/install.html)
- [Alpaca 0.42.0 release metadata](https://pypi.org/pypi/alpaca-py/0.42.0/json)
- [Pinned Alpaca HTTP transport](https://github.com/alpacahq/alpaca-py/blob/v0.42.0/alpaca/common/rest.py)
- [Alpaca historical data API](https://alpaca.markets/sdks/python/api_reference/data/stock/historical.html)
- [yfinance history API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html)
- [Official Colab CLI](https://github.com/googlecolab/google-colab-cli)

The installed official CLI was inspected locally at version 0.7.4. Its `usage` parser recognizes the installed release's exact balance/rate/assignment labels and refuses unknown output. Logging/settings/history writes are redirected to the project, without changing authentication or weakening the compute ceiling. Authenticated usage now works; the preflight balance was 382.82 units, rate 0.00/hour, with zero assignments. Current project charges and cleanup belong in [verification.md](verification.md).

## Adapter operations

All adapters satisfy `BarProvider` and return canonical bars. Baseline execution remains independent of any historical provider. `Request()` fixes the four-symbol/date request; explicit requests and `--universe` can change the symbol set in research. The optional Yahoo sector adapter captures dated current metadata separately from bars; see [sectors/universe](sectors-universe.md). No unsupported family is silently replaced by bars.

| Adapter | Inputs | Availability/calendar convention | Primary limitation |
| --- | --- | --- | --- |
| `SyntheticProvider(seed=11)` | Seed and dates | Weekdays; fixed UTC open/close, close+15 minutes | Not an exchange calendar, no market evidence |
| `FileProvider(path)` | Canonical CSV or Parquet | Preserves imported canonical timestamps | Import authority/action quality must be audited |
| `YahooProvider()` | Optional yfinance, unauthenticated history | XNYS open/close; reconstructed +15-minute availability | Revised split-adjusted history, unofficial personal use |
| `AlpacaProvider(feed='iex', client=None)` | Environment credentials or mocked/injected historical client | XNYS open/close; reconstructed +15-minute availability | Feed entitlement, pre-2016 coverage, dividend omission |

The accumulator isolates symbol/year retrieval and saves raw-source chunks. It resumes verified cached chunks; explicit refresh requests new revisions. See [storage](storage.md) for cursors and [datasets](datasets.md) for coverage gates.

### Synthetic

Produces deterministic OHLCV with separate symbol seeds and no corporate events by default. The accumulation request is chunked, so generator state restarts within each request chunk; this fixture is intended for accounting/wiring rather than a realistic long market process. Generated timestamps/volume are labelled synthetic. Routine tests must remain independent of API availability.

### Files

CSV columns are the canonical `Bar` names, including source/feed/adjustment and temporal fields. Numeric columns are converted to floats; unexpected columns are ignored. Defaults exist for dividend/split in the canonical dataclass, but an imported file with missing action evidence must not be described as audited raw history. Parquet requires pandas/PyArrow. Filtering respects symbol and inclusive-start/exclusive-end dates; order/duplicates are validated rather than repaired silently.

### Yahoo

Uses per-symbol `Ticker.history(interval='1d', auto_adjust=False, back_adjust=False, actions=True, repair=False, raise_errors=True, timeout=30)`. It keeps cash dividends and records that OHLC are split adjusted, setting canonical split=1. XNYS session validation rejects non-session bars. Original frame rows are preserved in source snapshots.

Historical availability is a conservative reconstruction, not verified public-availability vintages. Do not insert current sectors/options metadata into old episodes. Personal-use and data reliability limits remain even after a successful download. No fallback is supplied if history is empty, throttled or unavailable.

### Alpaca

Uses the official historical client and `StockBarsRequest` with daily timeframe, explicit IEX/SIP and split adjustment. It reads credentials only when the adapter is called. The official SDK handles pagination; an isolated pinned transport extension bounds GET connect/read waits to (5,30) seconds and SDK retry attempts to two. The adapter never constructs a trading client or submits an order.

Alpaca daily bars omit dividend cash flows. Its snapshots therefore carry `dividends_verified=false`; an explicit research override does not establish realistic total-return accounting. Actual pre-2016 history and account feed entitlements must be sampled later. Mocked-client tests exercise request/feed/normalization wiring when dependencies are installed; they do not prove account access.
