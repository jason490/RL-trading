# Historical membership and delisting evidence

## Scope and current status

This layer imports dated security identity, ticker mappings, universe membership,
raw prices/corporate actions and terminal-outcome evidence. It stores immutable
snapshots and generates a coverage report plus an eligible-session request table.
It reduces the risk of silently constructing a universe from today's survivors;
it cannot manufacture missing historical facts or independently certify a vendor.

Implemented modules are under `data/universe/historical/`, the read-only listing
adapter is under `data/providers/alphavantage/`, and the CLI is
`rl_stocks.runtime.historical`. No new dependency, broker connection, Colab runtime
or training run is required. Existing models, datasets and the frozen shadow
campaign remain separate. The current real 22-stock sample is still retrospective.

**Real historical coverage is not yet verified.** No credentialed historical source
or vendor export has been supplied. The supplied example is synthetic software
evidence only. `coverage_passed` means the supplied evidence meets these checks,
conditional on its attestations; `training_ready` remains false. Historical episode
construction, suspensions, membership-exit execution, delayed/non-cash settlements
and an approved new research protocol are separate integration gates. Existing
`data/generation/datasets.py` does not consume this new evidence automatically.

## Provider audit, 2026-10-17

| Source | Documented capability | What must still be established |
| --- | --- | --- |
| [Alpha Vantage LISTING_STATUS](https://www.alphavantage.co/documentation/#listing-status) | Active or delisted US stock/ETF lists at dates after 2010-01-01 | Not index membership, permanent security identity, publication-vintage proof or terminal-return data |
| [CRSP permanent IDs](https://www.crsp.org/research/permno/?e-page-52e5552=2) and [delisting definitions](https://www.crsp.org/crsp_pdf/crsp-us-stock-indexes-databases-data-descriptions-guide-crspaccess/) | Stable PERMNO/PERMCO and explicitly defined delisting outcomes/missing-return codes | Licensed access, normalization, source-field conventions, historical membership and vintage evidence for the selected universe |
| [Norgate content](https://norgatedata.com/data-content-tables.php) and [FAQ](https://norgatedata.com/data-package-faq.php) | Historical constituent series, stable assetid and delisted coverage in documented packages | Entitlement; announcement timestamps are not supplied; post-delisting events and historical classifications have stated limitations |
| Current Yahoo bars | Price/cash-dividend research inputs | Historical membership, stable identities and verified terminal outcomes remain unverified |

The Norgate FAQ distinguishes major-exchange removal from continued OTC trading,
and states that it does not provide post-delisting event information. Do not label
all delisted vendor prices as fully settled liquidation cash. A listing-removal
date is not necessarily the last tradable session, a bankruptcy date or the payment
date. No paid subscription, SDK or data purchase is initiated by this project.

## Import and inspect the example

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.historical import \
  examples/historical-universe/history.json --output data/historical-demo

PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.historical audit \
  data/historical-demo/HISTORY_ID --output artifacts/historical-demo --require-complete
```

Replace `HISTORY_ID` with the directory printed by the import command. The example
uses two fictional securities sharing a recycled ticker and a documented synthetic
zero-value settlement. It contains no historical market observations and cannot
justify performance or universe-coverage claims. For real files, use
`--output data/historical` and a new report directory. The importer verifies the
schema and writes `history.json` under its SHA-256 identity. Editing it afterward
causes a checksum failure; create a new version instead.

The audit's `--require-complete` flag exits with status 2 if any coverage issue is
present, after writing the complete report. Success approves evidence coverage
only; it does not override the separate `training_ready=false` gate.

## Canonical file schema

The top-level object has exactly these keys:

| Key | Contents |
| --- | --- |
| `version` | Integer 1 |
| `universe` | `name`, inclusive `start`, exclusive `end`, `calendar="XNYS"`, `selection_rule`, `source` |
| `attestations` | Explicit evidence checks below |
| `securities` | Stable security/issuer identity and lifecycle audit facts |
| `aliases` | Versioned ticker-assignment events |
| `membership` | Versioned add/remove membership events |
| `terminals` | Terminal-outcome evidence and unresolved cases |
| `prices` | Stable `security_id` plus canonical `bar` records |

Use the committed example for exact field names and types. Unknown keys, duplicate
IDs, ambiguous same-time event versions, invalid chronology and duplicate
security/session prices are rejected. Tickers can change or be reused by a new
security; they cannot serve as permanent IDs. Multiple share classes need separate
security IDs even when they share an issuer. Vendor identifiers must be namespaced
consistently (for example `crsp:permno:...`); do not invent permanent identity from
a ticker/IPO-date combination and label it verified.

Every security has `security_id`, `issuer_id`, `exchange`, `instrument_type`
(`stock` or `etf`), `listed_from`, `last_trade_session` (nullable), `status`
(`active`, `delisted`, `unknown`), and source provenance. Active securities cannot
have terminal trading dates. An unknown lifecycle, or a delisted cohort security
with no last-trade date, blocks coverage. Lifecycle facts are audit labels and
are never used to decide which later winners to include historically.

Alias, membership and terminal events share `event_id`, `security_id`, `source`,
`revision`, `available_time` and `retrieved_time`. Times must be timezone-aware UTC;
retrieval cannot precede public availability. A source correction uses the same
event ID with a new availability time/revision. Do not overwrite the first vintage.
Distinct records claiming incompatible states at the same effective date cause
an audit failure rather than arbitrary tie-breaking.

Aliases add `ticker` and `effective_from`. Membership adds `universe`, boolean
`member` and `effective_from`. An add can be announced before its effective date;
it becomes usable only after both conditions hold. A removal is its own event,
so a future removal cannot truncate past membership before the information was
available. All events must reference the declared universe and known security IDs.

### Evidence attestations

Every payload explicitly supplies these entries, each with boolean `verified`,
`source` and `revision`:

- `membership_complete`: the source covers the declared universe and range, including initial membership and removals. Otherwise an absent row cannot prove non-membership.
- `identity_verified`: permanent security IDs, ticker changes/reuse and share classes are mapped correctly.
- `membership_vintages_verified`: availability timestamps have evidence; current lists or reconstructed effective dates are not silently backdated into announcements.
- `corporate_actions_verified`: executable prices and action quantities/cash flows are coherent.
- `delisted_price_coverage_verified`: dead/removed names have been sought, not silently lost through failed downloads.

These are explicit source assertions, not an automated certification. Supply real
vendor/audit references and revisions. Setting them true without evidence does not
remove survivorship bias. Import incomplete evidence with false flags to obtain
a useful failure report; there is no `allow-unverified` bypass for coverage approval.

## Price and terminal conventions

Each price row has `security_id` and the existing canonical `Bar` fields, including
the historical ticker, provider/feed, revision, retrieval/event/availability times,
OHLCV, cash dividend and split factor. This first interchange version requires
`raw_with_actions` prices; converting split-adjusted histories and payout units
requires a separately audited normalization. Duplicate dates within one security
are rejected even if the ticker changed. Different securities with a recycled
ticker remain distinct price histories.

For every expected member-session, the audit checks that a bar exists, its ticker
matches the known alias, open/close match the exchange calendar and it was available
by the declared decision time (session close plus 15 minutes). Missing or delayed
bars are retained as coverage problems. The audit does not forward-fill prices,
join securities by today's ticker, substitute a feed or exclude a failed download.
Suspension-aware valuation remains a later episode/execution integration gate.

Terminal records add `last_trade_session`, `settlement_time`, `kind`,
`reference_price`, `value`, `currency="USD"`, boolean `verified`, and
`basis="additional_to_last_close"`:

| `kind` | Meaning of `value` | Coverage treatment |
| --- | --- | --- |
| `cash_per_share` | Verified nonnegative settlement value per final share | Zero is permitted only as an explicit verified outcome |
| `delisting_return` | Verified additional simple return relative to the last marked close | Must be at least −1; proceeds = reference × (1 + return) |
| `unresolved` | Null | Blocks coverage; never treated as zero or −100% |
| `non_cash` | Null | Blocks this version until successor-security entitlements are modeled |

The reference must match the final recorded close in the same share/price units.
The terminal date must match the security master and settlement cannot precede
the last close. Both provenance and outcome verification are required. Unresolved
records can leave the reference null rather than fabricate a final price.

`terminal_proceeds()` verifies the reference and calculates an outcome value; it
is not an order, feature, immediate historical liquidation or an integrated trading
environment. A long can lose its full remaining position value at zero settlement;
a short has the opposite cash obligation, subject to later borrow/timing modeling.
Do not replace a missing close with a settlement quote or credit cash before its
actual payment/availability. Ordinary last-bar dividends/actions are already
booked; the extra outcome must not count them again.

CRSP missing-return codes such as −55/−66/−88/−99 are not simple returns and are
rejected. Import them as unresolved, retaining their explanation in source records.
Source formats can already incorporate terminal returns in another return series;
do not blindly multiply them again. This importer expects raw prices plus a
separately normalized additional outcome, not precombined total-return series.

## Point-in-time lookup and coverage outputs

```python
from rl_stocks.data.universe.historical.storage import load_history
from rl_stocks.data.universe.historical.registry import member_asof, ticker_asof

history = load_history("data/historical/HISTORY_ID")
member = member_asof(history, security_id, session, decision_time)
ticker = ticker_asof(history, security_id, session, decision_time)
```

Lookups first choose the latest version of each event available at the decision,
then choose the applicable effective state. A newly published correction cannot
rewrite what an earlier decision knew. Ticker resolution also distinguishes listing
lifecycles. Selection uses membership events only; future terminal outcomes do
not select, weight or exclude securities. Supplied completeness/vintage attestations
remain necessary: a function cannot infer missing historical announcements.

Reports include:

- `coverage.html`: summary and every issue, with the conditional interpretation.
- `coverage.json`: identities, counts, failures, unselected master IDs and full eligible panel.
- `eligible-sessions.csv`: stable ID, historical ticker, session, decision timestamp and price presence. This is an ingestion/request plan, not an RL dataset.
- `security-coverage.csv`: expected/covered member-session counts, missing dates and terminal coverage for every master security.

A cohort security whose last trade falls in the audited range needs terminal
evidence even if it left the index earlier. The final reference price is required
beyond membership-price coverage where necessary. Master securities not selected
by the declared as-of rule are listed separately; missing selected prices never
become silent exclusions. Check sector/classification coverage separately before
adding historical sector labels.

## Read-only Alpha Vantage discovery

Set `ALPHAVANTAGE_API_KEY` privately in the process environment, then request one
specified date/state at a time:

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.historical listings \
  --date 2015-01-02 --state active --output data/listings
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.historical listings \
  --date 2015-01-02 --state delisted --output data/listings
```

No key goes in arguments, logs, stored URLs or source control. Each command makes
one bounded-time request, validates the documented CSV, stores its raw text and
candidate records immutably, and fails on quota/error/empty responses. It does not
retry, batch years of calls or assume a particular subscription's request limit.
Check the account's entitlement/rate terms before scheduling many dates.

Historical `asof_date` is recorded separately from retrieval/public availability.
Candidates are explicitly marked as lacking stable identity, historical-publication
vintage and terminal-return verification. No listing candidate is automatically
converted into certified membership, an index constituent, a last-trade date or a
terminal outcome. Retrieve both active and delisted states when building discovery
coverage; reconcile identities with independently verified vendor/file evidence.

## Verification and next integration

Tests use only synthetic fixtures and mocked HTTP, including ticker reuse,
announcement/effective-date separation, later corrections, future outcome
independence, failed/dead-stock downloads, zero/buyout outcomes, missing-return
codes, incomplete attestations, immutable storage and secret-safe API errors.
The demo contains no market data. Real coverage requires actual source exports
and evidence; neither the optional API key nor a licensed dataset is configured.

After a real cohort passes, implement and verify historical episode construction
and membership/terminal-aware policy and benchmark accounting. Preserve declining
and failed stocks, chronological splits/purging and training-only normalization;
never require every IPO/delisted security to appear in all three date splits.
Declare a separate matched universe experiment before training. The retired
2024–September 2026 test interval stays retired, and the existing frozen shadow
campaign's symbols and models must not be silently widened.
