# Additional stock histories — 2026-10-19

The user requested **more stocks pulled**, superseding the proposed longer
training comparison. This task acquired raw histories only; no research dataset,
model evaluation, model training, Colab allocation or broker order was added.

## Coverage

The [configuration](../../configs/universes/stock_research_110.json) retains the
prior 55 stocks and adds 55, giving **110 stocks, ten per declared sector**, plus
four broad ETFs and eleven sector ETFs. These are retrospective research names
and current sector annotations; historical membership and classifications are
not certified. Previous experiment configurations and model bundles are unchanged.

| Source | Stocks with bars | Requested period | Daily bars |
| --- | ---: | --- | ---: |
| Yahoo primary snapshot | 107, plus 15 ETFs | 2010–2023, respecting declared inception bounds | 425,350 |
| Separate Alpaca SIP supplement | EA, AVB, EQR | 2016–2023 | 6,036 |

The 55 added stocks contribute **189,180 newly acquired daily bars**. All 110
stocks have some history across the two separate sources. Yahoo returned
`YFPricesMissingError` for EA, AVB and EQR; each remains in the primary request
and its coverage report with 3,522 missing sessions. This response does not prove
a delisting or its cause. Alpaca supplied all 2,012 expected sessions for each
of those three names in 2016–2023, without verified dividend cash flows.

The supplements are **not merged into the Yahoo snapshot**. They neither fill
2010–2015 nor resolve corporate-action units, historical identities, membership
or terminal outcomes. The primary snapshot is incomplete and the combined
sources are not a training-ready harmonized dataset. No missing bars are filled,
no symbols are silently dropped, and no provider is silently substituted.

## Files and reproducibility

- [Manifest](manifest.json): exact local immutable snapshot paths, requested
  universe, additional names, source coverage, acquisition failures and limitations.
- [Yahoo coverage](yahoo-coverage.csv): all 125 requested symbols, including failures.
- [Alpaca coverage](alpaca-coverage.csv) and [supplement provenance](alpaca-supplement.json).
- Raw snapshots and source chunks remain under ignored `data/raw/`; original
  provider bars and private credentials are not committed or redistributed.

The existing 70-symbol cache is reused; each bar retains its original retrieval
timestamp. Successful source chunks are checksummed before reuse. This command
reproduces/resumes the Yahoo request without starting training:

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.local ingest \
  --provider yahoo --universe configs/universes/stock_research_110.json \
  --start 2010-01-01 --end 2024-01-01 --full-range --continue-on-error \
  --output data/raw/universe-expansion
```

`--end` is exclusive. `--continue-on-error` retains failed provider requests in
snapshot provenance and continues the other requests; default ingestion still
fails immediately. Failed requests are not cached as successful. Storage or
checksum errors still stop ingestion. Default dataset coverage gates remain
unchanged and reject incomplete evidence. Repeating this opt-in command retries
the failed Yahoo symbols; it does not reinterpret them as successful downloads.

The Alpaca supplement uses `AlpacaProvider('sip')` with
`Request(('EA','AVB','EQR'), date(2016,1,1), date(2024,1,1))`, full-range
accumulation into `data/raw/stock-pull-alpaca-2026-10-19`. The account's
200-request/minute ceiling is honored by the existing process-wide transport
pacer below 180/minute. Keys stay in the ignored local `.env`.

## Acceptance and budget

Focused ingestion acceptance passed 17 tests. Full local acceptance passed
**177 tests**, with no failures/skips and 13 existing warnings, in 88.71 seconds.
Tests verify date isolation, explicit failure retention, secret-safe failure
metadata, unchanged default fail-fast behavior, and checksum-error rejection.
Changed-file unused-import checks pass. No dependencies or interpreters changed;
the research environment remains Python 3.13.16.

No Colab session was provisioned and no project compute units were added. The
existing project ledger remains 35.458188/150 units. Immutable market histories
and source caches were retained; no transfer archives, training outputs or
temporary model artifacts were created by this pull. `plan.html` is preserved.
