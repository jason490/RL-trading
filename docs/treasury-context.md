# Lagged Treasury rate context

Status: actual fixed source/coverage audit passed with one unavailable training
date. Model inputs remain disabled; no return improvement is claimed.
The [fixed audit](../configs/experiments/treasury_source_audit.json) precedes source
acquisition. It does not reopen failed VIX, compact-SPY-context or GAE families.

## Information hypothesis

The baseline observes own-stock price/volume indicators. Previous sector and
SPY-relative comparisons concern equity prices; VIX concerns implied volatility.
Two fixed yield coordinates would instead describe the short nominal Treasury
rate and term structure: source two-year par yield in percent and ten-year minus
two-year par yield in percentage points. These are market interest rates, not
realized bond returns, inflation-adjusted rates or a guarantee of predictive value.
No maturity/window/lag/feature-subset search is authorized by this source proposal.

## Official source

Treasury's [XML feed documentation](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed)
provides a year-filtered endpoint and lists daily par yield history from1990.
Use only fifteen annual2009–2023 requests;2009 supplies lag warmup. Never request
all years or follow pagination into2024 onward. Preserve original response bytes,
URL, retrieval UTC and SHA256 in immutable storage before normalization.

The [methodology page](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology)
says rates are generally published by18:00Eastern, with possible delays. It also
records the2021-12-06 change from quasi-cubic Hermite interpolation to monotone
convex interpolation. Retain reported historical observations, document that break
and report readiness on both sides. Do not backcast current methodology onto old
dates. Current archives do not establish original publication vintages or rule out
later corrections.

## Time and missingness contract

XML `NEW_DATE` is an observation business-date label, not a UTC publication time.
For research, assume availability at close plus15minutes of the **second XNYS
session strictly after that date**. This conservative lag is an assumption, not
proof that every historical delay was shorter. Production use would require
actual retrieval/publication timestamps and staleness controls.

Backward as-of joins may use only an actual eligible observation. Maximum age is
three XNYS sessions and seven calendar days, allowing explicit Treasury/NYSE
holiday differences. Report age distributions and every unavailable decision;
never interpolate, zero-fill yields or silently substitute another source.
Missing two/ten-year facts remain null source rows and reduce readiness.

## Boundaries and acceptance

The implemented collector enforces: one request per year, one-second minimum spacing,
5s connection/30s read timeouts,600s total operation deadline,2MiB per response and
300 entries per year. Stop on the first access/schema/source failure; no automatic
retry, redirect, pagination, alternate feed or expanded cap. Cache reuse must check
receipts/raw bytes and preserve the original cumulative research budget.

Coverage will use all111 training and26 development episodes in fixed dataset
`cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b`.
Require99.5% overall and per era,99% per year, independently for both unique
decision dates and all consumer rows. Eras are2010–2014,2015–2020 and
2021–2023. No normalization, model replay, return scoring or retired-test access is
part of source acceptance. A passed source audit still needs a separately fixed
matched legacy/masked/full experiment, training-only preprocessing, common
initial tensors, unchanged reward/actions/costs and multi-seed/prospective gates.

## Offline parser

`data/providers/treasury/xml.py` accepts only bounded UTF8 Atom XML and exact
source namespaces. It preserves decimal percent strings, missingness and dates;
rejects duplicate facts/dates, nonfinite/impossible values, out-of-year/weekend
observations, wrong schemas, entity declarations, pagination and retired years.
It never makes requests or fits preprocessing. Synthetic fixtures establish
parser behavior only; they do not establish actual provider compatibility,
coverage, historical availability or trading performance.


## Acquisition and coverage command

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_treasury_source.py
```

The controller owns one local operation lock, enforces request-start pacing and
writes `.runtime/treasury-source-progress.json` before acquisition. An existing
journal forbids automatic restart. Failures retain complete/incomplete raw bytes
and secret-free receipts where a response exists; transport failures retain the
journal. The controller releases its lock, cancels/restores its alarm and writes
failure evidence. No redirect, compressed-response expansion, retry or fallback.
Immutable raw/receipt/index cache reuse checks hashes before returning; corruption
fails without a replacement request. Actual complete raw responses precede XML
parsing; a retained response does not by itself mean accepted source data.

Consumer alignment uses the actual XNYS calendar, including Hurricane Sandy,
Good Friday, early closes and daylight-saving offsets. Eligible source dates may
be Treasury business days when NYSE is closed. The latest eligible missing fact
reduces readiness rather than falling back to an older complete observation.
Reports preserve source/availability timestamps, age counts, missing reasons,
per-symbol/split coverage and methodology periods. Source/data/normalizer/ledger
hashes must stay unchanged; model registry and preprocessing remain disabled.

48 offline checks pass across parser, acquisition, pacing, calendar alignment,
coverage and handled CLI failure. The latter verifies lock/alarm cleanup and
restart rejection. These are fixtures, not live coverage or provider acceptance.
Actual source acquisition is the next step after committing this implementation.


## Actual source acceptance

[Accepted evidence](../results/treasury-source-2026-10-25/README.md):15 official
annual requests,3,753 observations/5,194,901 raw bytes,25.56s. Readiness3432/3433
unique dates and317894/318004 consumer rows; thresholds pass for both denominators,
every era and year. One null-maturity source row makes2010-10-13 unavailable for110
training consumers; no fallback/imputation hides it. All19058 development rows ready.
Independent original XML/pandas/calendar verification matches all values, timestamps
and counts exactly; dataset, normalizer and original ledger unchanged. Worker exited0,
lock released, no temporary files/new Colab units. Current-vintage/publication
assumptions remain unverified. Readiness-matched payload acceptance and separate
training preregistration remain before any input/model allocation.


## Fixed readiness-matched payload construction

[Payload rules](../configs/experiments/treasury_payload_rules.json) are fixed before
actual payload preprocessing. Masked and full arms share identical source readiness,
20 preceding-row purge before each unready training date and contiguous segmentation.
Discard segments shorter than513 bars; require exactly one remaining segment per
original training symbol. Reject multiple eligible segments instead of selecting
by return or silently dropping a symbol. Every original validation row must be
ready and remain present. The original legacy dataset is preserved separately.

Both arms append the same two column names; masked raw values/means are zero and
scales one. Full extra means/scales fit retained training rows only. The original
seven fields and names/means/scales remain exact, as do all retained bars/ATR,
validation dates and episode keys. Metadata records exact original row bounds,
missing/purged/short-segment counts and identical512-step eligible starts. Future
sampling keeps all111 training symbols and unchanged stock/ETF group mass.

Generic training explicitly rejects Treasury schemas, fields or manifests until a
separate registered model experiment exists. Payload preprocessing cannot enable
learning by itself. No model, optimizer, provider request or Colab runtime is used:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/prepare_treasury_payloads.py
```

Source summary/verification/alignment and original preprocessing hashes are pinned.
The600s local worker writes a durable progress journal and rejects automatic restart;
immutable datasets are reloaded and exactly reconstructed before acceptance. Synthetic
checks cover gap purging, exclusion accounting, multiple/short segment rejection,
unchanged evaluation, identical sampler bounds, source immutability and validation-
only perturbations that cannot change fitted statistics. Actual payload acceptance
and independent verification remain pending; no performance benefit is claimed.
