# Treasury source feasibility — passed with one unavailable date

The fixed official-source/causal-coverage audit passes its preregistered thresholds.
This is source readiness under explicit assumptions, not a trading result. No
Treasury model input is enabled, preprocessing fitted, model replayed or Colab
runtime allocated. Failed GAE/VIX/compact-market families remain closed.

## Acquisition

Exactly15 annual requests acquired2009–2023 official Treasury par yield XML,
retaining3,753 observations from2009-01-02 through2023-12-29. All15 raw bodies and
secret-free immutable receipts are retained locally:5,194,901 bytes total.
Operation duration25.56seconds, within600s. No retry, redirect, all-years query,
pagination, alternate feed or request for2024 onward occurred.

[Fixed protocol](../../configs/experiments/treasury_source_audit.json) SHA256:
0beb46986f3279047bd47ac3e245d1a9f3b749c72cae72b9ab3c45e840238161.
The two fields are source two-year par yield in percent and ten-year minus
two-year par yield in percentage points. Missing source fields remain null.
These are not bond returns or inflation-adjusted yields.

## Coverage

Consumer dataset is fixed
`cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b`,
with111 training and26 development episodes:298,946 training rows plus19,058
development rows. No retired-test split or price history is accessed.

| Denominator | Ready | Total | Readiness |
| --- | ---: | ---: | ---: |
| Unique decision dates | 3,432 | 3,433 | 99.970871% |
| All consumer rows | 317,894 | 318,004 | 99.965409% |
| Development rows | 19,058 | 19,058 | 100% |

Both denominators pass99.5% overall/era and99% per year. Minimum yearly fraction
is99.507389%, in2010. All per-symbol/split, year, era, methodology-period and age
counts are retained in [summary.json](summary.json).

The unavailable decision date is**2010-10-13**, affecting110 training rows.
Its latest eligible source date2010-10-11 explicitly has both target yields null.
The audit does not zero-fill, interpolate, backfill with an older complete value
or discard the missing date from the denominator. Downstream model construction
must separately define matched readiness handling and contiguous episodes before
any screen. Source coverage acceptance alone does not accept a model payload.

Of ready unique dates,3,407 use age2 XNYS sessions and25 use age3. Calendar ages
range2–6 days. The fixed maximum is3 XNYS sessions/seven calendar days. Source
dates may be Treasury business days on which NYSE is closed. Current-vintage
holiday rows and true observations are preserved as received.

## Availability and vintage limits

Assumed availability is close plus15minutes of the second XNYS session strictly
after the source observation date. Treasury's [methodology documentation](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology)
says yields are generally published by18:00Eastern but can be delayed. A two-
session lag does not prove every historical publication or revision timestamp.
The current feed is not verified point-in-time history. A prospective deployment
would need actual observed receipt times and stale-input rejection.

Treasury changed interpolation methodology on2021-12-06. Coverage before that
change is2,932/2,933 dates; after it,500/500. Values remain as reported for each
date; no historical backcast or smoothing is performed. The official
[annual XML documentation](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed)
describes the fixed source endpoint and year-filtered requests.

## Independent acceptance and cleanup

[Verification](verification.json) independently reconstructs all3,753 rows from
checksummed original XML, checks all15 receipt hashes/HTTP statuses/complete flags,
URLs and years, then uses pandas backward as-of alignment and vectorized exchange-
calendar searches. All3,433 date joins,318,004 consumer counts, year/era counts,
source/availability timestamps and feature values agree exactly; maximum feature
value difference0.0. The original dataset, frozen normalizer and project ledger
remain unchanged.48 offline tests and targeted Ruff passed before acquisition.

Actual worker exited0. The operation lock is released, alarm cleanup completed,
and no task temporary files remain. Raw responses, receipts, snapshot and date
alignment are retained under ignored `data/context/treasury/`; the mutable terminal
journal is `.runtime/treasury-source-progress.json`. No source originals are deleted
or uploaded with model artifacts. Source ID:
`bc52edefc6baccafdb283e054c5d8f24c6593fcdf962e0bd15ec6dcf0e6a0bd4`.

No additional Colab units: cumulative project39.351004/150, last official account
348.74 idle; all prior owned runtimes remain stopped. There is no new model return.

## Next acceptance boundary

Audit readiness-matched payload segmentation and preserve original seven-field
preprocessing. Freeze a separate legacy/masked/full comparison only after that
acceptance, with common original tensors/RNG, unchanged rewards/costs/actions,
selected/final stock/ETF/median/risk gates, bounded CPU smokes and screens, and
multi-seed/prospective requirements. No maturity, lag, age-cap or gate search.
A feasible source may still produce an unsuccessful trading model.
