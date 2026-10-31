# Historical instrument-type evidence

This category owns evidence used to distinguish stocks and ETFs in the
[task-conditioning study](task-conditioning-research.md). Historical type
admission is incomplete. Today's universe labels and CIK lookup results are
discovery aids; neither supplies historical availability or ticker continuity.
No collected evidence changes the universe, normalization or performance gates.

## Immutable source handling

Reuse the SEC original-instance downloader/cache and the privately configured
request identity. Never print the identity or credentials. The original receipt
binds endpoint, retrieval time, byte count and SHA256. Raw files remain outside
Git under `data/raw/sec-xbrl`; source/audit receipts and compact derived evidence
are retained. Reuse cached bytes before a new request. Existing single-request,
byte/time bounds and rejection rules apply; do not retry access rejection or
silently replace a missing document with a newer filing.

The AAPL2009 source was already retained: accession0001193125-09-214859,
SHA256`5bf4c28bd9ac411c366a012bb4e929d4f13167b454c8761a7297d4f60cf99c20`.
Its original instance explicitly reports AAPL, Apple as registrant and issuer
common shares. [Acquisition receipt](../results/task-conditioning-2026-10-31/aapl-primary-acquisition.json).
This is a dated document observation, not approval of a constant type across an
entire historical security series.

## Restricted original-instance extractor

`data/providers/sec/instrument_type.py:inspect_stock_type` reads at most20MB,
checks exact SHA256 and rejects DTD/entities, namespace rebinding, non-instance
roots, duplicate context/unit IDs and orphan selected facts. It recognizes the
standard DEI fields for trading symbol, registrant, form, report end and issuer
common shares. It does not load schemas, linkbases or remote resources.

Only facts whose context identifies the expected CIK under the SEC identifier
scheme can contribute. Segment/scenario facts, late or invalid periods, nil
values and invalid/nonpositive share counts are excluded. Shares must use a
simple XBRL shares unit. Numeric counts validate source evidence and are omitted
from output; no financial feature is produced. Co-registrant and differently
qualified facts do not establish the parent issuer's identity.

`document_exact_stock_identity_hint` requires the expected ticker, exact form,
registrant and an eligible common-share fact in that original instance. Missing
ticker, old ticker or issuer mismatch cannot silently pass. The term *hint* is
intentional: the result always leaves historical admission, inferred validity
interval and model-input enablement false. Even an exact document observation
needs reviewed availability, security continuity and the declared classification
interval before joining historical observations. Common-share terminology alone
is not a universal ETF classifier.

## Completed cached-source audit

Run without downloads, credentials, fitting or Colab:

```bash
MPLCONFIGDIR="$PWD/.runtime/matplotlib" PYTHONPATH=src \
  .venv/bin/python scripts/audit_cached_stock_types.py \
  --output artifacts/new-task-stock-type-audit.json
```

The output must be new. The script binds the existing dataset-type audit to the
original candidate ledger, inspects only retained source receipts, and requires
both acceptance and filing dates strictly before the first training day. This
avoids assuming same-day availability; it does not independently prove public
dissemination. Receipt/source/audit/code/ledger hashes are rechecked after reading.
No record gets a fabricated `known_from` date.

The [completed audit](../results/task-conditioning-2026-10-31/cached-stock-evidence.json)
inspected85 original documents for107 stocks, preserving175 protected inputs:

| Status | Stocks | Interpretation |
| --- | ---: | --- |
| Exact document identity hint |33| Explicit expected ticker/form/issuer common-share evidence |
| Cached source without exact ticker hint |52| Source exists but the strict document criterion fails |
| No cached earlier instance |22| Requires other retained or bounded newly collected primary evidence |

Of the52 nonmatching cases, GOOGL/LIN/MDLZ explicitly show GOOG/PX/KFT. These
are review cases, not automatic alias substitutions or proof of price-series
continuity.49 others lack an eligible explicit ticker in the selected instance.
The22 without a cached earlier instance differ from the ten without any earlier
candidate ledger filing: candidate presence is not raw-document availability.
All107 classifications remain unadmitted. Four ETF primary records also remain
outside this stock-instance audit.

## Next evidence work

For the52 cases, inspect original filing cover/market sections or another
contemporaneous original source, binding accession and publication metadata.
For the22 cases without a retained earlier source, select a dated primary source
before collection; preserve absent/rejected cases. Resolve renames, predecessors
and share classes explicitly. For ETFs, preserve dated fund registration and
ticker evidence, including QQQ's earlierQQQQ identifier. Do not use a later filing
as if it had been available when the older history occurred.

Only after complete reviewed records cover the exact111 training symbols can
`validate_type_manifest` admit the manifest. Physical source review, metadata
validation and model performance are separate requirements. No Colab allocation
is needed for this source work; cumulative credit and retired-test restrictions
remain unchanged.

## Bounded original HTML collection

`html_source.py:acquire_filing_html` supports exact pre2024 SEC filing endpoints,
normalized CIK/accession binding, private identified requests, immutable cache
reuse, single-request20MB/60s bounds, no redirects/retries, and immediate rejection
of concurrent collectors at the same cache root. A new source retains its raw
bytes and receipt. This collector must not overlap the separate XML downloader.
`filing_index.py:primary_html` selects only the single sequence1 document whose
form matches the original index; it does not guess a filename or substitute an
exhibit/new filing.

`scripts/collect_task_type_covers.py` fixes the first three alphabetically sorted
cached unresolved cases, at most one index plus one primary source each. A durable
claim prevents automatic retry, including after rejection/interruption. Any failure
stops the remaining collection; original evidence and ledger hashes are retained.
Collection itself never admits a classification or performs model/provider-price
operations. Tests use mocked HTTP, including single-attempt403 rejection and
cache-corruption detection. Actual source receipts are separate from these tests.

## Actual three-case original-cover result

The bounded collection completed all six index/primary operations for ABT, ADBE
and AEP, retaining7,298,946 raw bytes with source receipts. There were no retries,
redirect substitutions, Colab sessions or model operations. The selected indexes
supply official filing and acceptance timestamps; the body document is the
explicit sequence1 form selected from each original index.

[Collection](../results/task-conditioning-2026-10-31/original-cover-collection.json)
and [review](../results/task-conditioning-2026-10-31/original-cover-review.json)
retain the exact endpoints, hashes, brief checked source anchors and ledger/code
invariance. `scripts/review_task_type_covers.py` reproduces the fixed observations
against those raw bytes, with a fresh output path. It is a review verifier for
these three documents, not a universal automatic classifier.

| Symbol | Filed | Conservative availability | Original-document finding |
| --- | --- | --- | --- |
| ABT |2010-02-19|2010-02-23| Registered common shares; no admitted exact ticker clause |
| ADBE |2010-01-22|2010-01-26| Registered common equity plus explicit ADBE trading-symbol statement |
| AEP |2010-02-26|2010-03-02| Parent's registered common stock distinguished from subsidiary debt; no admitted exact ticker clause |

Availability uses the existing explicit research assumption: second XNYS open
strictly after both SEC-local acceptance and filing dates. It is not a verified
exact first public dissemination timestamp. All dates precede each first training
session. Original cover statements improve document evidence, but do not establish
a constant type/security validity interval over2010–2020 or cure survivorship.
The cached33 exact hints remain their original immutable result; ADBE supplies an
additional HTML observation. Complete111-symbol metadata remains unadmitted.

Next review original incorporated market sections/exhibits for ABT/AEP and other
unresolved records, using a separately bounded source selection. Do not treat a
company abbreviation or a compound/drug name containing a ticker as a market
symbol statement. Preserve failed/partial evidence and earlier collection claims.

## Historical fund source scope

The master-index reader now accepts an explicit restricted fund/424B2 form scope;
its default10-K/10-Q behavior is unchanged. Retired metadata exclusions, safe
paths, joint-filer identity and source hashes remain enforced. This does not add
fund reports to any existing financial feature ledger.

`collect_task_fund_sources.py` fixes four pre-training accessions against retained
2009q4/2010q1 master bytes: SPY/QQQ/DIA fund485BPOS registrations and a2009 bank
424B2 describing IWM. The IWM source is a bank's dated derivative filing, not an
iShares registrant filing; its issuer must never be assigned to IWM. Both the
original index and explicitly typed primary HTML are collected, at most eight
operations, with one durable claim and stop-on-failure/no-retry behavior. Source
selection/collection alone does not verify the fund statements or admit metadata.

## Actual incorporated exhibit and four-fund observations

A separate single-request AEP collection retained the original sequence11 EX-13
annual-report HTML (17,995,757 bytes). Its [review](../results/task-conditioning-2026-10-31/aep-exhibit-review.json)
confirms a common-stock/NYSE market statement, but retains the distinction
between a company abbreviation and an explicit ticker keyword clause. ABT's
original index has no EX-13 annual-report attachment; its main filing supplies
the common-share market description. No missing exhibit was guessed or replaced.

The fixed fund collection completed all eight index/primary operations, retaining
3,598,117 raw bytes. [Source receipts](../results/task-conditioning-2026-10-31/fund-source-collection.json)
and [checked observations](../results/task-conditioning-2026-10-31/fund-source-review.json)
record source/master/index agreement, short source anchors and unchanged protected
inputs. All conservative dates precede the first training session:

| Canonical research symbol | Source filed | Conservative availability | Observation limit |
| --- | --- | --- | --- |
| SPY |2010-01-27|2010-01-29| Fund registration names ETF trust and SPY market symbol |
| QQQ |2010-01-26|2010-01-28| Fund registration describes unit investment trust trading asQQQQ; canonical alias still requires review |
| DIA |2010-02-26|2010-03-02| Fund registration names ETF trust and DIA market symbol |
| IWM |2009-10-29|2009-11-02| Bank424B2 names IWM and describes its underlying fund as an ETF; not fund-issuer identity verification |

`review_task_fund_sources.py` reproduces these four fixed observations against
retained primary bytes. Bank registrant19617 belongs to the source filing and
must not be assigned to IWM. The observation is limited to dated product-type
evidence; it does not establish financial issuer identity or historical price
linkage. Likewise, QQQ's old ticker cannot be silently overwritten in evidence.
All records leave whole-history metadata admission and inferred validity intervals
false. The two-session reconstruction remains an explicit research assumption.
No Colab/optimizer/model operation or retired-test evaluation occurred.
