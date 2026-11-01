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

## Deterministic stock master selection and bounded batches

`type_sources.py` and `prepare_task_stock_sources.py` query the retained2009q2–
2010q1/2012q4/2013q1 full master indexes, preserving current/predecessor CIK
candidates as discovery-only. Prefer the latest eligible original10-K, else the
latest original10-Q. A selection-stage two-session estimate uses the filing date;
actual original-index acceptance/filing metadata must pass again after collection.
No amendment substitution, return ranking or metadata admission occurs.

The [selection](../results/task-conditioning-2026-10-31/stock-master-selection.json)
finds105/107 stocks with eligible original candidates:36 already have a collected
document observation,69 need primary HTML collection, and ABBV/DIS lack an
eligible candidate in this scope. Missing XBRL entries are not evidence of missing
historical filings. ABBV needs registration/spinoff evidence and DIS needs an
explicit older issuer candidate; neither gets a guessed mapping.

`collect_task_stock_batch.py --batch N --output NEW_PATH` allows only nine fixed
batches over those69 candidates, at most eight cases/sixteen index-primary
operations each (the last batch is smaller). Claims prevent repeated batches;
a global stop record prevents continuing after any failed operation. Actual
index/master dates and conservative availability must agree before primary
collection. Preserve retrieved partial sources and failed cases. Batches select
only by the frozen source plan, never by returns or later model results. Complete
collection still requires source-content/identity review before model inputs.

## First stock master batch and compact review hints

Batch0 completed APA/APD/C/CAT/CL/COP/COST/CSCO: sixteen index/primary operations,
21,826,972 raw bytes, with original publication metadata available before each
training start. Seven primary files have a conservative registered-common-equity
cover hint; five have an explicit expected-ticker clause near common-equity text.
These are review candidates, not automatic ticker/security or interval admission.

`html_stock_type.py` checks bounded SHA256 source bytes, removes script/style
text, searches registration/common-equity cover context, and requires explicit
symbol wording plus the exact canonical ticker near common-equity text. A company
abbreviation, drug code or different share-class suffix cannot satisfy that
literal-ticker rule. Conservative misses remain unresolved, including cases where
registration details appear beyond the fixed cover window. The helper retains
only a short source review excerpt and always leaves historical identity, input
enablement and metadata admission false. It does not infer an issuer from a match.

`audit_stock_html_batches.py --batches 0,1,... --output NEW_PATH` processes only
explicit complete batches sharing the original selection digest. It rejects
repeated symbols, partial source records or changed hashes. Review against the
original document is still required. Remaining batches1–8 follow the original
fixed69-source plan; any failed batch stops the sequence without automatic retry.

## Stopped master-source sequence and joint-filer cause

Batches0–3 completed32 primary documents; batch4 retained MLM/MRK/MSFT before
stopping at NEE's index resolution. No batch5–8 started, no operation was retried,
and the global stop/individual claims remain.35 complete index/primary pairs plus
the failed case's index retain90,366,834 bytes across these attempts. All declared
protected source and cumulative usage-ledger hashes remain unchanged.

[Complete-batch hints](../results/task-conditioning-2026-10-31/stock-master-complete-batch-hints.json)
find28 cover hints and13 literal ticker hints across32 documents.
[Partial-batch hints](../results/task-conditioning-2026-10-31/stock-master-batch-4-retained-hints.json)
cover only the three retrieved documents, explicitly retaining the stopped/partial
status; they do not accept the batch. No classification was admitted.

[Joint-filer inspection](../results/task-conditioning-2026-10-31/nee-joint-filer-inspection.json)
identifies a deterministic endpoint-authority mismatch, not an access rejection.
The retained original NEE index was requested under FPL Group CIK753308, but its
sequence1 primary link uses Florida Power & Light CIK37634. Both appear under the
same10-K accession in the official master. The strict resolver rejected the link
before issuing the primary request. A future explicit joint-filer resolver must
verify both records against immutable master bytes, preserve the failed attempt
and use a separately recorded reviewed resume. Do not loosen arbitrary-CIK URL
checks or clear the existing stop to repeat the old batch automatically.

The [two missing-source candidates](../results/task-conditioning-2026-10-31/missing-stock-source-candidates.json)
are now located in cached masters: ABBV's January2,2013 spinoff8-K and an older
Disney CIK1001039 annual filing from December2,2009. These are source selections,
not verified historical identity links; neither original primary has been
collected or substituted into the111-symbol metadata manifest by this discovery.

## Master-verified joint-filer resolver

`primary_html_joint` adds an explicit alternative to the unchanged strict
`primary_html` API. It verifies the bounded original index, exact sequence1 form,
SEC host, same accession and safe filename, then reads checksum-verified master
bytes for both declared and target-directory registrants. Both must have exactly
matching accession/form/filing-date rows, and the target directory must itself
be listed. Missing counterparts, another accession/date, malformed endpoints or
unrelated CIKs reject. Joint-source authority does not verify a traded-security
identity or allow financial issuer joins.

The [actual NEE resolution](../results/task-conditioning-2026-10-31/nee-joint-resolution.json)
passes against the retained index and2010q1 master. Original failed batch, global
stop and cumulative ledger hashes remain unchanged; this check issued no request
and did not resume collection. Any resumed source work must record a new claim
and retain the old failure instead of deleting/reusing the original batch claim.

## Reviewed source-only resume plan

`resume_task_stock_sources.py --prepare --plan NEW_PLAN` verifies the diagnosed
original stop, unchanged69-source selection,35 retained primaries and the actual
NEE joint resolution. The [concrete v2 plan](../results/task-conditioning-2026-10-31/stock-reviewed-resume-plan-v2.json)
contains only34 unfinished cases, preserving original batch groups5/8/8/8/5 and
at most68 index/primary acquisition operations. NEE's retained index is reused
from the immutable cache. Already retrieved primary files are not downloaded
again. Original stop, claims, source receipts and cumulative ledger remain
protected; preparation performs no request.

Explicit execution requires the exact plan to equal a fresh preflight. A separate
one-use v2 claim and stop record govern the resume; the original global stop is
never cleared. All primary links use master-verified authority, and original
filing/acceptance metadata must again satisfy first-training availability. Each
retrieved case publishes a progress receipt. Failure stops the resume without
retry and retains both generations of evidence; temporary progress writes clean
up on failure. This is a reviewed compatibility correction, not reuse of a failed
claim or an extension of any model experiment. No metadata admission or Colab
allocation is performed. Mocked tests cover changed-plan rejection, persistent
old stops and single-attempt failure with no temporary-file residue.

## Completed reviewed resume and independent combined audit

The source-only v2 resume completed34/34 unfinished primaries in68 acquisition
operations, reusing the retained NEE index from cache. New raw bytes total
88,940,815. No already retrieved primary was downloaded again. Original batch4
remains stopped, all v1 claims/global stop remain unchanged, and no automatic
retry or Colab allocation occurred.

[Resume receipt](../results/task-conditioning-2026-10-31/stock-reviewed-resume-result-v2.json)
and [independent combined audit](../results/task-conditioning-2026-10-31/stock-source-combined-audit.json)
cover the exact69-source selection across35 original and34 resumed primaries.
The verifier independently recomputes original index/master authority, byte counts,
checksums, publication metadata and conservative availability. It validates NEE,
SO and SRE's three joint-filer directories without assigning a subsidiary issuer
to a stock's price series.164 protected inputs remain unchanged. The collection
scope is complete; historical classification admission is not.

Across69 primary files, the conservative helper finds60 common-equity cover hints
and31 explicit expected-ticker clauses. These counts are discovery aids, not69
verified historical ticker/type intervals. Source aliases, missing literal clauses
and initial-versus-whole-history evidence still require review. ABBV/DIS's two
special event/predecessor candidates remain outside this completed selection.

`audit_stock_source_sequence.py --output NEW_PATH` reproduces the combined audit
from both generations of receipts without relabeling the original failed batch.
Use the recorded source revision: source-code changes can invalidate current-tree
protected-input checks even while the retained original receipts remain valid.
The [cleanup receipt](../results/task-conditioning-2026-10-31/source-progress-cleanup.json)
removes only the142,610-byte progress file after byte equality with the complete
final receipt and all164 protected inputs are reverified. Every raw source,
scientific receipt, claim and original stop is retained.

## Two exceptional stock source selections

`collect_task_exception_sources.py` fixes ABBV's January2,2013 spinoff8-K and
Disney's December2,2009 older-registrant10-K. It verifies the two exact selected
rows against checksum-bounded cached master indexes without broadening the
financial master reader's default form scope. Each case collects only its original
index and sequence1 typed primary, at most four operations total. Original dates
must agree with master metadata, and conservative availability must precede each
first training session. One persistent claim prevents automatic retry; partial
or rejected cases remain. Collection alone does not verify either company's
historical traded-security mapping or admit metadata.
