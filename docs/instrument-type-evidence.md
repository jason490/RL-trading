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
