# Causal financial filing research

Status: outcome-free coverage audit preregistered on 2026-10-24. No financial
features, provider downloads, model fitting, or Colab allocations have been
performed for this family. The [fixed audit specification](../configs/experiments/filing_coverage_audit.json)
is separate from any later training experiment. Existing failed price-feature
and model families remain closed; this proposal does not establish improved return.

## Source and scope

Use the SEC's quarterly [Financial Statement Data Sets](https://www.sec.gov/dera/data/financial-statement-data-sets.html)
for 2009q2 through 2023q4. Warmup filings may precede the 2010 training start;
2024 onward remains retired and excluded. The SEC documentation describes
submission-level records and financial values, including amendments. It also
warns that the extracted tables omit some filing metadata and may contain
inconsistencies. Preserve original accession documents for source verification.
Read the [table definitions](https://www.sec.gov/dera/data/fsds.pdf).

SUB identifies submissions by accession (`adsh`) and issuer (`cik`). NUM joins
on accession and distinguishes concept, taxonomy version, period end, duration,
unit and co-registrant. TAG describes concepts; PRE describes presentation.
Joining only on issuer and fiscal period discards the knowledge vintage and is
not acceptable. Keep exact source archive bytes, URL, retrieval UTC and SHA256.
Archive updates/reprocessing must produce a new stored version, never replace one.

## Availability and revision rules

`period` is an accounting period end, not a publication date. SUB provides both
`accepted` and `filed`; SEC documentation says late acceptance can result in a
following-business-day filing date. Verify the timezone and actual dissemination
semantics before implementation. Do not silently interpret a naive timestamp as
UTC or assume acceptance guarantees immediate public availability.

At minimum, exclude same-local-date decisions: use the first eligible exchange
session strictly after both verified acceptance and filing dates, with any longer
verified dissemination delay. Missing or ambiguous timing excludes the record.
This conservative reconstruction is distinct from the date a quarterly ZIP was
published or downloaded. It requires evidence from original filings.

**Never use `prevrpt` as an input or historical eligibility filter.** SEC defines
it using whether the report was subsequently amended: that carries future
information. An amendment is a separate accession available only after its own
publication. The earlier accession remains the earlier knowledge state. A current
Company Facts view must not overwrite that state. Audit how archive reprocessing
may affect historical extraction before claiming point-in-time fidelity.

## Identity and numeric validation

Establish dated issuer-to-instrument mapping with evidence. Today's ticker list
is a discovery aid, not proof of historical identity; account for renames,
reorganizations, multiple share classes and ambiguous CIKs. Report unresolved
mapping, missing years and excluded issuers explicitly. This does not repair the
existing universe's survivorship bias or supply delisting returns.

Audit standard versus issuer-specific concepts, taxonomy changes, currencies,
units, instant versus duration values, duplicate/conflicting values and subsidiary
rows. Do not mix quarterly and year-to-date amounts or substitute superficially
similar concepts. Revenue growth, profitability and leverage are candidate
families only; exact concepts and formulas remain disabled pending the audit.
ETFs need an explicit missing/not-applicable treatment, not fabricated company
financials. A later matched experiment must keep all arms' evaluation rows equal.

## Access and bounded operation

The official [developer guidance](https://www.sec.gov/about/developer-resources)
sets an aggregate maximum of ten requests per second across machines. The future
downloader should default well below this, cache immutable archives, use bounded
retries/timeouts and stop on access rejection. It must require an honest SEC
request identity configured privately as `SEC_USER_AGENT`, never invent a contact
or print it in logs. No identity was configured when this audit was registered.
No new dependency or Colab runtime is needed for the source audit.

## Acceptance before training

Produce the reports enumerated in the specification without model inference,
returns, declining-stock labels, normalization fitting or checkpoint selection.
Inspect coverage by symbol/year/split, not just an aggregate success percentage.
Reject unresolved timing, identity or vintage fidelity. Once source feasibility
is established, freeze exact feature formulas, missingness/staleness handling,
coverage thresholds, train-only preprocessing and matched control construction.
Then preregister a separate bounded Colab comparison using the small recurrent
policy, unchanged rewards/costs and multi-seed/prospective promotion rules.

Do not infer that financial data will improve twenty-session trades. The audit
can fail, and a feasible dataset can still produce an unsuccessful model. Record
both outcomes rather than expanding a feature grid against reused development
returns. See [research operation](research-loop.md) and [results](../results/README.md).

## Offline archive inspection

The implemented structural reader streams the four tables directly from a local
ZIP, without extracting paths or querying a provider. Run from the repository:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_sec_archive.py \
  --archive /path/to/2020q1.zip --quarter 2020q1 \
  --output artifacts/filings/2020q1-structural.json
```

It checks required columns, submission uniqueness, accession references, dates,
finite numeric values and the uncompressed size limit. All records with filing,
acceptance, accounting or numeric period dates in 2024 onward are rejected.
The requested quarter is scope-checked; it is caller metadata, not proof that the
archive came from SEC or belongs to that quarter. Retrieval provenance must be
provided by a future downloader. Acceptance strings remain uninterpreted local
source timestamps. Original and amendment accessions are retained separately;
`prevrpt` is omitted from normalized submission evidence. The source ZIP must be
retained unchanged so all source fields remain recoverable.

Output reports counts, anomalies, archive checksum and submission records. It
explicitly leaves publication, timezone, identity mapping and causal readiness
unverified. It does not yet resolve concept relationships, duration conflicts,
duplicate numeric facts or source revisions, and therefore is not the complete
coverage acceptance report. No model-ready financial dataset is produced.

## Bounded archive acquisition

The downloader is available once an authorized `SEC_USER_AGENT` is configured in
`.env` (quote the application name and contact email as one value) or the process
environment. Do not put the identity in shell arguments or commit it.

```bash
PYTHONPATH=src .venv/bin/python scripts/download_sec_archive.py --quarter 2020q1
```

Only 2009q2–2023q4 quarter URLs on the fixed SEC endpoint are accepted. Redirects
are rejected. At most three attempts are made, with a two-second pause before
each; HTTP403/429 and other non-transient statuses stop immediately. A 120-second
operation deadline is checked while waiting for the local lock and between reads;
a blocking transport call can add up to its bounded20-second timeout. The256MB
compressed limit applies with or without a Content-Length header. Responses must
have a ZIP signature; the separate structural reader performs table validation.

A shared cache-root file lock serializes local acquisitions. It does not
coordinate other machines or SEC clients: use one downloader and coordinate
aggregate traffic before parallelizing. Cache reuse verifies receipt hash,
archive hash and size before returning without a request. Corruption stops rather
than silently replacing retained evidence. Acquisitions store raw ZIPs by SHA256
and immutable receipt files containing URL, quarter, byte count, retrieval UTC
and attempt count; the identity is excluded. Temporary files are removed on
handled success/failure. A hard process kill may leave a temporary file, which
must not be mistaken for a verified archive.

Verification:32 synthetic offline reader/downloader checks pass, including actual
CLI report writes, immutable cache reuse, receipt tampering, short/oversized
responses, bounded retry, access rejection and redirect rejection. No actual SEC
archive has yet been downloaded or audited; causal readiness stays false.

## Source-schema review: December2024 reprocessing

The [SEC archive page](https://www.sec.gov/dera/data/financial-statement-data-sets.html)
reports that historical archives were reprocessed in December2024 and that NUM
added `segments`. Use its linked [current specification](https://www.sec.gov/files/financial-statement-data-sets.pdf),
not solely the older `/dera/data/fsds.pdf`. The frozen initial audit scope stays
unchanged; this source review adds requirements before observed financial data.

Structural report version2 records `legacy-v1` or `segments-v2`. Numeric keys
include accession, concept, taxonomy, period, duration, units, segments and
co-registrant. Distinct segment values must not be collapsed into issuer totals.
The reader counts identical duplicate keys and conflicting values separately,
without choosing one for model input. Decimal representations of the same value
agree; a missing value differs from zero. Comparisons use temporary SQLite disk
storage rather than an in-memory set of every numeric key. This is not yet a
concept-resolution or financial-feature pipeline.

The [SEC FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions)
explains that acceptance is distinct from public availability, lag is variable,
and no first-publication timestamp exists in the header. Its terminology labels
acceptance time EST; daylight-saving interpretation still needs explicit review
before timezone conversion. The [access guide](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
adds late dissemination and post-acceptance correction behavior. Conservative
session lags remain reconstruction assumptions, not proof of original public
availability. [Review evidence](../results/filing-source-review-2026-10-24/README.md).

Current verification:37 focused reader/downloader checks pass. Actual coverage,
public availability and historical-vintage fidelity remain unverified.
