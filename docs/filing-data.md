# Causal financial filing research

Status: outcome-free coverage audit preregistered on 2026-10-24. The first
2020q1 archive is downloaded; structural acceptance failed on future NUM period
labels. No financial features, model fitting or Colab allocations have been
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


## First actual archive inspection

The first official2020q1 archive is retained under SHA256
`48ed9834c66d565c21130d087eae941b25770e2d7cc8351d17652612289fcd67`,
96,584,608 compressed bytes, with its acquisition receipt. The request identity
is now configured privately. The ZIP includes an optional exact `readme.htm`;
the reader now accepts that name without extraction and rejects other unexpected
members. All5,816 submission acceptance strings end in `.0`. Strict optional
fractional-second parsing preserves these source strings without timezone guesses.

44 focused checks pass, including these actual-format regressions. Full structural
acceptance did **not** pass: NUM contains period labels in2024 onward and the
reader stopped under its existing cutoff rule. No normalized financial dataset
was created. Source-period metadata needs inspection to distinguish known-at-filing
future disclosures from date errors; no future numeric record may silently pass
into model inputs. The source ZIP and failed-run evidence remain retained.

## Forward-period quarantine and resolved first exception

Original accession`0001062231-20-000009`, filed2020-02-14, has an inline XBRL
context for2024-01-01 through2024-12-31 explicitly carrying
`srt:ScenarioForecastMember`. Its amortization amount is a forward disclosure,
not a realized2024 result. The original HTML is retained by checksum with its
receipt; [resolution evidence](../results/filing-source-review-2026-10-24/forward-period-resolution.json)
records metadata without the amount. Numeric period and public availability are
different concepts. This resolves that exception's meaning without proving
point-in-time fidelity for the full archive.

The initial strict audit failure remains recorded. The default reader still
rejects post2023 NUM dates. An [explicit addendum](../configs/experiments/filing_forward_period_addendum.json)
adds a separate inspection mode:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_sec_archive.py \
  --archive /path/to/2020q1.zip --quarter 2020q1 \
  --quarantine-forward-periods \
  --output artifacts/filings/2020q1-quarantined-structural.json
```

Version3 reports which policy was used. Quarantine excludes **every** NUM period
later than that accession's SUB reporting period, including forward periods before
2024; it is not a one-record exception. Excluded amounts do not enter numeric
comparisons or model preparation. Counts and at most20 metadata examples are
reported, while original records stay in the immutable ZIP. Dates must still be
valid; retired SUB filing, acceptance and reporting dates remain rejected.

A period comparison does not identify every forecast or make remaining facts
model-ready. Concept/scenario/duration selection, original-vintage verification,
identity mapping and publication evidence remain separate mandatory work. The
mode never enables financial inputs or training.48 focused checks pass, including
quarantine of both2020 and2024 periods, amount exclusion and retained strict
retired-submission rejection. Actual archive results are recorded separately.

## Current identity discovery versus historical evidence

SEC's current`company_tickers.json` supplies candidate CIKs for all107 research
stocks (the four ETFs are excluded from this company-data discovery). The raw
snapshot/receipt are retained under SHA256
`93050f93c3540af9af80a05885d562fcf4e7de4e8e5acce0944a2891bcfbec48`.
This is current metadata, never a historical observation or verified identity map.

The reusable candidate audit compares these hints with a retained structural
report without enabling model inputs:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_sec_identity_candidates.py \
  --discovery artifacts/filings/current-identity-discovery.json \
  --structural artifacts/filings/2020q1-quarantined-structural.json \
  --output artifacts/filings/2020q1-candidate-coverage.json
```

For2020q1,102/107 current candidates have qualifying10-K/10-Q/amendment records.
APA,BLK,IBM,KR,XOM do not. These counts are not historical coverage acceptance:
quarter timing, source extraction gaps and registrant reorganizations differ.
Report current candidate ambiguity/shared CIKs explicitly; ticker alone is not a
security identifier. No positive join sets a historical-verification flag.

[APA's dated2021 reorganization filing](https://www.sec.gov/Archives/edgar/data/1841666/000119312521063695/d127090d8k12b.htm)
explains why today's parent must not be backdated to2020. Meanwhile IBM's
[original2019 annual filing index](https://www.sec.gov/Archives/edgar/data/51143/0001558370-20-001334-index.htm)
shows a filing exists even though its current CIK has no qualifying record in the
inspected archive. Investigate source coverage and original filings independently;
never fill missing records with current statements or silently change CIKs.

51 focused checks pass. Identity tests require current metadata to remain
unverified after matching, expose shared/absent candidates and keep unrelated
forms from filling required financial coverage. Candidate counts do not establish
historical continuity, per-quarter completeness or any predictive value.

## Dated issuer evidence for missing current candidates

Original2019 annual filings published in2020 explicitly declare issuer CIK and
trading symbols for three candidates missed by today's CIKs:

| Stock | CIK in original2020 filing | Accession |
| --- | --- | --- |
| APA | 6769 | 0001733037-20-000004 |
| BLK | 1364742 | 0001564590-20-007807 |
| XOM | 34088 | 0000034088-20-000016 |

The new inline-identity inspector verifies original HTML checksum, records explicit
DEI facts and their contexts, requires exact expected ticker/issuer matching and
preserves alternative securities. For example,APA also lists`APA/29` and BlackRock
lists`BLK25`; substring matching or dropping alternate classes is unsafe. The
inspector rejects ambiguous or malformed issuer facts and does not infer a
validity interval from a dated document.56 focused SEC checks pass.

Original HTML and acquisition receipts remain under content hashes in
`data/raw/sec-filings/`. The [dated identity evidence](../results/filing-source-review-2026-10-24/2020q1-dated-identities.json)
links each source URL, checksum, filing metadata and observed DEI facts. It proves
those three document-level pairings, not full2010–2023 ticker continuity or
same-security financial-history eligibility. It does not authorize replacing
current IDs globally or applying evidence before public availability. Financial
inputs remain disabled. IBM/KR archive absences and wider chronological coverage
still require investigation.
