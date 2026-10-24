# Causal financial filing research

Status: the registered financial/master catalogs and concept-presence audits are
complete. Original-filing prototypes work, but a complete causal financial dataset
has not been accepted. Only43.7% of assessed financial-table candidates contain
all required current tags; this is not an arithmetic or stock-price readiness rate.
No financial inputs, model fitting or Colab allocations have been enabled for this
family. The [fixed audit specification](../configs/experiments/filing_coverage_audit.json)
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

## IBM/Kroger archive absences: source inspection

Kroger's original annual filing`0001558370-20-003501` was published2020-04-01,
placing it in Q2, not Q1. The retained2020q2 archive
SHA`bd796ea10c88e21205da013f9c3f6a20b2b14d3de94f0075e9f529e0ef9a3543`
contains that accession plus its June quarterly filing. It also contains IBM's
April quarterly filing. This adjacent-quarter inspection reads SUB metadata only;
it is not a full NUM/schema/conflict audit of Q2.

IBM's2019 annual filing exists in February2020; its main HTML has none of the
selected assets/liabilities/revenue/net-income concepts inspected, while its
Exhibit13 contains them. Main HTML, exhibit and original indexes are retained
with hashes. This establishes that statements are available in the accession
outside the main document; it does not establish the exact internal SEC archive
extraction cause. Never interpret the missing ZIP submission as no annual report,
and never merge an alternative extraction silently. A separate original-filing
import contract and equivalence/coverage checks are needed before supplementation.

Original source dates and archive labels can differ: Kroger's index gives
2020-02-01 as report period end; the SUB row labels it20200131. The original
accepted timestamp has seconds09:01:18, while SUB has09:01:00. Preserve both;
do not silently relabel or claim exact timestamp/period parity. Financial growth
and duration logic must verify original periods and fiscal-year context, not
assume quarter/fiscal-year identity from rounded archive labels.

The original main IBM filing also explicitly matches CIK51143 and tickerIBM;
this is dated-document evidence only. [Absence/source resolution](../results/filing-source-review-2026-10-24/archive-absence-resolution.json),
[IBM identity](../results/filing-source-review-2026-10-24/ibm-dated-identity.json)
and [Q2 metadata](../results/filing-source-review-2026-10-24/2020q2-ibm-kr-sub.json)
retain the scope and uncertainty. No financial dataset or training is enabled.

## Restricted original XBRL financial extraction

`sec.facts.extract_facts` now reads the original accession's SEC-extracted XBRL
instance, including IBM's exhibit statements. It is a source-review tool, not a
financial-feature assembler. It requires source SHA256, accession, expected issuer,
original report end and filing date. A mismatched original DEI report end rejects
an archive's rounded month-end substitute. Filing dates from2024 onward reject.

Only a small explicit standard US-GAAP money-concept set is inspected: assets,
liabilities, net income and specified revenue concepts. Original namespace/concept,
context/unit references, exact instant or duration, decimal text and declared
precision are retained. No float conversion, scaling refit or concept priority is
applied. Every matching fact is kept; duplicate/conflicting candidates are not
silently collapsed into a preferred value. Growth, margin and leverage features
remain disabled until a separate concept/duration/vintage specification passes.

Only unqualified expected-issuer, non-forward, simple USD facts receive source
amounts in the private extraction. Segment/scenario, forward, nil, wrong-issuer
and non-USD rows carry explicit exclusions and no amount. This conservative
filter is not a guarantee that all forecast or accounting ambiguities are solved.
It does not infer availability or certify ticker continuity. Orphan references,
invalid/nonfinite eligible values, namespace rebinding, DTD/entities, malformed
periods and source checksum mismatches reject. No external schemas or entities
are fetched. Source files are bounded at20MB and retained unchanged.

Example (substitute the recorded source hash/path):

```bash
PYTHONPATH=src .venv/bin/python scripts/extract_sec_original_facts.py \
  --source data/raw/sec-xbrl/SHA/instance.xml --sha256 SHA \
  --accession 0001558370-20-003501 --cik 56873 \
  --report-end 2020-02-01 --filed 2020-04-01 \
  --output artifacts/filings/kr-original-money-facts.json
```

Actual source inspection found IBM237 selected rows,211 excluded qualified rows
and26 unqualified USD rows; Kroger54 selected,21 excluded and33 unqualified USD
rows. Kroger retains exact2019-02-03–2020-02-01 duration endpoints. These counts
show extraction works on two sources, not chronological coverage, financial
predictiveness or causal training readiness. Raw extracted amounts stay in
ignored artifacts; the [public summary](../results/filing-source-review-2026-10-24/original-extraction-summary.json)
contains metadata/counts only.71 focused SEC checks pass, including original
period mismatch, precision, exclusions, duplicate retention and parsing rejection.

## Bounded chronological archive collection

The sequential collector follows the original59-quarter2009q2–2023q4 scope:

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_sec_archives.py \
  --state .runtime/sec-coverage-collection.json
```

It reuses verified immutable cached archives, permits at most8GB of retained
compressed bytes in the planned range, keeps a2GB free-disk reserve plus one
maximum-sized request, and has a one-hour inclusive deadline with140 seconds
reserved before starting each source operation. Per-request byte/timeout/retry
limits remain in the downloader. It allocates no Colab runtime and never trains.

An atomic progress journal records protocol hash, deadline, current quarter,
completed archive identities/sizes and terminal status without request identity.
A controller lock prevents duplicate use of the same journal. Existing journals
are rejected rather than automatically restarting an interrupted/failed run;
review terminal state, retained files and the provider failure before any new
explicit collection window. Verified cache reuse avoids repeating prior downloads.
`downloaded_only` means source bytes were collected, not that all tables passed
inspection or financial model inputs are ready.74 focused SEC tests pass.

## Chronological submission inventory and coverage visualization

The full registered collection completed:59 archives,4,613,634,809 compressed
bytes,601.7 seconds. All source hashes were verified for the SUB inventory.
It contains366,856 unique submissions,342,981 qualifying10-K/10-Q/amendment
submissions across14,100 issuers. No duplicate/conflicting cross-archive accession
was observed. This is submission metadata coverage, not all NUM validation.

The first strict inventory stopped on three blank reporting periods, retained in
[diagnostic evidence](../results/filing-source-review-2026-10-24/invalid-sub-date-diagnostic.json).
They belong to S-1,486BPOS and N-2, outside the registered financial-form family.
The inventory now reports missing periods for such non-periodic forms without
inventing dates. Qualifying financial forms still require valid report periods;
regression tests enforce that distinction. Every observation remains source-only,
with publication/historical identity/financial-readiness flags false.

Reproduce from the retained collection journal and discovery source:

```bash
PYTHONPATH=src .venv/bin/python scripts/inventory_sec_submissions.py \
  --state .runtime/sec-coverage-collection.json \
  --output artifacts/filings/chronological-sub-inventory.json
PYTHONPATH=src .venv/bin/python scripts/report_sec_coverage.py \
  --inventory artifacts/filings/chronological-sub-inventory.json \
  --discovery artifacts/filings/current-identity-discovery.json \
  --output artifacts/filings/candidate-coverage.html \
  --summary results/filing-source-review-2026-10-24/candidate-year-counts.json
```

The offline Plotly heatmap displays current-CIK candidate counts by filing year,
not certified stock history. Source hash/size, registered range, terminal journal,
incomplete scope, duplicate/conflicting accessions and retired rows are checked
or reported. Counts distinguish filing year from the archive's period-year labels.
The metadata-only inventory reads no financial amounts. Candidate summary and
[collection evidence](../results/filing-source-review-2026-10-24/chronological-inventory-summary.json)
are retained in results; full inventory/HTML stay in ignored artifacts.

Examples: today's BLK/XOM CIKs produce zero counts across the range; APA's current
parent begins2021, GOOGL's current registrant begins2015. Earlier source-verified
registrants must not be discarded or globally backfilled without dated evidence.
Zeros can reflect issuer changes, listing timing, source gaps or filing cadence.
Missingness has not been supplied to a model as a trading input.

81 focused SEC tests pass. The HTML/summary reproduce byte-for-byte on repeated
report generation. Collection is terminal; no temporary download files remain.
All59 original ZIPs/receipts and original-filing evidence are retained, with no
raw evidence deleted. Last checked Colab balance349.02, rate0/assignments0;
collection allocated no runtime and consumed no Colab units. Next: issuer lineage,
original-period/concept equivalence and causal financial feature acceptance.

## Source-backed succession and daily availability policy

The first two [issuer succession records](../configs/filings/issuer_evidence_v1.json)
are retained with original event/index HTML checksums:

| Research symbol | Predecessor CIK | Successor CIK | Effective event date | Event type |
| --- | --- | --- | --- | --- |
| GOOGL | 1288776 | 1652044 | 2015-10-02 | Holding-company reorganization |
| LIN | 884905 | 1707925 | 2018-10-31 | Business combination |

[Alphabet's filing](https://www.sec.gov/Archives/edgar/data/1652044/000119312515336577/d82837d8k12b.htm)
documents corresponding share-class conversion. The separate
[Nasdaq notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2015-156)
also identifies Class A/GOOGL and Class C/GOOG. This does not resolve the earlier
2014 Google ticker/share-class transition or certify Yahoo's entire adjusted
history. [Linde's filing](https://www.sec.gov/Archives/edgar/data/1707925/000119312518313073/d643979d8k.htm)
combines Praxair and Linde AG under Linde plc; do not infer economic equivalence
of predecessor/successor financial ratios. Linde's pre-merger registrant existence
in2017 is not evidence that those statements describe the stock's earlier
operating history. An unrelated company named Alphabet Holding is present in
inventory too: name matching alone is unsafe.

The issuer query is metadata-only. It filters evidence by both effective session
and available timestamp, returns unknown before the first eligible observation,
and requires an explicit predecessor link for a changed CIK. A later subsidiary
observation cannot silently replace the public parent. Queries never enable
financial features or certify price links/comparability. Full issuer histories
and price-security joins remain incomplete. [Actual query evidence](../results/filing-source-review-2026-10-24/issuer-query-inspection.json).

The [fixed daily availability policy](../configs/experiments/filing_availability_policy.json)
uses the second XNYS open strictly after the later original source-local filing
or acceptance date. It retains original naive acceptance strings and avoids
inventing a UTC interpretation of their hours. Holiday/weekend/DST-aware calendar
opens produce reconstructed timestamps: Alphabet2015-10-06T13:30Z and
Linde2018-11-02T13:30Z. Effective and availability dates remain distinct.

This lag is an explicit research assumption. SEC says dissemination latency is
variable and no first-publication timestamp exists; the rule cannot guarantee
actual first-publication timing or solve post-acceptance corrections. It never
sets exact-publication verification true. Source dates from2024 onward reject;
a late2023 filing whose reconstructed availability falls in2024 is unavailable
to all2023 decisions.90 focused SEC tests pass. No financial dataset or training
has been promoted by these evidence queries.

## Fixed coherent-accession financial snapshot rules

[Feature rules](../configs/experiments/filing_feature_rules.json) are fixed before
any price join or performance scoring. The optional pure builder in
`features/fundamentals/snapshot.py` computes three reported ratios: revenue growth,
net margin and liabilities/assets. It is not registered in the default market
schema, does not fit normalization, and never enables financial inputs itself.

For annual forms, current flows must end at the original reporting date and span
330–380 inclusive days. Quarterly forms permit28–330 days, accommodating16-week
quarters and other fiscal calendars. Choose the longest common revenue/net-income
interval within those bounds; do not substitute a shorter interval when selected
facts conflict. This is the longest reported subannual interval, not an assertion
that every source represents a calendar quarter, YTD or trailing-twelve-month
figure. Assets/liabilities must have the exact same current instant date.

Prior revenue must have exactly one comparable interval: both endpoints350–380
days earlier and duration difference at most7 days. Current/prior must share a
supported revenue concept. Duplicate or competing total-revenue facts require
exact decimal agreement; otherwise arithmetic is unready. No priority/summing
rule silently combines potentially different accounting measures. Current/prior
revenue and assets must be positive, liabilities nonnegative; net income may be
negative. Every amount comes from one accession, preserving that publication's
own comparative/restatement vintage without rewriting earlier observations.

The output distinguishes `arithmetic_ready` from financial/publication/price-link
readiness. Excluded facts cannot influence ratios; mixed accessions, issuers,
units, unexcluded forward facts and nonfinite values reject. Conflicts, missing
comparables and invalid denominators produce an explicit unready reason and no
feature values. There is no clipping, imputation, currency conversion or fitting.
This is reported financial growth, not organic or merger-adjusted performance.

Actual IBM/Kroger original annual snapshots both pass arithmetic inspection,
with exact ratio equality against an independent rational-arithmetic calculation.
IBM uses2019-01-01–2019-12-31; Kroger uses2019-02-03–2020-02-01. The
[historical inspection](../results/filing-source-review-2026-10-24/financial-snapshot-inspection.json)
records source/rule/code hashes, ratios and scope. These numbers describe company
filings, not model trading returns.108 focused SEC/financial checks pass. No
price join, financial dataset promotion, training or normalization was performed;
chronological issuer/price eligibility, availability and feature coverage remain
required before a bounded model experiment.

## Source-snapshot as-of history and staleness

[As-of rules](../configs/experiments/filing_asof_rules.json) fix source-history
selection before any price join/performance scoring. `ResearchFinancialHistory`
reconstructs availability from original filing/acceptance dates, filters by exact
requested issuer and decision timestamp, then chooses latest reporting end and
latest available accession at that end. Equally ranked conflicting accessions
reject. An older-period amendment stays its own vintage and does not displace a
newer reporting period or recompose another accession's ratios.

A newly available unready snapshot blocks fallback to a convenient older value.
Annual reporting periods expire after450 calendar days, quarterly after180,
measured from the original reporting end rather than amendment publication.
Amendments therefore cannot refresh stale financial information. Unknown issuers,
unavailable snapshots and stale/unready sources return no values. Dates from2024
onward are excluded in this research reader. Input records and returned arrays
are copied to prevent caller mutation from changing later queries.

**A complete supplied filing ledger is still required.** The reader cannot detect
an omitted filing it was never given. Include all qualifying sources, including
unready snapshots; otherwise apparent source availability is only a partial-ledger
upper bound. `source_ready` is a query result for that supplied ledger, not a
publication guarantee, a stock/issuer verification, a complete-coverage claim or
permission to enable model inputs. Every result leaves financial inputs disabled
and historical price linkage unverified. No full-universe coverage percentage is
claimed from the current two prototype source snapshots.

Actual IBM/Kroger index hashes and timestamps verify no value one second before
the reconstructed boundary, correct values at the boundary and no value after
expiry. IBM boundary is2020-02-27T14:30Z; Kroger2020-04-03T13:30Z.
[Source-backed inspection](../results/filing-source-review-2026-10-24/financial-asof-inspection.json)
records incomplete catalog status explicitly.116 focused checks pass, including
future-amendment isolation, old-period amendment precedence, no fallback,
immutability, ambiguous-accession rejection and unchanged period-age expiry.
Next: complete the original-filing cohort ledger and verify issuer/price joins
before measuring actual training/development coverage. No financial training yet.

## Original-source candidate ledger and early-taxonomy preflight

The reproducible candidate builder reads the59 retained financial-table SUB
archives and creates6,161 accession targets across112 current/predecessor CIKs.
The fixed[discovery scope](../configs/filings/cohort_discovery_v1.json) includes
all107 research stocks plus five explicitly source-identified predecessor IDs;
it does not use fuzzy company-name matching or certify issuer/price intervals.
Private/subsidiary filings may remain collection candidates and must never be
assigned to public shares without the separate dated lineage gate.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_sec_original_ledger.py \
  --state .runtime/sec-coverage-collection.json \
  --discovery artifacts/filings/current-identity-discovery.json \
  --output artifacts/filings/original-candidate-ledger.json
```

The ledger retains archive hashes, raw period/publication labels, fiscal metadata,
original instance filenames and fixed SEC URLs. Source hashes, unique accessions,
complete registered quarter range and safe instance basenames are checked.
No original reporting date is invented from an archive label. The
[compact evidence](../results/filing-source-review-2026-10-24/original-candidate-ledger-summary.json)
explicitly says the **full filing catalog is incomplete**: IBM's known2019
annual accession is omitted by those financial-table ZIPs.

A compressed[SEC Q1 master index](https://www.sec.gov/Archives/edgar/full-index/2020/QTR1/master.zip)
preflight recovers the missing IBM accession. The ZIP is3,809,202 bytes and
expands to29,470,320 bytes, avoiding the earlier20MB raw-index request cap.
Raw ZIP/receipt and[inspection](../results/filing-source-review-2026-10-24/compressed-master-inspection.json)
are retained. Full quarterly indexes must be compared across the registered range
before calling the accession ledger complete. These indexes describe filing
metadata; they do not provide facts or guarantee original publication vintages.

Early original IBM/Kroger2010 filings use legacy`xbrl.us` US-GAAP/DEI namespaces.
The extractor and snapshot validator now support those namespaces explicitly,
while mixed US-GAAP taxonomy versions remain unready without an explicit mapping.
Both early originals extract successfully, but their arithmetic is **unready**
under the unchanged feature definitions: IBM uses`ProfitLoss` and Kroger goods
sales tags rather than the required comparable total concepts. No alias was
silently broadened or added to force an arithmetic pass. The
[early inspection](../results/filing-source-review-2026-10-24/early-original-source-inspection.json)
retains that negative outcome. Source-concept equivalence and cohort coverage
must be audited before original collection is scaled or a financial model runs.
121 focused source/financial checks pass. No financial inputs, fitting or new
Colab allocation/units were introduced by these preflights.

## Bounded compressed master-index collection

The collector also supports the separate registered master-index source:

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_sec_archives.py \
  --source-kind master --state .runtime/sec-master-collection.json
```

It uses fixed SEC quarterly master.zip endpoints over2009q2–2023q4, a separate
protocol hash/cache root,10MB per ZIP/1GB total compressed limits and the same
one-hour deadline, operation reserve, private identity and immediate403/429 stop.
Existing financial collection behavior remains unchanged. Verified cache entries
must respect the requested byte limit too; cache reuse cannot bypass a smaller
limit. Collection still does not parse or certify original financial statements.

## Full master-catalog comparison completed

All59 compressed master indexes are retained:186,760,088 bytes collected in
157.9seconds, including reuse of the verified preflight ZIP. The comparison reads
14,876,676 index rows and preserves `(registrant CIK, accession)` rather than
assuming a filing belongs to only one issuer. The filing-directory CIK is retained
separately; it need not equal the registrant in a joint filing.

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_sec_master_catalog.py \
  --state .runtime/sec-master-collection.json \
  --discovery artifacts/filings/current-identity-discovery.json \
  --financial-ledger artifacts/filings/original-candidate-ledger.json \
  --output artifacts/filings/master-candidate-catalog.json \
  --summary results/filing-source-review-2026-10-24/master-catalog-comparison.json
```

For the same112 candidate CIKs/forms, master metadata identifies6,412 issuer/
accession pairs,6,410 unique accessions. All6,161 financial-table pairs are found;
251 additional pairs correspond to249 additional unique documents. Two joint
Google/Alphabet2016 accessions explain the difference: they register both old
Google and new Alphabet. Do not count co-registrant links as extra independent
statements or assign private Google filings to public Alphabet shares.

The known omitted IBM annual is recovered. Six matched pairs have differing
filing dates between master and financial-table metadata; their forms agree.
Every discrepancy is retained, not silently resolved to a convenient earlier
publication date. No cross-quarter duplicate conflict was observed. The
[comparison](../results/filing-source-review-2026-10-24/master-catalog-comparison.json)
contains all deltas; the full paired candidate catalog remains in ignored artifacts.

This completes the index comparison over the registered candidate scope, not an
original-statement collection, historical price/issuer certification or vintage
availability guarantee. SEC indexes can reflect post-acceptance corrections.
Original statements, header dates, concept coverage and financial eligibility
still need validation before a financial model experiment.131 focused checks pass;
full comparison output reproduces exactly on repeated generation. All source
ZIPs/receipts remain retained, no temporary downloads remain, and the master
collector is terminal. Last official Colab balance349.02/rate0/assignments0;
no additional units were consumed. [Retention evidence](../results/filing-source-review-2026-10-24/master-retention-inspection.json).

## Bounded financial-table concept audit

The fixed[concept-audit protocol](../configs/experiments/filing_concept_audit.json)
checks current-period metadata for all financial-table candidates before large
original-document collection. It requires standard US-GAAP tags, USD, empty
segment/co-registrant labels, matching accession/period label and compatible
quarter-count groups. No amounts are retained or used to compute ratios.

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_sec_concept_coverage.py \
  --collection .runtime/sec-coverage-collection.json \
  --ledger artifacts/filings/original-candidate-ledger.json \
  --output artifacts/filings/concept-audit-v1 \
  --state .runtime/sec-concept-audit-v1.json
```

Two local preprocessing workers stream cached tables, with300-second per-archive
checks and2GB expanded NUM limits. The journal records code/protocol/source hashes
and completed quarters. Failure cancels pending work and waits for active workers
to stop before marking cleanup complete; existing journals/outputs are not
restarted automatically. No training or Colab allocation is involved.

Tag presence is not arithmetic, original-context or price coverage readiness.
Archive period labels/quarter counts are approximate; amounts may conflict,
comparables may be absent, and original filings can contain facts omitted from
flat tables. Unsupported operating income, ProfitLoss, goods/services and bank
revenue labels are diagnostic only, never aliases for the fixed feature formulas.
The251 master-only issuer/accession pairs remain unassessed by this table scan.

## Completed concept-presence findings

The two-worker audit streamed148,956,855 NUM rows across all59 quarters in
212.4seconds, then stopped/reaped both workers. All6,161 financial-table candidate
pairs were assessed;251 master-only pairs remain unassessed. Metadata-only findings:

| Required current concept/group | Present candidates | Fraction of assessed pairs |
| --- | ---: | ---: |
| Supported total revenue | 5,259 | 85.4% |
| NetIncomeLoss | 5,283 | 85.7% |
| Assets | 6,125 | 99.4% |
| Liabilities | 3,873 | 62.9% |
| All four, with common flow-quarter group | 2,695 | 43.7% |

Direct Liabilities is the largest individual presence gap. Diagnostic operating
income/ProfitLoss/goods/services/bank-revenue labels can coexist with supported
labels; their counts overlap and must not be summed into failure totals. No
financial feature definition was broadened and no missing amount was imputed.
Source tables can omit facts available in originals, so this is neither an actual
original-snapshot readiness rate nor stock/trading-day coverage. Values, duplicate
agreement, exact comparative periods, vintage and price/issuer links remain gates.

[Summary](../results/filing-source-review-2026-10-24/concept-presence-summary.json)
retains per-year/per-concept counts and quarterly proof hashes. Full metadata rows
and a standalone offline heatmap are in ignored artifacts at
`artifacts/filings/concept-presence-records-v1.json` and
`artifacts/filings/concept-presence-v1.html`. Rebuild with:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_sec_concepts.py \
  --state .runtime/sec-concept-audit-v1.json \
  --reports artifacts/filings/concept-audit-v1 \
  --ledger artifacts/filings/original-candidate-ledger.json \
  --master-summary results/filing-source-review-2026-10-24/master-catalog-comparison.json \
  --output artifacts/filings/concept-presence-records-v1.json \
  --summary results/filing-source-review-2026-10-24/concept-presence-summary.json \
  --html artifacts/filings/concept-presence-v1.html
```

144 focused source/financial checks pass. Reports/HTML reproduce exact bytes from
the retained quarterly results, dependencies match the audit source commit, and
no temporary files/workers remain. [Retention proof](../results/filing-source-review-2026-10-24/concept-audit-retention.json).
Original/raw evidence is retained. No ratio calculation, normalization fitting,
price join, financial input activation or new Colab allocation/units occurred in
this audit. Next validate balance-sheet accounting identities and original concept
equivalence before scaling original extraction or preregistering a training run;
bigger networks cannot resolve missing/ambiguous source facts.


## Original balance-identity audit — 2026-10-24

The separately registered [accounting audit](../configs/experiments/filing_balance_identity_audit.json)
checks all four previously retained IBM/Kroger annual instances. It does not
expand the default money extractor or change financial-feature definitions.
`extract_balance_facts` produces a separate audit schema; `inspect_balance` checks
same-issuer/accession, exact current instant, unqualified USD facts, agreeing
duplicates and one taxonomy. Decimal arithmetic is independent of caller precision.
Missing temporary equity remains unknown; missing liabilities are never inferred.

| Original report end | Issuer | Assets − liabilities − equity including NCI | Assets − liabilities − parent equity |
| --- | --- | ---: | ---: |
| 2009-12-31 | IBM | $0 | $118,000,000 |
| 2010-01-30 | Kroger | $0 | $74,000,000 |
| 2019-12-31 | IBM | −$1,000,000 | $143,000,000 |
| 2020-02-01 | Kroger | $0 | −$29,000,000 |

Three of four reported three-term identities reconcile exactly. IBM 2019 does
not: the nonzero residual is preserved, without widening tolerance or claiming a
verified rounding explanation. None of these selected sets explicitly reports the
total temporary-equity concept, so the four-term identity is not evaluated.
These four source prototypes do not establish general filing coverage.

Parent equity and equity including noncontrolling interests are distinct concepts;
temporary equity needs separate treatment. Therefore the audit does not authorize
an Assets-minus-equity substitute for the direct liabilities feature. Likewise,
ProfitLoss and parent-attributable NetIncomeLoss must not become automatic aliases.
The fixed financial snapshot rules remain unchanged and financial inputs remain
disabled. A distinct equity/assets feature would require its own definition,
original-source coverage, causal availability and matched experiment protocol.

[Exact results](../results/filing-source-review-2026-10-24/balance-identity-inspection.json)
retain source and protocol hashes. All four extended extractions, inspection
results and unchanged default money payloads reproduced exactly from retained
checksummed originals. 152 focused SEC/financial checks and targeted Ruff pass.
No model fitting, normalization update, price join, retired-test access or Colab
allocation occurred. Required next work is to decide a feasible, explicitly defined
financial family from source evidence, establish original-filing and causal
issuer/price coverage, then preregister a matched training comparison. This audit
does not improve or change any recorded model return.


## Fixed missing-liabilities original-source probe — 2026-10-24

The [registered probe](../configs/experiments/filing_missing_liabilities_probe.json) selected the earliest filing by archive filing
date/accession among 33 annual 2020 candidates with supported current revenue,
net income and assets but no current direct liabilities in the financial tables.
The resulting Intel accession is `0000050863-20-000011`; no prices, model outcomes
or future declining-stock labels influenced selection. One identified, bounded
request retrieved the original extracted instance (4,497,035 bytes), retained at
`data/raw/sec-xbrl/b7b59760abcb0026c90814d3a5709ca8e26a02a38e56a018e21de025b9193fa7/instance.xml`
with its acquisition receipt. No identity/contact information is in that receipt.

Original DEI report end is **2019-12-28**, versus archive label **20191231**.
The original contains current unqualified Assets, but no qualifying current
Liabilities fact. Therefore original extraction does not recover this particular
missing direct input. The unchanged snapshot builder is unready with
`missing_or_conflicting_prior_or_balance_values`; no ratio values are emitted.
A single deterministic probe is not an estimate of the remaining cases or proof
that all original extraction is infeasible. It does show that neither trusting
archive period labels nor expecting original extraction to fill every gap is safe.
No missing amount is imputed and no financial feature is enabled.

[Result](../results/filing-source-review-2026-10-24/missing-liabilities-probe.json) records source/protocol/input hashes, exact report dates, source
counts and the full unready snapshot. Reproduce selection and extraction offline:

```bash
PYTHONPATH=src .venv/bin/python scripts/inspect_missing_liabilities.py
```

The report reproduces byte-for-byte from retained source/candidate metadata;
targeted Ruff passes. Existing 152 focused tests remain green. Raw evidence and
money payload remain retained; acquisition completed and no download process or
temporary file remains. No model fitting, source normalization, price join,
retired-test access or Colab allocation/units occurred. Next financial-family
review should consider a separately specified flow-only family with explicit
missingness and coverage gates, rather than silently changing liabilities or
parent-income definitions. Any such family still requires original vintage,
issuer/price links and causal availability before a training screen.


## Distinct flow-only feasibility family — 2026-10-24

A separately fixed [two-flow definition](../configs/experiments/filing_flow_feature_rules.json) is implemented as
`build_flow_snapshot(payload, form)` in the financial snapshot module. Its
`filing-flow-snapshot-v1` schema contains reported comparable revenue growth and
parent net margin. This is an optional new information family, not a replacement
for `filing-financial-snapshot-v1`; the latter still requires directly reported
liabilities/assets. Default registries and the existing financial as-of history
acceptance have not been expanded to enable the new schema.

The two-flow family inherits exact same-accession issuer/USD/taxonomy validation,
original report end, form-duration bounds, longest common revenue/income interval,
unique same-concept prior comparable and exact duplicate agreement. Current/prior
revenue must be positive; negative parent income is allowed. No ProfitLoss alias,
component summing, shorter-interval fallback, cross-filing fill or balance inference
is introduced. All unexcluded source facts still undergo coherence checks, even
when a balance amount is not used in the formula. Invalid required flows emit no
partial vector. No normalization or clipping is fitted.

All five previously retained originals were inspected, without outcome-based
selection. IBM2009 and Kroger2009 remain unready (missing supported current flows).
IBM2019 and Kroger2019 reproduce exactly the first two existing three-feature
ratios; their complete legacy snapshots are unchanged. Intel2019 is newly ready
for source arithmetic: revenue growth **1.5766%**, parent net margin **29.2476%**.
These are company accounting ratios, **not trading returns**. Its three-feature
snapshot remains unready. Every ready flow result was independently verified
using exact Fraction arithmetic, and the five-original sample is not a general
coverage estimate.

Of 6,161 assessed financial-table pairs, **4,512 (73.2%)** have supported current
revenue and parent income in a common source quarter group. This diagnostic omits
the assets/liabilities requirement; it does not establish prior comparables,
original exact periods, values, duplicates, vintage, stock-day coverage or causal
availability. The 251 master-only pairs remain unassessed. It is not a trained
family, an accepted dataset or permission to drop difficult stocks.

[Source arithmetic and presence evidence](../results/filing-source-review-2026-10-24/flow-feasibility-inspection.json) reproduces offline with:

```bash
PYTHONPATH=src .venv/bin/python scripts/inspect_filing_flows.py
```

162 focused checks and targeted Ruff pass. Evidence/raw sources remain retained;
no worker/runtime allocated and no additional Colab units consumed. Before any
training, preregister a bounded original-source coverage collection and fixed
issuer/price-day acceptance thresholds, retain failures/missingness and audit
original publication/vintage availability. Then freeze preprocessing/data and a
matched masked/full control with unchanged gates, followed by multi-seed and
prospective confirmation only if it passes. Existing failed families stay closed.


## Cross-era original-flow pilot prepared — 2026-10-24

The fixed [pilot protocol](../configs/experiments/filing_original_flow_pilot.json) selects the earliest original10-K by archive
filing date/accession in each of2009–2011,2015–2017 and2021–2023 for every one of
112 candidate CIKs, including issuer IDs with no ledger records. No amendments,
flow-presence filtering, price outcomes or replacement after failure. The manifest
has **319 selected originals and17 absent issuer/era slots** (336 possible slots).
Candidate issuer IDs do not imply a verified historical share/price identity.

Prepare or verify the immutable manifest without any network request:

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_original_flow_pilot.py --prepare-only
```

Then collect locally (source preprocessing only; all model training stays in Colab):

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_original_flow_pilot.py
```

Acquisition uses the private configured SEC identity, exact ledger URLs, a shared
original-cache lock and2-second inter-request delay. One request per selected
source, no retries/redirects,20MB source maximum,6GB cumulative source ceiling,
2GB free-disk reserve and3600-second collection deadline with80 seconds reserved
before the next operation.60-second checked request deadlines use at most20-second
socket waits. HTTP404 is retained as a missing case;403/429/other transport/status
errors stop the collection. Existing journals never restart automatically.

Content-addressed original bytes and URL-keyed acquisition receipts stay under
`data/raw/sec-xbrl`; identities/contact information never enter those receipts.
The manifest and extracted fact payloads remain in `artifacts/filings`; mutable
progress/terminal state is `.runtime/sec-original-flow-pilot.json`. The collector
persists the source commit and manifest/rules/ledger hashes, original report dates,
unready or unsupported cases, source hashes/bytes and elapsed time. SIGTERM is
handled as an interruption and terminal state is saved; no worker pool or remote
runtime is created. Original extracted-instance timestamps are not proof of the
original publication/vintage; that gate remains explicitly unresolved.

At least80% of selected originals overall and70% in each era must yield ready
flow arithmetic. Missing/unsupported/unready selected documents count as failures;
absent issuer/era slots are reported separately. Passing permits further causal
coverage review only, not training or dropping difficult stocks. A failure must
remain recorded, without expanding income/revenue aliases or moving thresholds.
176 focused source/financial checks and targeted Ruff pass before acquisition.
No pilot outcomes, causal price coverage, trained model or return improvement are
claimed by this preparation step. Original/raw evidence must remain retained.
