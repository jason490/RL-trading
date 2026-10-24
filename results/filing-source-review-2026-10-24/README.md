# SEC filing source review — 2026-10-24

Outcome-free documentation review identified two requirements before historical
financial features can be enabled. No financial archive was downloaded and no
model performance was scored. The existing return-improvement goal is unmet.

1. SEC's [archive page](https://www.sec.gov/dera/data/financial-statement-data-sets.html)
   describes December2024 reprocessing of historical archives and a new NUM
   `segments` column. The [current table specification](https://www.sec.gov/files/financial-statement-data-sets.pdf)
   includes it in the numeric key. The earlier `/dera/data/fsds.pdf` describes an
   older layout. Implementation now reports legacy versus segment-aware schema
   and includes segment identity when counting duplicate/conflicting facts.
2. SEC's [timestamp/publication FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions)
   distinguishes acceptance from public availability, which has variable lag and
   no explicit first-publication timestamp. Its [access guide](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
   also describes delayed dissemination and post-acceptance corrections. The
   audit retains timing and revision uncertainty; structural success is not
   point-in-time fidelity.

The original [audit preregistration](../../configs/experiments/filing_coverage_audit.json)
remains unchanged. This review adds source-schema requirements before observed
financial data. No model/coverage gates are relaxed. Immutable archive retention
and original-filing verification remain required to assess reconstruction limits.

37 offline reader/downloader tests pass, including distinct segment facts,
identical decimal duplicates, conflicting/missing duplicate values, legacy schema
reporting and immutable CLI output. Numeric-key comparisons use a temporary
SQLite database to avoid retaining every fact key in Python memory. It requires
temporary disk space proportional to input size; source archives are retained.

Pending: private request identity, actual source acquisition, historical identity
mapping, original-vintage verification, concept/unit/duration resolution,
coverage/staleness gates, feature specification and a separate training protocol.
No Colab allocation, training, new compute units or broker orders in this review.

## First archive acquired and inspected

The configured request identity enabled actual2020q1 acquisition:96,584,608 bytes,
SHA256`48ed9834c66d565c21130d087eae941b25770e2d7cc8351d17652612289fcd67`.
Raw archive and acquisition receipt remain in ignored`data/raw/sec/`. It contains
5,816 submission records, all acceptance strings with a `.0` fractional suffix,
and an optional`readme.htm`. The reader now supports these formats strictly;
44 focused offline checks pass, with lint/whitespace verification.

Actual structural acceptance **failed** under the retained cutoff rule. A separate
metadata-only scan counted3,099,352 NUM records, including exactly one2024-12-31
period label for`AmortizationOfIntangibleAssets`. No numeric amount was recorded
in that diagnostic. This may be a forward-looking disclosure or a date issue;
the label alone cannot establish its meaning or public availability. It was not
accepted as a model input. No retired test bars/returns were opened, no model
fitting or Colab allocation occurred. [Diagnostic](first-archive-diagnostic.json).

Next: resolve original-filing semantics and freeze explicit out-of-scope record
handling before another acceptance attempt. The failed strict audit must remain
reported, and source records must never be silently relabeled or overwritten.

## Original-filing resolution

The [original2020 filing](https://www.sec.gov/Archives/edgar/data/1062231/000106223120000009/axl201910-k.htm)
contains a2024 amortization context explicitly tagged`ScenarioForecastMember`.
It is a forward disclosure, not a realized2024 observation. Original HTML/receipt
are retained under SHAdede7989…; [machine-readable resolution](forward-period-resolution.json)
records the verified context and source identity, without a financial amount.

The first strict failure is retained. A separate [audit addendum](../../configs/experiments/filing_forward_period_addendum.json)
introduces explicit quarantine of **all** NUM periods beyond their accession's
SUB period, including pre2024 dates. Default strict mode is unchanged; no retired
filing becomes admissible. Version3 reports mode and bounded metadata examples;
quarantined amounts never enter numeric comparisons/model preparation.48 focused
checks pass; remaining causal-readiness flags stay false. This is source
feasibility work with no training, normalization fitting or Colab allocation.

## Completed explicit-quarantine inspection

The new-mode full archive scan completed.5,816 filings/5,404 CIKs;
3,099,352 numeric rows total,3,096,249 inspected and3,103 forward-period rows
quarantined. All inspected numeric keys are unique; no conflicting duplicate
key was found.108,017 inspected values are missing;1,538,091 inspected rows have
segments. Three submissions have acceptance dates after their stated filing
dates and require review. Quarantined rows are not all necessarily forecasts:
post-reporting-period events and rounded dates can also occur. None is used as a
financial input. [Compact report](2020q1-quarantined-summary.json).

The full structural report remains in ignored`artifacts/filings/` with
SHAbc452aa8…. Structural completion does not mean causal acceptance: publication,
timezone, historical identity mapping and feature eligibility remain unverified.
No performance scoring or fitting occurred. The strict original failure remains
in the earlier diagnostic and has not been overwritten or converted into a pass.

## Current candidate discovery and first archive coverage

Current SEC ticker discovery matches all107 research stocks, source snapshot
SHA93050f93…. A reproducible read-only join finds qualifying10-K/10-Q/amendment
records for102 current CIK candidates in2020q1. Five absent:APA,BLK,IBM,KR,XOM.
All historical-verification/model-input flags remain false. Full derived
[candidate report](2020q1-candidate-coverage.json) records accessions and counts;
current raw metadata and discovery receipts stay outside Git.

Do not classify all five as companies without filings: the original
[IBM filing index](https://www.sec.gov/Archives/edgar/data/51143/0001558370-20-001334-index.htm)
shows its2019 annual filing, while
[APA's2021 reorganization disclosure](https://www.sec.gov/Archives/edgar/data/1841666/000119312521063695/d127090d8k12b.htm)
shows a later parent-registrant change. Current identifiers cannot substitute for
dated instrument/issuer evidence. This inspection does not yet verify any full
2010–2023 identity history or train a model.51 focused tests pass, lint passes,
and exact derived report reproduction succeeds. No new Colab allocation/units.

## Three document-level historical issuer matches

Original2019 reports filed in2020 declare APA/CIK6769,BLK/CIK1364742 and
XOM/CIK34088. Checksummed original HTML and receipts are retained; the
[dated evidence](2020q1-dated-identities.json) records exact DEI issuer/ticker,
security-title/exchange and context metadata. These older IDs explain three
current-candidate absences without inventing filings or globally backdating a
new parent. Alternate symbols`APA/29` and`BLK25` are retained distinctly.

56 focused SEC tests pass, including exact ticker matching, current-parent
mismatch, shared/invalid issuer ambiguity, source tampering and no inferred
historical interval. No whole-history identity chain or financial feature dataset
is certified. IBM/KR absence and chronological source coverage remain pending.
No model fitting, normalization updates, retired test prices/returns, Colab
allocation or new compute units were involved.

## IBM/Kroger source and adjacent-quarter resolution

Kroger's annual filing date is2020-04-01, and its accession is present in the
retained2020q2 archive (82,008,710 bytes/SHA bd796ea1…). IBM's April10-Q and
Kroger's June10-Q are also in that archive's SUB table. This is a targeted SUB
inspection, not full Q2 numeric validation. [Evidence](2020q2-ibm-kr-sub.json).

IBM's February annual report exists outside the Q1 ZIP. Its retained main HTML
has no selected assets/liabilities/revenue/net-income inline numeric concepts;
its Exhibit13 does. The exact SEC extraction cause is unproven. Retained indexes,
main HTML and exhibit establish source availability, not automatic permission to
substitute an unvalidated extraction. [Source review](archive-absence-resolution.json).
Its original main document explicitly declares issuer51143/tickerIBM, confirmed
by the checksummed inline inspector. [Dated identity](ibm-dated-identity.json).

Kroger's original report period2020-02-01 differs from SUB20200131; index
acceptance09:01:18 differs from SUB09:01:00. Preserve original/normalized labels
separately and verify fiscal periods before financial growth calculations.
Causal readiness/whole-history identity remain false.56 focused SEC tests and
lint pass; source checks confirm recorded dates/hashes and exact accession
presence. No fitting, retired prices/returns, allocation or additional units.
An attempted large Q1 master-index inspection exceeded its20MB cap and stopped;
no response was retained or used. Targeted original indexes supplied the evidence.

## Original money-fact extraction feasibility

Two checksummed original SEC-extracted XBRL instances were inspected: IBM
SHAfc20caf0… and KrogerSHA9e326eef…. Restricted US-GAAP money extractor preserves
exact contexts, units, decimals, original dates and all competing fact rows.
IBM237 selected/211 excluded/26 unqualified USD rows; Kroger54/21/33. Qualified
segments/scenarios have no extracted amount. Publication, historical identity
and financial-feature readiness stay false. [Metadata-only summary](original-extraction-summary.json).

Kroger's original annual duration2019-02-03–2020-02-01 remains exact, and supplying
the ZIP's2020-01-31 report end rejects. Source instance bytes/receipts and full
amount-bearing extraction remain in ignored data/artifacts.71 focused SEC tests
pass, including exact decimal preservation, forward/non-USD/nil exclusions,
original-period mismatch, source integrity, orphan references and duplicate
retention. No normalization fitting, training, retired prices/returns or new units.
Chronological coverage and a separate concept/feature protocol remain required.

## Full source collection and chronological SUB inventory

All59 registered2009q2–2023q4 ZIPs are retained (4,613,634,809 bytes) with
acquisition receipts; collector terminal after601.7 seconds. Hash-verified SUB
inventory:366,856 unique submissions,342,981 qualifying financial filings,
14,100 issuers. No cross-archive duplicate/conflicting accession observed.
[Source inventory summary](chronological-inventory-summary.json).

The initial inventory stopped on three blank periods in non-periodic forms,
not the10-K/10-Q family. [Original failure/diagnostic](invalid-sub-date-diagnostic.json)
is retained; updated inventory reports those blanks without inventing periods,
while qualifying forms still require valid dates.81 focused SEC tests/lint pass.
This does not mean every financial table/fact or historical security is accepted.

Current-CIK candidate annual counts are saved in[candidate-year-counts.json](candidate-year-counts.json).
Standalone offline HTML lives at`artifacts/filings/candidate-coverage.html`;
full metadata inventory at`artifacts/filings/chronological-sub-inventory.json`.
Repeated HTML/summary generation reproduces exact bytes. Current BLK/XOM IDs have
zero source counts; APA begins2021 and GOOGL2015 under current IDs. Earlier
issuers need dated evidence, not deletion or silent backfilling. None of these
counts is a model input or a performance result. All causal-readiness flags false.

[Retention receipt](collection-retention.json): all raw ZIPs/receipts and original
filings kept, no temporary download files or new Colab allocation. Last official
balance349.02/rate0/assignments0; source collection used zero Colab units.
Source financial concepts, exact periods, historical lineage and availability
must still pass before a matched training protocol is enabled.
