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

## Succession events and causal issuer queries

Retained original filings/indexes establish Google→Alphabet effective2015-10-02
and Praxair/LindeAG→Lindeplc effective2018-10-31. Source hashes, official dates and
event types are in the[review registry](../../configs/filings/issuer_evidence_v1.json).
The latter is a business combination, not proof of financial-ratio continuity.
Neither certifies the full adjusted price-security history. Name-only discovery
also encounters an unrelated Alphabet Holding company, so cannot substitute for
source-backed identity.

The fixed daily reconstruction waits until the second XNYS open after the later
filing/acceptance local date: AlphabetOct6,2015 and LindeNov2,2018. Effective event
and reconstructed availability must both pass the as-of query. Before then the
unanchored issuer remains unknown; no earlier CIK is inferred from future events.
Actual[query evidence](issuer-query-inspection.json) retains all financial/price
comparability flags false. This is not guaranteed first-publication timing:
[policy](../../configs/experiments/filing_availability_policy.json) records the
assumption and leaves exact-publication verification false.

90 focused SEC tests pass, including future-event isolation, unknown prehistory,
conflicting parent/subsidiary observations, DST/holiday/weekend lag and late2023
availability crossing into2024. No model fitting, normalization update, retired
price/return evaluation or new Colab units. Full chronological identity and
financial feature eligibility remain pending.

## Financial snapshot arithmetic verified

Fixed[coherent-accession rules](../../configs/experiments/filing_feature_rules.json)
produce revenue growth, net margin and liabilities/assets without price joins,
normalization fitting, concept-priority substitution or cross-accession vintage
mixing. Annual and subannual interval bounds distinguish fiscal calendars;
unique comparable prior periods and exact duplicate-value agreement are required.
Conflicting/missing sources remain unready rather than imputed.

Actual original IBM/Kroger annual snapshots pass, with all three values exactly
equal to independently computed rational arithmetic. Source/rule/code hashes and
historic company ratios are in[inspection evidence](financial-snapshot-inspection.json).
These are financial statement ratios, **not model returns**.108 focused checks
pass, including16-week quarters, negative net margin, conflict rejection,
ambiguous comparables, mixed-vintage/unit rejection and source immutability.
All publication/price-link/model-enabled flags remain false. Full chronological
financial dataset acceptance and a separate training protocol are still pending.
No training or new Colab units.

## Source-history publication/expiry boundaries

The optional research history reader now selects only available same-issuer
snapshots, respects original reporting-period precedence and handles amendments
without rewriting past queries. New unready reports block fallback; source age
is450days annual/180days quarterly from original report end, so amendments cannot
refresh stale periods. Input/output copies preserve earlier query values.

Actual original IBM/Kroger indexes confirm publication-boundary and expiry tests:
IBM reconstructed2020-02-27T14:30Z; Kroger2020-04-03T13:30Z. One second before
availability no value is returned; at availability source arithmetic matches;
after period expiry values are absent. [Inspection](financial-asof-inspection.json).
116 focused checks pass. This tests a **partial supplied ledger** only; omitted
filings cannot be detected by the reader, and no full-universe coverage rate is
claimed. Include all unready/ready sources before dataset acceptance. Stock-price
links, complete catalog and financial-input approval remain false. No price join,
normalization fitting, model training or new compute units.

## Original candidate ledger and earliest-source compatibility

All59 financial SUB archives produce6,161 candidate original instances across112
current/predecessor CIKs for107 research stocks. All archive hashes and target
filenames were checked; candidates do not certify historical issuer assignment.
The[ledger summary](original-candidate-ledger-summary.json) explicitly flags the
full filing catalog incomplete, including the known omitted IBM annual accession.
A3.81MB compressed2020Q1 master ZIP recovers that accession; its29.47MB expanded
index exceeded the earlier raw request cap. [Inspection](compressed-master-inspection.json).
Full master-index comparison remains required before catalog completeness.

Original2010 IBM/Kroger instance bytes/receipts are retained. Legacy`xbrl.us`
namespace support fixes extraction compatibility, but both snapshots remain
arithmetic-unready under fixed concept rules. IBM uses`ProfitLoss`, Kroger goods
sales; no silent aliasing was used to claim comparable parent net margin or total
revenue. Mixed taxonomy versions also remain unready. [Early-source evidence](early-original-source-inspection.json).
121 focused checks/lint pass, including predecessor selection without fuzzy names,
unsafe paths, incomplete source range and corrupted archives. No model fitting,
price joins, final-test evaluation or new Colab units. Next complete filing-index
catalog and concept-coverage audit before scaling original extraction.

## Complete registered-quarter master comparison

59 master ZIPs retained,186,760,088 compressed bytes/157.9seconds;14,876,676
metadata rows scanned. Same112 candidate CIKs/forms:6,412 issuer/accession pairs,
6,410 unique accessions. All6,161 financial-table pairs match, with251 additional
pairs and zero financial-only pairs. Additional unique documents total249;
two Google/Alphabet2016 joint accessions add co-registrant links, not new documents.
The known omitted IBM annual is recovered. Six filing-date discrepancies remain
explicitly unresolved; no form differences or duplicate-master conflict observed.
[Comparison/deltas](master-catalog-comparison.json).

Bounded parser verifies source hash, compressed/expanded size, unique member,
metadata dates and safe paths. Joint issuers remain separate, including a distinct
filing-directory CIK.131 focused tests/lint pass; full comparison reproduces exact
bytes. Candidate catalog completeness is relative to current SEC indexes, which
can reflect corrections. It is not financial-fact, original-vintage, price-link
or model-input approval. The original statement cohort is still uncollected.

[Retention inspection](master-retention-inspection.json): collector terminal,
all59 archives/receipts kept, no temporary downloads. Original preflight bytes
reused without another request. Official Colab349.02/rate0/assignments0; no
additional compute units or model fitting. Next audit concept coverage and collect
validated original filings from the corrected catalog before training.

## Full concept-presence audit completed

148,956,855 NUM rows/59 quarters scanned by two local preprocessing workers in
212.4seconds; workers stopped and quarterly evidence retained. All6,161 financial
candidate pairs assessed;251 master-only pairs unassessed. Required direct tag
presence:revenue5,259/85.4%,NetIncomeLoss5,283/85.7%,assets6,125/99.4%,
liabilities3,873/62.9%. All four in a compatible flow group:2,695/43.7%.
[Per-year/concept summary](concept-presence-summary.json).

This is **not arithmetic or stock-price coverage readiness**. Missing table facts
may be present in originals; positive presence does not verify amounts, prior
comparables, contexts, vintages or issuer links. Unsupported-label counts overlap
and are diagnostic, not aliases or disjoint failure groups. Source fields remain
unchanged; no amount imputation, price join, fitting or performance scoring.
The direct-liabilities gap and incompatible labels require accounting/source
validation before large original collection or financial-model training.

144 focused tests/lint pass. Full metadata/HTML at`artifacts/filings/` reproduce
exact bytes from retained quarter results. [Retention/dependency proof](concept-audit-retention.json)
confirms stopped workers/no temporary files and source dependency integrity.
No additional Colab units or financial-input activation. Model-return results
remain unchanged; no robustness/promotion claim.


## Original balance-identity audit — 2026-10-24

The separately registered [accounting audit](../../configs/experiments/filing_balance_identity_audit.json)
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

[Exact results](balance-identity-inspection.json)
retain source and protocol hashes. All four extended extractions, inspection
results and unchanged default money payloads reproduced exactly from retained
checksummed originals. 152 focused SEC/financial checks and targeted Ruff pass.
No model fitting, normalization update, price join, retired-test access or Colab
allocation occurred. Required next work is to decide a feasible, explicitly defined
financial family from source evidence, establish original-filing and causal
issuer/price coverage, then preregister a matched training comparison. This audit
does not improve or change any recorded model return.


## Fixed missing-liabilities original-source probe — 2026-10-24

The [registered probe](../../configs/experiments/filing_missing_liabilities_probe.json) selected the earliest filing by archive filing
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

[Result](missing-liabilities-probe.json) records source/protocol/input hashes, exact report dates, source
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

A separately fixed [two-flow definition](../../configs/experiments/filing_flow_feature_rules.json) is implemented as
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

[Source arithmetic and presence evidence](flow-feasibility-inspection.json) reproduces offline with:

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

The fixed [pilot protocol](../../configs/experiments/filing_original_flow_pilot.json) selects the earliest original10-K by archive
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
