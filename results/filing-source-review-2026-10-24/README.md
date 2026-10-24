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
