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
