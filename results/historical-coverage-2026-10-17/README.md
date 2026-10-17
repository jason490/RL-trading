# Historical universe evidence layer

Implemented stable security/issuer IDs, dated ticker and membership events,
immutable imports, coverage reports and a read-only Alpha Vantage listing adapter.
Every lookup respects effective and public-availability dates. Missing failed-stock
prices, unknown lifecycles and unresolved/non-cash terminal outcomes block approval;
zero values require explicit verified outcomes. No failed download is silently
dropped, and future terminal returns do not choose cohort members.

- [Detailed schema and operations](../../docs/historical-data.md).
- [Synthetic example](../../examples/historical-universe/history.json).
- [Software acceptance](software-acceptance.json): 170 full tests pass; nine targeted historical tests pass. The two fictional securities reuse a ticker and include a documented zero-value terminal outcome.
- [Actual current-data readiness](current-data-readiness.json): real historical membership, stable IDs and terminal outcomes remain unverified. No historical-data credentials or vendor exports are configured.

The demo's local report is `artifacts/historical-demo/coverage.html`, with companion
JSON and CSV request/coverage tables. Its passing evidence checks are software
results only, not real coverage certification. `training_ready` remains false even
when supplied evidence passes: historical episode/exit/suspension/terminal-aware
policy and benchmark accounting still need separate integration before training.
No provider subscription, bulk request, Colab allocation or new training was started.
Project usage remains 35.389379 conservative units. Existing models, datasets and
frozen shadow protocol were not modified; the retired test interval was not reopened.

Next: supply a declared universe's real dated source exports, import and audit
coverage, resolve every gap, then declare and implement the separate expanded
training protocol. Listing lists alone cannot satisfy those evidence requirements.
