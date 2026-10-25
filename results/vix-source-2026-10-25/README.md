# Implied-volatility source feasibility

The fixed date-bounded Yahoo `^VIX` source/lag coverage audit passed.
[Result](summary.json), [independent verification](verification.json), and
[protocol](../../configs/experiments/vix_source_audit.json) retain identities and
limits. **No model input is enabled and no training/return result is claimed.**

## Actual source and coverage

One top-level installed-yfinance history request,2009-01-01 inclusive through
2024-01-01 exclusive, returned3,774 positive finite closing observations dated
2009-01-02–2023-12-29. No post2023 observation is retained. The immutable local
snapshot is content-addressed `62cb5200876bcc8eddfb9ee413b12fbbc6c3058ed4b092ff4c4eb205c9bfd87e`;
its SHA256 and provider/retrieval/request metadata verify. Original index values
are kept locally, separate from executable equity bars and public result files.

| Fixed decision cohort | Unique dates | Ready | Coverage |
| --- | ---: | ---: | ---: |
|2010–2014 original training panel|1,209|1,209|100%|
|2015–2020 original training panel|1,491|1,491|100%|
|2021–2023 original development panel|733|733|100%|
|All original train/development dates|3,433|3,433|100%|

Every assessed calendar year also has100% coverage. The fixed99.5% overall/era
and99% yearly gates pass without threshold changes, extra providers or retries.
Coverage counts unique decision dates of the fixed original26-symbol dataset;
it is not proof for every other dataset or a historic point-in-time universe.

The two optional coordinates are log prior-XNYS-session VIX close and its log
change over20sessions (prior versus21st-prior session). Missing exact endpoints
yield uncovered decisions; no stale filling. An assumed availability timestamp
at the following XNYS session close+15minutes must be no later than the actual
UTC decision timestamp. All3,433 joins satisfy both strict prior source dates
and that assumption.

## Important scope

The snapshot is current-vintage unofficial Yahoo personal-use research data.
It does **not** establish original publication timestamps or historical vintages.
A conservative lag cannot repair later historical revisions. Cboe's
[historical documentation](https://www.cboe.com/tradable_products/vix/vix_historical_data)
distinguishes modern SPX VIX from old OEX/VXO methodology; this source starts
2009 after the2003 transition. Later methodology/revision assumptions must stay
visible in any experiment; this is not an individual-stock options/OI archive.

Neither the full current Cboe CSV nor retired2024–September2026 price rows were
requested. The raw stock dataset and seven-feature normalizer hash are unchanged.
No normalization fit, policy, reward, action/execution contract, broker order,
training, checkpoint selection or performance score changed. A coverage pass
is not predictive evidence or exact historical availability certification.

## Verification and next action

Four offline tests pass for request bounds/schema, exact lag/lookback, future
perturbation isolation, missing-data behavior and UTC availability enforcement.
The actual worker completed in4.71seconds and exited0; its journal is complete.
Independent pandas calendar-reindex/shift arithmetic reproduces every coordinate
for all3,433 decisions within1e-12 and verifies source, dataset and normalizer
identities. No temporary source worker remains or new Colab session/credit.
Cumulative project usage remains39.154379/150.

Next verify this same retained source against the separately accepted111-symbol
training dataset before proposing a fixed three-arm legacy/masked/full feature
comparison. Keep the seven original normalization coordinates frozen; fit only
new coordinates on training data, retain the publication-vintage limitation,
fix initial tensors/RNG and run actual Colab smokes before a screen. No experiment
is allocated or approved by this source result alone; existing failed families
remain closed and prospective confirmation would still be required.


## Expanded consumer verified

[Consumer audit](consumer.json) passes for the accepted111-training-symbol and
26-development-symbol dataset. All298,946 training and19,058 development rows
are ready; all3,433 unique decision dates, eras and years have100% coverage.
The original seven-field normalizer identity is exact.
[Independent verification](consumer-verification.json) reproduces all coordinates
with pandas calendar shifts to1.11e−16 maximum error, verifies every protected
source hash and confirms no new request, fitting, training or Colab allocation.
The consumer worker completed in10.16seconds and exited0; its journal is complete.
This remains current-vintage lag-assumption feasibility, not historical
publication/vintage verification, point-in-time membership or performance proof.

A separate [feature preregistration](../../configs/experiments/vix_context_preregistration.json)
is now fixed before constructing any new policy. Immutable augmented payloads,
mapped initializers, frozen/runtime contracts and actual smokes are pending.
No training or new model return has occurred.
