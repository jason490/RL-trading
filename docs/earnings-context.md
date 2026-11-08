# Earnings dates and results

Earnings dates, actual earnings, analyst consensus and earnings surprises are
not present in the current policy observations. SEC filing ingestion and sector
metadata do not automatically supply those fields. Price gaps may reflect an
announcement, but that is not an explicit earnings input.

## Historical timing requirements

Upcoming-event features need a point-in-time calendar: store when the scheduled
date became public, before/after-market status, revisions and cancellations.
A calendar downloaded today cannot establish what was known before a historical
announcement. Days-until-earnings must use the schedule vintage available then.

Reported-result features need the original public release timestamp, fiscal
period, units, diluted/basic EPS distinction and revision identity. Compare EPS
against a consensus snapshot recorded before the release to derive surprise;
SEC actuals alone do not establish historical analyst expectations. A filing's
acceptance time is not necessarily the earlier earnings press-release time.
After-market results cannot enter decisions made before that release. Unknown
times require a conservative availability barrier; no silent midnight assumption.

## Proposed separate experiment

Audit a provider's historical event/consensus coverage and terms first. Start
with earnings proximity and recently reported-event flags; admit EPS/revenue
surprise only with valid vintage evidence. Store these separately from equity
bars, join backward by availability, preserve missingness, purge overlapping
label horizons and freeze training-only normalization. Preregister a matched
masked/full experiment with unchanged reward/execution and multi-seed gates.
No earnings source, historical surprise dataset or model feature is enabled yet.
