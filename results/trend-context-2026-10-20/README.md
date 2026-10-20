# Longer-horizon context — smokes accepted; screen pending

The optional family adds causal63/252-session log price returns and252-close
trailing drawdown. It uses forward-only split continuity and excludes cash
dividends from the signal. Net-equity rewards still include dividends and costs.

[Preflight](preflight.json) reproduces the default dataset identity exactly,
then builds matched7-field baseline,10-field masked, and10-field full datasets.
All3 have64,922 training and19,058 validation rows with identical eligible raw
bar/ATR panels and first7 training normalizer statistics. Original validation
bars are unchanged. Training-only jointly negative quarterly/annual context
coverage is12.85%; this does not establish future predictive power.

The committed [protocol](../../configs/experiments/trend_context.json) requires
CPU smokes, then a3-arm seed11 50k screen. Full must pass against BOTH controls;
only then run23/37. Keep net-log reward, actions/costs/discount/starts fixed.
Ex-post declining-stock labels remain reporting-only and require joint positive
total return and net short P&L. No retired-test data or allocation is used.

[Feature definitions, controls and commands](../../docs/trend-context.md).

Actual Gymnasium and SB3 checkers pass for21/24/24 observation dimensions.
Masked/full untrained policy tensors match exactly, with3,881 smoke parameters
each ([environment receipt](environment-preflight.json)). Source snapshot ends
2023-12-29 and its full provenance matches the existing dataset. Default dataset
regeneration and all causal/matched-data checks pass. Relevant checks:53 core/
data tests,32 feature/profile tests,6 final focused tests (overlapping suites).
That preflight used no allocation; subsequent remote smoke acceptance is recorded below.

## Colab smoke acceptance

All3 variants completed512 transitions on Google Colab CPU, defaultPython3.13.16.
Artifacts passed strict dataset/feature/source/curve checks. All3 owned sessions
stopped and shutdown was verified; account350.65,rate0.00 and0 active assignments.
The3 frozen bundles pass234 exact CPU prefix predictions, deterministic repeat,
immutable normalization, and schema/feed/nonfinite rejection. This verifies
wiring and restoration; smoke returns are not model-performance evidence.

[Acceptance receipt](smoke-acceptance.json) binds all models/datasets to frozen
source780f382f6ea6cd862535b12c1723ee4dc06ade87. Study conservative usage0.022849
units; project35.818227/150. The same exact study baseline35.79537808105018
and2-unit allowance apply across all phases. Next run the preregistered3-arm
seed11 50k screen, then confirm only if full context beats BOTH controls.
