# Longer-horizon context — prepared; no training yet

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
No new training, allocation, or credits consumed; project35.795378/150.
