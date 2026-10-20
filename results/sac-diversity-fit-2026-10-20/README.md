# Final100k SAC fitted-window replay

Fixed protocol committed before replay:ba4e9f3. Both final seed11 policies replayed
all26 common symbols on the existing2018-01-02 through2020-12-02 training
window, decision anchor2017-12-29. No fitting, normalization updates, optimizer
calls, checkpoint selection, providers, Colab or retired-test access.

| Final policy | Mean fitted return | Stocks | ETFs | Median stock excess |
| --- | ---: | ---: | ---: | ---: |
| Original26-symbol training | 91.72% | 91.54% | 92.69% | +26.38pp |
| Expanded111-symbol training | 62.43% | 64.02% | 53.70% | +3.45pp |
| Matched holding | 56.45% | 58.15% | 47.15% | — |

Cumulative net episode returns, not annualized or an investable portfolio.
Both satisfy the earlier descriptive fitted-comparability thresholds. This does
not override failed development gates: same final models10.38%/24.19% against
30.60% holding in2021–2023. The fitted/development contrast is consistent with
a generalization gap; periods/regimes differ, so it does not prove a single
cause or justify selecting a winner on these training results.

Reporting-only20-session trailing declines of at least10% yield1022 overlapping
accepted decisions per arm. Original short requests53.13% on96 subsequently
continued-decline observations versus22.92% on384 large rebounds; expanded
16.67%/8.59%. Original mean approved exposure in continued-decline cohort−5.95%,
expanded+64.70%. The expanded model is more long-biased even on fitted declines.
Future cohorts are reporting-only, never observations/sampling/rewards/checkpoint
selection. Gross price cohorts omit dividends/costs and are not independent
trades or evidence of positive net short profits.

All source/model files remain checksummed and unchanged; actual policy tensors,
SAC alpha, update counts and preprocessing remain unchanged. Full derived paths
and interactive charts retained at `artifacts/sac-diversity-fit-2026-10-20`.
[Source identities and exact results](summary.json),
[protocol](../../configs/experiments/diversity_fit_diagnostic.json).
Twelve slicing/causality/rejection tests pass. Usage remains36.263895/150; no
owned session allocated. No files removed by this inference-only diagnostic.

Next investigate one fixed regularization hypothesis against a fresh identical
expanded-data control, rather than extending the failed diversity study. Preserve
actual matched sampling, initialization, costs, entropy and selected/final absolute
stock/ETF gates; confirm multiple seeds only if the predeclared screen passes.
A proposal is not yet an allocated study or evidence that regularization helps.
