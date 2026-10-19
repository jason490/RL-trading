# Execution-state observation study

Three CPU Colab smokes pass on actual Python 3.13.16: legacy, equal-dimension
masked and full held execution state, 512 PPO transitions each. All three owned
sessions stopped with verified cleanup. Three frozen bundles pass 234 exact
predictions, immutable normalization/data checks and incompatible-input rejection.
The legacy smoke exactly reproduces the earlier wide-initialization control's
selected and final 13-tensor policies. No local research training was performed.

Source: `8b39004de0e0fc0da92b14914d5362bd3a3cf615`; all phases use the frozen
export sharing the original cumulative ledger and allocation lock.

The predeclared 50k screen is next. Full must beat **both** legacy and masked
controls under the fixed selected/final and stock/ETF gates before confirmation.
The study has a two-unit incremental limit from 35.638335838194024 project units;
smokes ended at 35.65802928533885. All 2021–2023 results are development-only;
retired 2024–September 2026 tests remain untouched. Short-side success requires
positive total policy return **and** positive reconciled net short P&L, with
negative-holding labels used only for reporting.

See [methodology](../../docs/execution-observations.md),
[protocol](../../configs/experiments/execution_observations.json),
[preflight](preflight.json), [smokes](smoke.json),
[frozen checks](smoke-frozen-checks.json) and
[legacy parity](legacy-smoke-parity.json).
