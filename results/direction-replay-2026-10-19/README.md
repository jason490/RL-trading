# Universal direction replay — completed, objective unresolved

Fixed-policy replay confirms profitable shorting on some falling stocks, but
universal direction filters do not produce a competitive all-market strategy.
No training, model selection or Colab allocation occurred. The default evaluator
and frozen policies remain unrestricted.

## Three-seed legacy-model results

Wide-initialization PPO, 50k source transitions, seeds 11/23/37, same 22-stock/
four-ETF 2021–2023 development panel and costs. These are replayed frozen models,
not newly trained direction specialists. Stock episodes and ETF episodes are
weighted equally within their respective groups.

| Universal mode | Selected mean | Final mean | Final stocks | Final ETFs |
| --- | ---: | ---: | ---: | ---: |
| Unrestricted | 14.23% | 13.89% | 15.55% | 4.80% |
| Long-only | 15.63% | 15.49% | 16.91% | 7.69% |
| Short-only | -0.96% | -1.19% | -0.87% | -2.94% |
| Same-symbol buy-and-hold | 30.60% | 30.60% | 32.63% | 19.47% |

Long-only improves final means in all three seeds, by 1.60 percentage points on
average, but removes shorting and remains far below holding. Short-only loses
on stocks/ETFs overall. No mode is promoted or installed as an automatic filter.

## Falling stocks can produce positive short returns

The universal short-only filter applies to **every** symbol/session; future
negative-holding labels never activate it. Final results on the three declining
stocks, after fees, impact, borrow and dividend liabilities:

| Stock | Holding | Short-only three-seed mean | Positive total + short P&L seeds |
| --- | ---: | ---: | ---: |
| AMZN | -7.73% | +2.12% | 1/3 |
| DIS | -47.38% | +6.49% | 2/3 |
| NEE | -14.63% | +2.87% | 3/3 |

Six of nine stock/seed episodes have positive total return and positive net short
P&L. These are nine retrospective episodes, not nine independent markets or a
trade win rate. Reliability varies: seed 11 returns +16.30%, +15.88%, +1.37% on
AMZN/DIS/NEE; seed 37 returns -6.86%, -1.58%, +4.30%. A falling-stock subset's
success cannot establish a causal stock selector or future profitability.

## Observation-profile diagnostic

The initial seed-11 comparison also replayed masked and full execution-state
models. Final unrestricted/long-only/short-only means were:

| Profile | Unrestricted | Long-only | Short-only |
| --- | ---: | ---: | ---: |
| Legacy | 7.96% | 11.25% | -1.91% |
| Masked | 15.81% | 20.73% | -1.68% |
| Full | 22.59% | 21.29% | -2.28% |

The full model changes its **pre-filter** requested direction on 2,797 of
19,032 decisions (14.70%) between unrestricted and short-only account paths,
despite identical market features and weights. Legacy changes 215 (1.13%) and
masked 1,290 (6.78%). This measures account-path feedback, not mechanical action
clipping. Account dependence can be appropriate for costs and risk; it does not
prove that removing those inputs will improve returns.

## Numerical parity and preserved failure

All original unrestricted sessions match. Seed-11 and seed-23 phase paths, plus
seed-37 final, satisfy the original exact-quantity and tight CPU tolerances.
Seed-37 selected initially failed on two one-share differences. The first comes
from a 1.2e-7 exposure change crossing an integer sizing boundary with identical
prior equity; maximum equity drift is $0.9932, with no position-sign changes.
The strict failure is retained, rather than relabeled as exact.

A reference-only run measured both phases before any filtered seed-37 replay.
A separately committed calibrated protocol then allowed at most one share on ten
rows, identical sessions/signs, $2 daily equity, $0.05 costs, 0.001 action changes
and 0.00002 return/drawdown fractions. The actual selected path uses only the two
one-share exceptions. Restrictions are compared with their paired local
unrestricted paths. Both the original strict failure and bounded exception
remain explicit in the three-seed summary.

## Verification and artifacts

The initial 18 profile/phase/mode labels use nine unique deterministic replays;
selected/final weights are identical within each seed-11 run. The seed-23 and
seed-37 follow-ups add twelve labels and twelve unique replays. Selected/final
remain separate when fingerprints differ. Two additional unrestricted calibration
paths are retained, along with the original failed-attempt receipt. All 21
complete source-report checksums are verified. Original models, normalization,
source reports and data remain unchanged. All 314 retained execution-state-study
hashes and all 228 older initialization-study retained hashes were independently
verified. Focused environment/direction checks pass; the paired reporter also
tests incomplete evidence, mismatched weights, false profit flags and mechanical
filter-versus-request confusion.

Project consumption remains **35.687823/150 units**, with **114.312177** remaining
and **94.312177** before working teardown at 130. This diagnostic adds zero Colab
units; account reports zero active assignments. No retired 2024–September 2026
test was read and no broker orders were sent. Immutable datasets still have the
provider-adjustment and survivorship limitations documented in the source study.

- [Protocol](../../configs/experiments/direction_replay.json),
  [seed 23](../../configs/experiments/direction_replay_seed23.json),
  [original seed 37](../../configs/experiments/direction_replay_seed37.json),
  [calibrated seed 37](../../configs/experiments/direction_replay_seed37_calibrated.json).
- [Three-seed summary](three-seed.json), [paired per-symbol CSV](three-seed.csv).
- [Initial profile summary](summary.json), [seed 23](seed23-summary.json),
  [seed 37](seed37-summary.json), [initial profile CSV](per-symbol.csv).
- [Strict failure](seed37-initial-failure.json),
  [reference calibration](seed37-reference-calibration.json),
  [integer boundary](seed37-rounding-boundary.json), [preflight](preflight.json).
- [Methodology and commands](../../docs/direction-replay.md).

Full compressed replay paths stay local under `artifacts/direction-replay*`;
interactive charts show group, per-stock and per-seed results. Public assets
contain derived reports, not original market bars, credentials or replay buffers.

## Next controlled hypothesis

Test a PPO architecture whose signed-exposure output sees market features only,
while protective outputs and the critic retain full execution/account inputs.
Use a same-capacity two-branch unmasked control and the native full-state model,
with identical data, costs, initialization and budgets. Prove the information
mask and frozen compatibility before Colab smokes. Declare numeric promotion
rules and the original shared-ledger allowance before allocation. This is a
hypothesis about account feedback; it has not been implemented or trained and
might worsen cost-aware trading. No retrospective declining-stock label selects
an input, checkpoint, gate or trade.

## Publication and cleanup

[Download charts, CSV and diagnostic receipts](https://github.com/jason490/RL-trading/releases/tag/direction-replay-2026-10-19). All 19 server
asset hashes/sizes match ([verification](release-verification.json)). Cleanup
removed 18 byte-identical publication copies, 10,333,406 logical bytes (9.85 MiB),
with all 597 retained hashes unchanged. Full replay/calibration paths, original
models, source data and ledger remain local; summaries and original failure
receipts remain in Git ([cleanup](cleanup.json)). No new Colab session was
allocated; final account status reports zero active assignments.
