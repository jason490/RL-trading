# Causal sampling and universe screen — completed 2026-10-19

**No arm passed the declared screen. No confirmation or longer runs were added.**
All four seed-11 PPO cells completed exactly 50k transitions, using the same
original 22-stock/four-ETF 2021–2023 development panel. The experiment varied
training universe and causal start sampling; wide initialization, reward, actions,
costs and episode length were fixed. Each training universe kept its own frozen
training-only normalizer. The retired test was not evaluated.

| Arm | Selected return | Final return | Final stocks | Final ETFs | Final mean max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small, readiness-uniform reference | 2.22% | -5.43% | -7.21% | 4.33% | -32.02% |
| Small, downtrend-balanced | 16.20% | **16.20%** | 19.47% | -1.82% | -19.93% |
| Large, readiness-uniform | 0.54% | -3.48% | -4.49% | 2.03% | -25.13% |
| Large, downtrend-balanced | 10.19% | 4.05% | 4.29% | 2.73% | -24.75% |
| Buy-and-hold | 30.60% | 30.60% | 32.63% | 19.47% | — |

The small balanced candidate improves final overall return by 21.63 percentage
points, but loses 6.15 points in ETFs versus the fresh reference. The large
candidates also worsen the ETF group. The gate requires nonnegative gains in
both groups, so none qualifies. The large balanced run additionally deteriorates
from its selected 10.19% to final 4.05%. These are one-seed results, not a new
three-seed best model. The readiness-matched reference intentionally differs
from the earlier legacy-uniform initialization study; do not mix their controls.

## What sampling actually changed

Both modes share an eligible pool beginning at feature-row index 252 and ending
with the same 21-bar holding headroom. Labels use only already-observed yearly
price return below -20% and negative trailing-quarter return. No future recovery
or validation-stock outcome defines a sample, observation, reward or checkpoint.

| Arm | Downtrend starts | Downtrend transitions |
| --- | ---: | ---: |
| Small uniform | 1/112 (0.9%) | 1,443/50,000 (2.9%) |
| Small balanced | 67/119 (56.3%) | 5,646/50,000 (11.3%) |
| Large uniform | 10/109 (9.2%) | 2,564/50,000 (5.1%) |
| Large balanced | 66/117 (56.4%) | 7,760/50,000 (15.5%) |

The curriculum increases downtrend exposure, but equal-probability starting
classes do not force balanced transitions in 512-session episodes. Eligibility,
source hashes and final counters match preflight and downloaded artifacts.

## Short-side findings and training diagnosis

The small balanced policy is mostly long: mean end-session short frequency is
1.11%, versus 31.25% in the uniform reference. It never shorts AMZN or DIS in
final validation, and earns no positive-policy/positive-short case among the
three declining stocks. The large balanced policy has 31.90% end-session short
frequency and one such case: NEE returns **34.16%** with **$6,669.80** positive
short contribution per $100,000 account. This includes gains from long exposure;
it is not a pure short-strategy return. AMZN and DIS still lose 33.98% and 15.26%.
On DIS, positive short P&L of $11,949.87 does not offset its larger long losses.

`directions.csv` retains all 208 arm/symbol/selected-or-final observations,
including requested-negative-action and executed-short frequencies and reconciled
side P&L. Future-negative validation cohorts are reporting-only.

A separate **training-only** forward diagnostic finds that severe-downtrend
starts are followed by mean gross next-open-to-20th-close returns of **+3.25%**
(small) and **+3.27%** (large), versus +1.24% and +1.02% in other starts. Medians
are +3.68%/+2.68%, with about 60% positive outcomes. These overlapping historical
outcomes include cash dividends after entry day but no trading/borrow costs,
stops or rebalancing; they are not an executable-strategy backtest. They never
enter training or model selection. The result suggests that oversampling already
severe declines can emphasize rebound buying rather than teach shorting.
See `training-forward-audit.json` for exact definitions and counts.

## Concrete observation issue for the next experiment

A synthetic witness exposes a separate execution-state omission. Two legal
prior actions leave **identical seven causal market features and five visible
account values**, but different held stops. On the same next action and bar,
one position stops out at the gap and finishes at $97,970.60; the other remains
open at $98,982.97. The current observation cannot distinguish these situations.
`execution-observability-witness.json` records the exact 12-value observation,
protection levels, common action/bar and outcomes. This demonstrates hidden
execution state, not a claim that markets themselves are Markov.

Next test a versioned execution-state observation containing actual protection,
partial-exit state, last protective outputs and share-unit scale. Use a masked
same-dimension control, retain a legacy reference and preserve old frozen-model
compatibility. Keep the four-action contract, costs and net-equity reward fixed.
This structural hypothesis takes priority over extending this failed sampling
family or adding indicators without diagnosis. No new model is claimed yet.

## Verification, cost and artifacts

Four 512-transition CPU smokes plus four 50k research cells completed on actual
Colab Python 3.13.16: **202,048 transitions** total. All eight frozen bundles pass
**624 exact CPU predictions**, immutable normalization/data and rejection checks.
All eight owned sessions stopped; final CLI usage is zero assignments and
0.00 units/hour. Full local acceptance passed 207 tests with 15 warnings; the
public packaging regression also passes.

Conservative incremental usage is **0.075355 units**, within the runner-enforced
2-unit allowance. The project ledger is **35.638336/150 units**, leaving
**114.361664** (94.361664 before the 130-unit working stop). `compute.json` records
phase charges and every owned deadline/shutdown. The ledger and allocation lock
were shared across the source export and every phase; training source is
`c0d7e603983fb394faf5977cbf1164133b08d3cf`.

[The release](https://github.com/jason490/RL-trading/releases/tag/causal-sampling-2026-10-19) contains 17 assets, all verified against GitHub server
SHA-256 digests and sizes (`release-verification.json`). Cleanup then removed
36 duplicate archives/markers, reclaiming 512.95 MiB logical bytes. All 160
retained artifact hashes are unchanged (`cleanup.json`). The interactive chart
is `artifacts/causal-sampling/screen.html`. Raw data, selected/final weights,
full reports, curves and the ledger are retained. See
[protocol](../../configs/experiments/causal_sampling.json) and
[methodology](../../docs/training-sampling.md).
