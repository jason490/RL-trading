# PPO action initialization — completed 2026-10-19

Wider initial protective-action means improve matched **50k PPO** development
returns across seeds 11/23/37. They still miss the stock/ETF holding targets and
worsen the declining-stock cohort. This is a relative research improvement,
not a successful trading model or the best result across different budgets.

| Three-seed mean | Standard PPO | Wide initialization | Buy-and-hold |
| --- | ---: | ---: | ---: |
| Selected return | 11.77% | 14.23% | 30.60% |
| Final post-update return | 10.49% | **13.89%** | 30.60% |
| Final stock return | 12.61% | 15.55% | 32.63% |
| Final ETF return | -1.16% | 4.80% | 19.47% |
| Final mean maximum drawdown | -29.40% | -30.02% | — |

| Seed | Standard selected / final | Wide selected / final |
| --- | ---: | ---: |
| 11 | 4.59% / 4.59% | 7.96% / 7.96% |
| 23 | 16.28% / 15.99% | 19.02% / 18.74% |
| 37 | 14.45% / 10.91% | 15.69% / 14.98% |

The declared relative confirmation gate passes: mean selected/final gains are
+2.45/+3.40 percentage points, final returns improve in all three paired seeds,
stock/ETF final gains are +2.93/+5.96 points, and mean final drawdown worsens by
0.62 points (below the 2-point limit). Absolute comparability and outperformance
both fail. Returns are on the original 22 stocks/four ETFs during 2021–2023
**development validation**, not independent or retired-test results.

## Short-side result

For AMZN, DIS and NEE, whose holding returns were negative, final standard PPO
averages **-12.20%** and wide PPO **-16.50%**, versus holding **-23.25%**. Wide
has **0/9** stock-seed observations with both positive total policy return and
positive short P&L; standard has **2/9**. These are three stocks across three
seeds, not nine independent stocks. Wide's average short P&L is positive
($2,645.09 per $100,000 account), but other losses more than offset it.

The shorting objective is not met. The future-negative cohort is reporting-only:
it did not enter observations, sampling, rewards, checkpoint selection or the
promotion gate. `short-side/summary.json` and `short-side/per-symbol.csv` retain
all 312 symbol/seed/phase records for the matched six-model comparison.

## Screen, source and verification

The four-arm seed-11 screen held basic features, net-equity reward, costs,
training data, signed exposure and learned four-action semantics fixed. Standard,
low-noise, wide and combined final returns were 4.59%, 7.18%, 7.96% and 7.43%.
Wide won the declared final-return ranking; combined also passed but ranked
second. Low-noise failed because its ETF return deteriorated. Only wide and
standard received seeds 23/37. Exposure initialization was unbiased in every arm.

All eight research cells completed exactly 50k transitions, plus four distinct
512-transition Colab CPU smokes: **402,048 transitions total**. Actual Python was
3.13.16. Eight research and four smoke frozen bundles passed **936 exact CPU
prediction comparisons**, immutable normalization/data and schema/feed checks.
All three standard controls exactly reproduce the previous 50k control study:
**39 policy tensors and 18 checkpoint returns** match. Curves retain 10k
checkpoints and separate final post-update values. No retired test was loaded.

Training source is `dab2e3360b10f0e06f6b7e2818770e92cf266e80`. Confirmation used
an exact committed-source export so unrelated complete-case dataset engineering
on main could not change this experiment. Its package hash matches the screen;
its ledger, allocation lock, data and artifacts resolve to the original project
paths. No additional project budget was created. See `confirmation-source.json`.

## Compute, artifacts and next work

All **12 owned sessions stopped with verified cleanup**; final CLI usage reports
zero assignments and 0.00 units/hour. Conservative incremental usage is
**0.104793 units**, within the 2-unit study allowance. Metered-window debits total
0.05 units; the separate rounded account balance change was 0.06 units. These
observations are not added together. The conservative ledger exceeds both and
now totals **35.562980/150 units**, leaving **114.437020** (94.437020 before the
130-unit working stop). `compute.json` records each phase and every owned session.

Seventeen public assets are prepared with checksums: frozen models, selected and
final weights, derived equity reports, paired interactive charts, learning curves,
protocol, source/data manifests, gates and short-side evidence. Publication and
server checksum verification are pending; duplicate archives remain until then.
Local charts: `artifacts/action-initialization/screen.html` and
`artifacts/action-initialization/confirmation.html`.

The next hypothesis should address causal down-regime training coverage before
blindly extending this same sample. The separately prepared 107-stock dataset
contains 5.15% persistent trailing-decline observations versus 2.31% in the
original training panel. A bounded sampling/universe experiment can test this;
future recovery labels must never define its sampling. Retain wide as an
optional overall-return research baseline, not as a solution to shorting.
Prospective confirmation and historical membership/delisting evidence remain
outstanding. See [protocol](../../configs/experiments/action_initialization.json),
[operations](../../docs/action-initialization.md), and
[larger-data evidence](../complete-case-107-2026-10-19/README.md).
