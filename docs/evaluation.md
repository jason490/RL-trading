# Evaluation protocol and integrity

Evaluation uses deterministic predictions, frozen preprocessing and fresh single-symbol environments. Validation selects checkpoints; the held-out test audits an already frozen choice. Training reward alone is never presented as portfolio performance.

## Baselines

Every evaluation episode includes cash/no-trade and cost-aware buy-and-hold on the same local bar interval. Buy-and-hold enters at the first next-session open, buys affordable whole shares, pays the same fee/adverse-fill defaults, processes splits/dividends and marks the last close. Cash earns zero. Neither baseline is subject to the learned stop/zone/max-holding action rules; buy-and-hold is a benchmark holding strategy, not another learned swing policy.

Each contiguous dataset segment starts with its own initial equity. No P&L or holdings cross split/coverage boundaries. The selection score averages episode net returns equally; with verified full coverage there is one segment per symbol/partition. If an explicit research override yields extra segments, that equal-episode weighting must be considered when interpreting comparisons.

## Portfolio metrics

| Metric | Definition / caveat |
| --- | --- |
| Net return | Final marked equity / initial equity - 1 |
| Drawdown | Equity / running peak - 1; report its minimum |
| Annualized volatility | Sample std(daily net returns) × sqrt(252) |
| Sharpe | Mean daily net return / sample std × sqrt(252), risk-free=0 |
| Turnover | Cumulative traded reference notional / initial equity |
| Costs | Fees + adverse fill impact + short borrow |
| Win rate | Profitable exit fills / all exit fills |
| Average win/loss | Mean positive/negative exit-fill P&L |
| Profit factor | Sum positive exit-fill P&L / absolute sum negative; null if no losses |
| Expectancy | Mean exit-fill P&L; zero when no exits |

Trade diagnostics currently count partial exit fills separately, subtract each exit fee, and exclude entry fees from the realized diagnostic. Net equity still includes entry fees and all costs. Do not treat these sparse fill statistics as complete round-trip trade attribution or optimize win rate alone. Drawdown, costs and equity remain the accounting source of truth.

Zero return variance yields reported Sharpe zero instead of undefined division. There is no universal minimum sample length that proves profitability. Returns include open-position marked P&L at truncation and do not assume a cost-free guaranteed liquidation at the stop/target trigger.

## Final-test policy

`train --final-test` trains/selects entirely on training/validation, freezes the best candidate and only then runs the test. The audit is written alongside the run without selecting new weights. The default training command does not evaluate the test. Final-test rows already exist in the immutable dataset for reproducibility; this is a protocol boundary, not encryption or an access-control system.

Freeze features, reward, costs, hyperparameters and selection criteria before opening the holdout. If a test result influences redesign, retire that holdout as a development interval and reserve a new untouched interval. Repeated checkpoint selection and feature growth consume validation evidence; more replayed history does not create independent market evidence.

## Comparability

Match dataset identity, exposure/cost settings, seed set, transition budget and selection protocol. Report actual steps/device/duration as well as requested steps. `reporting/comparison.py` groups runs by these common settings and retains failures; it never reads test reports. Algorithm-specific parameter differences remain recorded.

For later ablations, introduce one feature family or reward term at a time. Optional data with short history requires a matched-period baseline and cannot receive credit from a changed evaluation window. Fixed retrospective ETF selection and reconstructed historical availability limit the scope of claims even when all software checks pass.
