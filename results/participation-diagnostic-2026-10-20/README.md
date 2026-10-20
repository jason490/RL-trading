# Saved-path participation diagnosis — complete

Five completed50k PPO runs, selected/final phases; no training or Colab allocation. All source reports remained unchanged. Different studies are descriptive comparisons, not matched training evidence.

| Final arm | Group | Net return | Holding | Incremental cost drag | Non-cost gap | Close short sessions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| net_log_equity-11 | stock | 25.13% | 32.63% | 4.85pp | 2.65pp | 1.5% |
| net_log_equity-11 | etf | 8.64% | 19.47% | 3.88pp | 6.95pp | 1.9% |
| net_initial_equity_change-11 | stock | 22.81% | 32.63% | 4.48pp | 5.34pp | 1.8% |
| net_initial_equity_change-11 | etf | 7.36% | 19.47% | 3.57pp | 8.54pp | 1.7% |
| matched_baseline-11 | stock | 0.07% | 32.63% | 4.01pp | 28.54pp | 26.5% |
| matched_baseline-11 | etf | 5.02% | 19.47% | 4.29pp | 10.16pp | 15.4% |
| masked-11 | stock | 14.84% | 32.63% | 4.76pp | 13.02pp | 3.2% |
| masked-11 | etf | 8.91% | 19.47% | 3.99pp | 6.57pp | 3.9% |
| full-11 | stock | 15.86% | 32.63% | 4.31pp | 12.46pp | 26.4% |
| full-11 | etf | 3.07% | 19.47% | 3.52pp | 12.88pp | 29.7% |

## Findings

- Native PPO is mostly long (94% of stock closes,93% of ETF closes) but average absolute exposure is85%/82%. Stock/ETF gaps remain after historical cost add-back.
- Full context increases ETF short-session share from3.9% masked to29.7%; net short contribution falls from−0.15pp to−6.79pp. It saves about0.48pp of costs but loses5.84pp of net return versus masked, so lower costs did not rescue performance.
- Native PPO has roughly332 stock and400 ETF same-side resize fills per episode. Only80/90 are below5% equity, and their direct historical fees/impact are0.05/0.065pp. Across all5 runs, those small-fill costs stay within0.05–0.12pp: fill count alone overstates the savings opportunity.
- Native PPO averages about34/35 holding-limit fills per stock/ETF episode and34/35 protective-or-limit re-entries within one session. Stops and target-near exits are rare on this native path. The20-session limit remains required; this is not evidence to relax it.

## Interpretation and next work

Cost add-back is an exact accounting identity on fixed paths, not a free-trading replay or proof that skipping trades improves return. Close inventory misses intraday activity. All gains/losses, including short P&L, are after actual modeled costs; positive short contribution alone does not meet the positive-total-return requirement.

Do not prioritize a small resize band as the main solution. Signal errors, participation and larger exposure changes remain material. Next investigate a training-only supervised signal/warm-start control with purged chronological targets before another RL allocation; no such model or profitability claim is made here. No future validation declining-stock label may supply inputs, supervision, reward, sampling or selection.

[Detailed summary](summary.json), [group CSV](summary.csv), [interactive comparison](comparison.html), [definitions and reproduction](../../docs/participation-diagnostics.md).

No datasets/test data or model weights were opened, no new artifacts requiring session cleanup were created, and no credits were spent. Original inputs are retained with checksum receipts. Project usage remains35.856639/150 units; all earlier owned sessions are stopped.
