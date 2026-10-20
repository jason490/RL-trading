# Frozen seed ensembles — completed diagnostic

All 18 selected/final cases completed; twelve previously verified cases were reused and six were newly evaluated. No training, Colab allocation, normalization update, or retired-test evaluation occurred. Source artifacts and model tensors remained unchanged; every compressed path checksum and recipe restoration passed.

| Fixed ensemble | Selected return | Final return | Final stocks | Final ETFs | Declining stocks positive total + short P&L (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| combined_mean | 21.42% | 22.16% | 24.61% | 8.68% | 0/3 |
| ppo_mean | 19.27% | 19.36% | 22.29% | 3.25% | 1/3 |
| sac_mean | 21.38% | 17.98% | 20.02% | 6.75% | 1/3 |

Matched holding returned 30.60% overall, 32.63% on stocks and 19.47% on ETFs. All candidates fail the absolute comparability gates. Combined averaging clears the descriptive relative gate (+2.22 pp selected, +5.46 pp final versus mean fresh individual references), but that is one ensemble, not independent confirmation. No candidate is promoted.

## Falling-stock requirement

Positive short P&L alone is insufficient: total return must also be positive after long losses, fees, impact, dividends and borrow. These labels are reporting-only and never choose trades, checkpoints or members.

| Final ensemble | Symbol | Total return | Net short P&L | Both positive |
| --- | --- | ---: | ---: | --- |
| ppo_mean | AMZN | -16.26% | $111.58 | no |
| ppo_mean | DIS | -39.36% | $5,175.19 | no |
| ppo_mean | NEE | 1.41% | $1,690.33 | yes |
| sac_mean | AMZN | -38.97% | $-3,388.72 | no |
| sac_mean | DIS | -42.46% | $2,672.98 | no |
| sac_mean | NEE | 2.69% | $2,051.21 | yes |
| combined_mean | AMZN | -16.68% | $-2,293.59 | no |
| combined_mean | DIS | -42.97% | $3,247.69 | no |
| combined_mean | NEE | -2.50% | $303.33 | no |

## Reference failure and limits

The original $1 source cost-drift gate failed on final SAC seed 11 ($3.24 drift). It remains failed. The separately preregistered continuation allows this cost-only discrepancy up to $5 for descriptive analysis, with the original equity/return/drawdown caps unchanged. Promotion is disabled regardless of the relative comparison. Inputs match the development-only snapshot contract and saved validation dates/closes; identical raw training provenance is not claimed.

## Evidence and operations

[Interactive comparison](comparison.html), [per-symbol metrics and side attribution](per-symbol.csv), [complete derived summary and frozen recipes](summary.json), [original failure receipt](original-failure.json), and [methodology](../../docs/ensembles.md). Complete compressed execution paths remain in the ignored local artifact directories; they are not duplicated in Git. The existing source model release is [algorithm-stability-2026-10-16](https://github.com/jason490/RL-trading/releases/tag/algorithm-stability-2026-10-16).

Current Colab usage: zero active assignments, rate 0.00 units/hour; account balance 350.69. Project cumulative usage remains 35.748563/150 units (114.251437 remaining). Both replay processes are terminal. No session teardown was needed. Failed-run evidence, source bundles, datasets, execution paths and the shared ledger are retained.

Next: investigate objective alignment with a separately declared matched control before further training; do not extend unsuccessful training blindly. Require aggregate stock/ETF gates and report joint total-plus-short success on falling stocks, then independent prospective evidence before any profitability claim.
