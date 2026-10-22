# Cash-choice action and accounting audit

All eight accepted selected/final phases from the [cash-choice study](../recurrent-cash-2026-10-22/README.md), covering208 symbol/phase episodes on the original22 stocks and four ETFs. No fitting, model loading, dataset opening, provider requests, Colab allocation or retired-test evaluation. Accepted summary, run identities, report hashes and source files are checked; all remain unchanged.

## Direction changed during training

These are deterministic development requests, averaged equally across each group's symbols. They describe the target requested before risk/execution, not action probabilities or final positions.

| Arm/phase | Group | Long requests | Short requests | Cash requests |
| --- | --- | ---: | ---: | ---: |
| Cash selected | Stocks | 98.01% | 0.35% | 1.64% |
| Cash final | Stocks | 68.78% | 16.30% | 14.92% |
| Cash selected | ETFs | 99.42% | 0.00% | 0.58% |
| Cash final | ETFs | 68.82% | 15.44% | 15.74% |
| Binary selected | Stocks | 93.94% | 6.06% | 0.00% |
| Binary final | Stocks | 68.96% | 31.04% | 0.00% |
| Binary selected | ETFs | 96.35% | 3.65% | 0.00% |
| Binary final | ETFs | 71.65% | 28.35% | 0.00% |

Cash stock end-long exposure falls from93.01% to65.08%, while end-short exposure rises from0.34% to15.66%. ETF end-long exposure falls94.28%→64.10%, end-short0%→14.84%. End exposure also reflects fills, protections, gaps, whole shares and holding exits. A cash request need not immediately flatten the position. The separate smoke's stochastic205/512 cash requests must not be confused with these deterministic validation fractions.

## Net directional contributions and costs

Contributions are net P&L divided by initial equity, expressed as percentage points; long plus short reconciles total return. Short P&L already includes its costs. Do not add the separate cost-drag view to short losses again.

| Arm/phase | Stock long contribution | Stock short contribution | ETF long contribution | ETF short contribution |
| --- | ---: | ---: | ---: | ---: |
| Cash selected | 31.19pp | +0.83pp | 13.97pp | 0.00pp |
| Cash final | 16.49pp | −9.21pp | 3.96pp | −0.92pp |
| Binary selected | 33.49pp | +3.45pp | 14.83pp | +5.49pp |
| Binary final | 13.55pp | −15.66pp | 10.19pp | −6.70pp |
| Posthoc final | 32.85pp | +2.54pp | 19.50pp | −0.95pp |

Cash incremental cost drag versus holding increases2.82→7.51pp for stocks and2.39→7.49pp for ETFs. Binary increases7.17→12.38pp and4.86→11.01pp. Cash final stock excess−25.35pp reconciles saved-path noncost excess−17.83pp minus incremental cost7.51pp; ETF excess−16.43pp reconciles−8.95pp minus7.49pp. Thus costs alone cannot explain the gap. These identities preserve the observed path; they are not replays without costs or causal estimates of changing execution.

The late policies reduce long participation and increase shorting; net short contribution deteriorates in both categorical arms. This describes the observed policy shift, not proof of a specific optimizer failure or that removing cash/shorts would fix it. Cash final still earns positive total return and net short P&L on two of three declining-stock episodes, while overall stock/ETF performance is weak. Future-defined declining labels remain reporting-only.

## Scope, checks and reproduction

[summary.json](summary.json) retains per-symbol action categories, equity/cost reconciliation, worst drawdown windows, accepted source/report hashes and script/module checksums. [groups.csv](groups.csv) is a compact16-row comparison. CSV returns/exposures/drawdowns are fractions; names ending `_pp` are percentage points. Drawdown windows are ex-post descriptions and differ across symbols; never use them as inputs or selection criteria. [Window definitions](../alignment-risk-actions-2026-10-22/README.md).

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_recurrent_actions.py \
 --study cash --output /tmp/cash-actions-new.json
```

The output path must be new; retained private study artifacts are required.28 focused tests pass, including all eight phases, actual cash requests, wrong report/run metadata, missing arms, extra runs, bad filenames/hashes, cleanup rejection and legacy three-arm compatibility. Accounting/action libraries reject invalid dates, actions, equity and cost chains.

No new Colab charge: project38.309006, last official account349.56 with zero rate/assignments. All12 study sessions remain stopped. Cash-choice family stays failed and closed; no extension, thresholds, architecture/entropy grids, confirmation or promotion. Next audit saved optimizer/learning diagnostics before any distinct preregistered intervention. No robust/prospective winner established.
