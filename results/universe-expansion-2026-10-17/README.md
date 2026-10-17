# Training universe expansion — 2026-10-17

This separate development experiment compares PPO trained on 22 versus 55 stocks,
plus the same four broad ETFs, while scoring both arms on the original 26-symbol
validation panel. The expanded sample remains retrospective and does not resolve
survivorship bias. Current sector annotations are not model inputs.

- [Declared protocol](../../configs/experiments/universe_expansion.json)
- [Dataset identities and coverage](data.json)
- [Separate Alpaca audit](alpaca-audit.json)
- [Operations and interpretation](../../docs/universe-expansion.md)

The Yahoo snapshot passed calendar coverage for all 70 configured symbols.
The control contains 70,200 training rows; the expanded arm contains 158,546.
Both have exactly the same 19,058 validation rows, including executable bars,
feature values, dates and cash dividends. Each normalizer fits only its arm's
training data. Neither dataset contains test episodes, and no 2024 onward bars
were requested for this experiment.

Alpaca returned 139,508 SIP bars for 70 symbols over 2016–2023. LIN has 713 missing
sessions; no failed symbol was dropped or feed substituted. Corporate-action
coverage remains unverified, so this is an ingestion audit, not a training-ready
Alpaca dataset. The credentials remain local and untracked.

## Completed comparison

All six CPU PPO runs completed 50,000 transitions each on Colab Python 3.13.16.
Both arms use the same 26-symbol validation panel in 2021–2023. Returns are means
of separate $100,000 stock accounts after modeled costs, not an investable
simultaneous portfolio, annual returns or fresh held-out results.

| Training panel | Mean selected return | Mean final return | Final mean maximum drawdown | Final mean Sharpe |
| --- | ---: | ---: | ---: | ---: |
| 22 stocks + 4 ETFs | 11.77% | 10.49% | −29.40% | 0.227 |
| 55 stocks + 4 ETFs | 10.53% | 5.60% | −26.22% | 0.158 |
| Buy-and-hold, same evaluation panel | 30.60% | 30.60% | See per-symbol reports | See per-symbol reports |

| Seed | Control selected | Expanded selected | Control final | Expanded final |
| --- | ---: | ---: | ---: | ---: |
| 11 | 4.59% | 2.13% | 4.59% | 0.77% |
| 23 | 16.28% | 13.94% | 15.99% | 5.18% |
| 37 | 14.45% | 15.51% | 10.91% | 10.85% |

The declared improvement gate **failed**. Expansion reduced mean selected return
by 1.25 percentage points and final return by 4.90 points; all three paired final
returns were lower. Drawdown was less severe, but mean final Sharpe also fell.
The selected-to-final gap increased from 1.28 to 4.93 points. Expanded final
accounts incurred lower mean modeled costs ($6,831 versus $7,369), so an increase
in those final fees alone cannot explain the return gap. These comparisons do
not prove a causal learning failure or that more data is inherently harmful:
fixed 50k budgets mean less training exposure per symbol in the larger panel.

The seed-11 control exactly reproduces the prior CPU PPO study's five on-step
checkpoints through 50k. The 50k **post-update** endpoint is distinct: for expanded
seed 23, its earlier pre-update 7.63% reading becomes 5.18% after the final update.
Published final numbers always use the completed optimizer update.

Evidence: [comparison](comparison.json), [per-seed CSV](comparison.csv),
[accounting diagnostics](diagnostics.json), [control reproduction](control-reproduction.json),
[training regimes](training-regimes.json), [feed audit](feed-comparison.json),
[frozen acceptance](frozen-checks.json), and [compute](compute.json).

## Verification, artifacts and compute

All six frozen bundles passed exact selected-versus-frozen and repeated CPU
predictions on three observations from every validation symbol: **468 prediction
comparisons**. Feature order, normalization and full data manifests match their
datasets; normalization remains unchanged, and incompatible schemas/feeds and
nonfinite inputs are rejected. All seven owned allocations (one smoke, six
research) were stopped and verified. No final-test evaluation or orders occurred.

Study account debit measured 0.05 units; rate-based conservative accounting is
0.068809 units including setup, idle time, transfers and teardown. Cumulative
project usage is **35.458188 / 150 units**, leaving 114.541812 units under the
project ceiling. The 130-unit working stop remains in force. The account balance
at cleanup was 350.88 units; that is separate from this project's remaining budget.

The interactive comparison is local at `artifacts/universe-expansion/comparison.html`;
each retrieved run retains detailed selected/final validation reports and compact
JSONL/CSV/HTML learning curves with percentage-point changes every 10k steps.
Publication and temporary-file cleanup evidence are added after release verification.

## Recommendation and next gates

Keep the smaller panel as the research control and retain the expanded data for
later experiments. This 50k expansion does not justify hundreds of stocks,
longer runs or paper orders. Both arms remain below buy-and-hold on repeatedly
inspected development validation.

Next resolve Alpaca's historical identity/coverage and corporate-action units,
including missing dividend evidence and raw-versus-adjusted price conventions.
The free feeds cannot be silently merged: systematic close-level differences
exist for some symbols, LIN is missing history, and the process-date dividend
probe has 29 SPY events versus Yahoo's 32 ex-date events. Neither source is ground
truth merely because it responds. Historical membership, terminal outcomes and
terminal-aware episodes remain separate required work for survivorship claims.

Then predeclare a walk-forward development comparison with simple risk-matched
baselines and fixed search limits; reserve future data for prospective assessment.
The retired 2024 onward interval stays retired. Test any longer training budget
or reward change separately after that protocol, rather than automatically
extending this failed expansion. The previous frozen shadow campaign has not
been activated or replaced by these research models.
