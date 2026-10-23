# Saved training-history action diagnosis

All four accepted selected/final phases (104 symbol episodes) audited against
accepted metadata/report hashes and exact original26 panel. Source files remain
unchanged. No model fitting, optimizer steps, dataset/provider access, Colab
allocation or retired-test evaluation. [Summary](summary.json), [compact metrics](groups.csv),
[accepted study](../training-history-2026-10-23/README.md).

## Recent-history selected-to-final behavior

| Metric | Stocks selected → final | ETFs selected → final |
| --- | ---: | ---: |
| Mean net return | +22.25% →+4.46% | +13.80% →+11.85% |
| Long request fraction | +85.77% →+76.11% | +96.79% →+93.10% |
| Mean closing long exposure | +77.59% →+69.67% | +86.38% →+85.11% |
| Mean closing short exposure | +11.82% →+19.55% | +2.75% →+5.67% |
| Net short return contribution | -3.58pp →-10.54pp | -0.79pp →-0.86pp |
| Incremental cost drag vs holding | +7.89pp →+7.26pp | +5.49pp →+5.20pp |
| Residual noncost excess vs holding | -2.49pp →-20.90pp | -0.17pp →-2.41pp |

The stock return decline accompanies more short requests and a worse net short
contribution. Aggregate incremental cost drag declines, so rising aggregate costs
cannot account for this selected-to-final fall. Cost drag is still substantial
versus holding. Short contributions already include allocated costs; never
subtract total costs from them again. Declining-stock joint positive total AND
short P&L success1/3 selected→0/3 final is reporting-only. Removing shorts or
choosing stocks using future returns is not justified by this audit.

Residual noncost excess is an accounting identity on observed paths, not a
cost-free rerun. Closing exposure misses intraday/opening fills and protections.
These correlations do not identify a sole cause, prove under/overfitting, or
establish a useful new reward. Unit-sign inference maps any nonzero Gaussian
exposure to full signed target; the learned three protective outputs stay active.

## Consequence and next diagnostic

History-composition family remains closed; no date grid, extension or promotion.
Before spending more credits, preregister a read-only comparison of these saved
selected/final models on the same fixed eligible2018–2020 training-period window,
using unchanged source features/normalization, costs and simulation rules. This
can show whether weakness also appears on eligible training history. Eligibility
and year-level counts cannot prove every replay bar was actually sampled; market
period differences also limit any generalization inference. No new checkpoint
selection, fitting, provider download or retired-test access. Exact numeric
source/model/window contracts and a clear diagnostic question must precede replay.
This next diagnostic is pending, not executed or a performance claim.

22 focused audit/action tests pass, including exact history-pair routing and
changed-report rejection plus legacy capacity/guard/cash regressions. Original
project ledger38.68329334/150 unchanged; last official account349.27 with zero
rate/assignments. All four study sessions remain verified stopped. Published
models/full reports, immutable datasets and source retained.
