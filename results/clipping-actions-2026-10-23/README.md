# Saved clipping action diagnosis

Four accepted selected/final paths,104 symbol episodes, audited against exact
accepted metadata/report hashes and the original26 panel. Source files unchanged.
No model fitting, optimizer step, dataset/provider access, Colab allocation or
retired-test evaluation. [Summary](summary.json), [compact groups](groups.csv).

## Candidate selected-to-final behavior

| Metric | Stocks selected → final | ETFs selected → final |
| --- | ---: | ---: |
| Mean net return |26.68% →20.47%|13.81% →8.15%|
| Long request fraction |96.44% →92.38%|98.84% →89.11%|
| Mean closing long exposure |88.16% →84.48%|88.67% →81.08%|
| Mean closing short exposure |3.47% →6.87%|1.15% →10.59%|
| Net short contribution |−0.99pp →−4.04pp|+1.73pp →−5.98pp|
| Incremental cost drag vs holding |5.11pp →5.92pp|4.90pp →6.50pp|
| Residual noncost excess vs holding |−0.83pp →−6.24pp|−0.76pp →−4.82pp|

Short exposure and costs increase while net short contributions deteriorate.
Stock long contribution also falls27.67→24.51pp; ETF long contribution rises
12.08→14.13pp but does not compensate for worsening shorts. Net-return differences
reconcile to long plus short contributions; costs are already allocated in those
contributions and must not be subtracted again.

These observed relationships do not isolate direction versus protection effects,
prove a sole cause, establish an attainable cost-free return, justify disabling
shorts or demonstrate a reward modification. Closing exposure omits overnight/
intraday timing. Future declining-stock cohort labels remain reporting-only.

Fixed clipping family remains closed: no cap sweep/normalizer refit/extension/
confirmation. The preceding L2 crossover failed its native numerical gate and
must not be revived with relaxed tolerances. Review remaining preregistered
hypotheses against prior results before any new Colab allocation; no new model
or indicator is selected by this diagnostic. Overall objective remains unachieved.

31 focused action/participation/registered-audit tests pass, including clipping
arm routing and changed-report rejection. All four clipping sessions remain
stopped; project38.81646887, account349.17 zero rate/assignments. No extra units.
