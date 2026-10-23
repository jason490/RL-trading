# Saved capacity-study action diagnosis

All four accepted selected/final phases (104 symbol episodes) audited against
accepted summary, metadata, report hashes and original26-symbol panel. Source
files remain unchanged. No model fitting, optimizer steps, dataset/provider
access, Colab allocation or retired-test evaluation.

[Full reconciled evidence](summary.json), [compact group metrics](groups.csv),
[accepted experiment](../recurrent-capacity-2026-10-22/README.md).

## Larger model's selected-to-final change

| Metric | Stocks selected → final | ETFs selected → final |
| --- | ---: | ---: |
| Mean net return |29.41% →12.83%|21.21% →16.73%|
| Long request fraction |96.24% →89.57%|99.11% →91.80%|
| Mean closing long exposure |90.70% →84.03%|93.13% →85.60%|
| Mean closing short exposure |3.70% →10.10%|0.88% →7.99%|
| Net short return contribution |−0.89pp →−7.41pp|+1.07pp →+0.43pp|
| Incremental cost drag vs holding |3.78pp →5.14pp|3.01pp →5.13pp|
| Residual noncost excess vs holding |+0.57pp →−14.66pp|+4.75pp →+2.39pp|

Stock return drops16.5853pp: residual noncost gap worsens15.2284pp and
incremental cost drag increases1.3569pp. ETF return drops4.4798pp:
residual gap worsens2.3570pp and drag increases2.1228pp. These identities
reconcile observed net excess; they are not cost-free reruns or causal effects.
Net short contributions already include allocated costs, so do not subtract
aggregate costs from them again. Closing exposure omits within-session fills.

Both Gaussian policies continue to make signed requests under unit-exposure
execution; nonzero output maps to full signed target exposure. Larger final
stocks have more short requests than its selected model, alongside worse short
contribution. This does not justify a short ban, hindsight stock selection or
labels based on future returns. Reporting-only declining-stock success remains
1/3 for both larger phases, versus0/3 control. Short profitability is not robust.

## Consequence

The larger candidate's selected gain fails to persist. More width or steps is
not supported; the capacity family remains closed under its preregistered gates.
This audit describes behavior, not proof of overfitting, a sole optimizer fault,
or a remedy. A distinct next question is whether training-history composition
changes generalization: compare full2010–2020 history with a fixed recent subset,
using the same architecture, seed, costs, normalization policy and untouched
common development dates. First audit cached coverage/readiness and sampling,
then preregister exact data hashes, budget and selected/final/seed gates before
any allocation. Do not choose window boundaries using observed stock returns.
More reused development comparisons still require prospective confirmation.

21 focused audit/action tests pass, including exact capacity-pair routing and
changed-report rejection. Original project ledger38.60166524/150 unchanged;
last verified official account349.34, zero assignments/rate. All four capacity
sessions remain verified stopped. Original models/reports and release preserved.
