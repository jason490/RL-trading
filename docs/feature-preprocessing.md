# Market-feature preprocessing research

## Frozen-scale audit

[Actual audit](../results/feature-scale-2026-10-23/README.md) examines the immutable
full111 training/original26 development dataset under its original frozen
normalizer. No model/reward/P&L outcomes used to choose settings, no preprocessing
fit or clipping, no provider/model access or Colab allocation. Only train and
reused development inputs; retired test absent.

`reporting/feature_scale.py` owns finite ordered-matrix validation and normalized
mean/std/min/max/quantiles plus fixed descriptive abs3/5/10 fractions. Float64
values match the frozen transform before the final float32 observation cast.
`scripts/audit_feature_scale.py` owns fixed dataset/group routing, input checksums
and source arithmetic examples. Statistics weight available rows, not uniformly
weighted symbols or realized sampler transition mass. Extra85 stocks are reported
separately from the original22-stock/four-ETF panel; no future-return cohorts.

Expanded training stocks have maxima about50.55σ return,51.13σ ATR ratio and
80.73σ relative volume. These are uncommon but not absent: per-feature fractions
above5σ roughly0.57–1.20% except RSI0. Relative volume's denominator is the prior
20 observations excluding today, so a ratio above20 is not inherently invalid.
Nine selected extreme training rows recompute exactly from retained source
return/ATR/volume arithmetic. This does not independently verify provider prices,
adjustments or economic causes, nor establish a performance effect.

## Possible next bounded experiment

A single fixed cap on standardized market inputs could test robustness to these
tails, but must be preregistered before fitting, with unchanged controls, return/
risk gates, budget and failure stop. Input statistics do not prove benefit.
Do not clip prices, execution ATR, corporate actions, costs, rewards or baselines,
and do not drop adverse stocks/dates based on outcomes. Retain raw features and
source provenance, fit no preprocessing on validation/test and change only the
explicit new policy-input transformation.

If implemented, clipping must be optional and versioned in dataset/schema/bundle
contracts. Original Normalizer serialization, default generated identities and
legacy inference behavior must remain exact; explicit clipped normalizer factory
must be used consistently by environment and frozen inference. Same initial
model tensors/shapes, execution and sampling; no combination with failed L2,
recent-window or capacity changes. New train/evaluate/save/load/frozen rejection
and immutability checks plus Colab smokes required before100k. No implementation
or allocation is implied by this audit; numeric protocol remains to be frozen.

## Verification

Seven focused checks pass for exact scalar-transform agreement, strict threshold
boundaries, unchanged rows/normalizer, invalid/empty/nonfinite inputs, wrong schema
and invalid scales. Actual dataset/source normalization remain unchanged; all nine
extreme examples match source arithmetic. Full result JSON and compact CSV are
tracked, original data remains local. Project38.74617914/150 unchanged; last
verified account349.23 zero rate/assignments, prior owned sessions stopped.

## Fixed cap5 preregistration

[Numeric preregistration](../configs/experiments/input_clipping_preregistration.json)
freezes control versus market-z cap5 before implementation. Same original means/
scales/full-history111/26 source,64-unit Gaussian recurrent model, default zero
Adam decay and100kseed11/10kcadence. Original selected+final relative2pp,
stock/ETF/median−2pp and drawdown2pp gates retained; failure closes the fixed cap
family without threshold/normalizer/model/reward/step grid. Conditional11/23/37
and prospective requirements unchanged. Shared2-unit allowance starts38.746179136005324
with unchanged project150/130/account150/170 limits. Candidate content identity
will be filled only from deterministic immutable preflight before Colab integration;
cap/gates/budget stay fixed. No allocation or performance claim.
