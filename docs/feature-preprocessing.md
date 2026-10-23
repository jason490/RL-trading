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

## Explicit cap5 transform implemented; immutable protocol finalized

`data/generation/clipping.py` owns `ClippedNormalizer` and the clipping-only
payload fork. Original `Normalizer` remains a three-field frozen dataclass with
unchanged transforms/serialization. The shared `normalizer_from` factory accepts
only the legacy fields or an explicit positive finite `clip_abs`; unknown fields
reject rather than silently dropping preprocessing. `ClippedNormalizer.fit`
rejects: candidate means/scales are copied from frozen source, not refitted.

The environment and FrozenPolicy already share that factory. New candidate schema
`daily-indicators-v1-zclip5-v1` and normalizer metadata make clipping explicit in
checksummed bundles; wrong schema requests reject. Default old bundles use the
unchanged class/three fields. Only transformed market features are capped; raw
feature matrices, bars, ATR execution and portfolio-state observations remain exact.

[Final protocol](../configs/experiments/input_clipping.json), canonical SHA-256
`ad11f4beb9022b73ad9c1d7843a5714614fc4cc02a251a7c4d340afbd8163877`, binds the
original preregistration hash and deterministic candidate identity. Cap5, original
relative/absolute/risk gates and2-unit allowance remain unchanged. Candidate:
`c895d6af1c98f1326f9f585009e8ad0b224ca13e885027bbc7a02c4ebc5a8894`.
Original normalizer SHA staysb3c8a298…; candidate includes explicit bound with
SHAde3ab713…. Raw episode hash is identical across arms. Same111 training symbols,
298,946 training rows/19,058 development rows, no retired test reconstructed.

83 focused checks pass: legacy serialization/transforms, immutable clipping and
invalid bounds, schema/unknown fields/source corruption, fixed-action rewards/
costs/quantities/events equivalence, untrained recurrent save/load/frozen prediction
and hidden-state parity, normalization immutability and wrong-schema rejection.
Actual preflight verifies all candidate transformed values≤5 and exact raw data/
original means/scales. Legacy default generator reruns its training-only statistics
and produces the exact oldc2e6c58c… identity. This is an identity check, not candidate
normalizer refitting or policy training. Sourcecdb93a33… unchanged.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python \
 scripts/preflight_input_clipping.py --output .runtime/input-clipping-preflight.json
```

[Actual evidence](../results/input-clipping-2026-10-23/preflight.json). Dedicated
Colab protocol/plan/worker/artifact/report wiring, full untrained construction/
checkers and two512 CPU smokes still required before100k. No allocation or
performance result; project38.74617914 and last official account349.23 unchanged.

### Overflow rejection and legacy restoration

Standardization must produce finite values before clipping. Finite raw inputs can
still overflow when divided by a very small scale; both inference preprocessing
and the distribution audit reject this condition instead of converting infinity
to an apparently valid capped value. The focused suite passes85 checks (two
existing Gymnasium warnings about unbounded observation spaces).

[Legacy restoration](../results/input-clipping-2026-10-23/legacy-restoration.json)
records156 exact saved-model versus frozen-bundle recurrent predictions across
the prior control/L2 bundles on the current host. This preserves legacy inference
behavior; it is not full-path parity with the original Colab reports. Clipped
bundles require the clipping-aware source and their explicit new feature schema;
historical clients that ignore extra normalizer fields are unsupported.
