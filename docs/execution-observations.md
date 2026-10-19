# Execution-state observations

The simulator permits signed exposure and profitable short selling after costs.
A policy also needs to observe the protection that remains active before its
next signal executes. Two accounts with the same cash, shares, basis and age can
have different next-session outcomes because their held stops differ. The
[synthetic witness](../results/causal-sampling-2026-10-19/execution-observability-witness.json)
demonstrates this aliasing; it does not establish that prices are predictable.

## Versioned profiles

`TradingEnv(..., observation_profile=...)` and `train(...,
observation_profile=...)` accept three explicit profiles:

| Profile | Portfolio fields | Purpose |
| --- | ---: | --- |
| `legacy` (default) | 5 | Preserve existing model inputs and random draws |
| `execution_masked` | 14 | Same larger architecture as full, with nine zeros |
| `execution_full` | 14 | Expose known execution state |

All profiles begin with exposure, cash/equity, holding age divided by the
20-session limit, unrealized P&L/equity, and equity/initial equity minus one.
Seven basic market features therefore give 12 inputs for legacy and 21 for the
other profiles. The action remains four components; execution, costs and net
log equity rewards are unchanged.

The nine appended full-profile values, in order, are:

1. Stop, near target and far target: `direction * (level - known_close) / ATR`,
   clipped to [-20,20] and divided by 20. A protective stop normally has a
   negative value in both directions; targets normally have positive values.
   Already-crossed levels retain their sign. Missing protection gives zeros.
2. Protection-present mask and near-target-already-filled flag.
3. Last approved stop, near and width action components, or zeros when flat.
4. One share's current price/equity, clipped to [0,1], exposing whole-share
   affordability scale.

The market close and ATR are those available at decision time. No next-session
bar or fitted validation statistic is used. Corporate-action adjustments remain
in the simulator. Flat accounts zero held protection/action flags; the unit
share fraction remains informative. Invalid price/ATR or nonfinite state fails
explicitly. The shared builder is `environment/observations/state.py`.

Price units follow the immutable dataset contract. In the current Yahoo study,
OHLC are retrospectively split-adjusted; the share-unit fraction therefore
describes that research simulator, not reconstructed historical unadjusted
execution prices. This feature does not resolve provider restatements or the
universe's survivorship limitations.

## Frozen compatibility and evaluation

New run metadata and frozen manifests record the ordered state schema, profile
and version. Older bundles without this field mean `legacy`. The loader checks
the schema and the saved model's input shape, then rejects wrong-length or
nonfinite observations. It never infers a profile from vector length: masked and
full have equal lengths but different declared semantics. Callers of `decide`
provide the complete ordered portfolio state for the bundle's profile.

Training checkpoint evaluation, the explicitly gated frozen-family audit, final
audit, standalone frozen verification and
shadow simulation carry the profile explicitly. Existing shadow campaigns keep
their frozen legacy contract. No broker orders are enabled.

## Verification and research limits

Tests construct equal visible account states with different held stops: legacy
and masked inputs are identical, full inputs differ, and the next gap produces
different simulator outcomes. Tests also verify legacy prefixes and RNG parity,
short-side distance signs, partial-exit flags, split scaling, flat state,
Gymnasium/SB3 checkers and save/freeze/load prediction parity without training.

This fixes missing information; it does not establish improved returns or
profitable shorting. A matched Colab experiment must compare full against both
masked and legacy inputs, keeping data, initialization, seeds, steps and costs
fixed. Negative full-period stock returns remain reporting-only labels. Report
both total policy return and reconciled net short P&L, as described in
[short-side.md](short-side.md).

## Declared Colab study

`configs/experiments/execution_observations.json` freezes three seed-11 50k PPO
cells (legacy, masked, full), each preceded by a 512-transition CPU smoke.
Initialization is `wide`, starts use `legacy_uniform`, and validation is the same
26-symbol 2021–2023 development panel. Curves retain 10k checkpoints and a
separate final post-update point. Full can be promoted only if selected and
final means improve by at least two percentage points against **both** controls,
final stock and ETF means do not deteriorate against either, and mean final
maximum drawdown worsens by at most two points. Only a passing screen permits
six additional cells: all three profiles at seeds 23/37. Confirmation evaluates
each control independently; absolute holding comparisons remain separate.

Use `scripts/build_observation_jobs.py --phase smoke|screen|confirm --output ...`
and the existing central Colab fleet with concurrency two initially. Reuse one
frozen committed source export and the original ledger across phases. The study
has a two-unit incremental allowance; project 150/130 thresholds and each
allocation's two-hour deadline still apply. `scripts/report_observations.py`
requires complete retrieval and verified cleanup, matching source/configuration,
identical baselines, ordered profile schemas and final post-update reports.
The short-side reports do not select checkpoints or promotion.
