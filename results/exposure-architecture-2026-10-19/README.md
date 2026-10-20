# Exposure architecture study — completed, failed promotion

Removing account inputs from the exposure head did not improve this screen.
The market-only treatment failed both predeclared control comparisons, so no
confirmation seeds or longer training were run. Native PPO remains the default;
the optional custom policies are retained as tested research implementations.

## Matched 50k PPO results

Seed 11, same 26-symbol 2021–2023 development panel, full execution observations,
wide initialization, legacy-uniform starts, unchanged actions/costs/net-log-equity
reward. All models were freshly trained in Colab from one source revision.

| Architecture | Selected return | Final return | Final stocks | Final ETFs | Mean final drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native full-state | 22.59% | 22.59% | 25.13% | 8.64% | -32.50% |
| Two branches, unmasked | 5.41% | 5.41% | 5.13% | 6.95% | -32.76% |
| Market-only exposure | -1.02% | -3.64% | -5.59% | 7.04% | -32.40% |
| Same-symbol buy-and-hold | 30.60% | 30.60% | 32.63% | 19.47% | — |

Treatment's final mean is 9.05 percentage points below the matched unmasked
branch control and 26.24 points below native. Against unmasked, stock return
falls 10.71 points; ETF return changes only +0.08 points. The slight drawdown
improvement does not compensate for those losses. The larger branch layout also
underperforms native, so the full gap cannot be attributed solely to the input
mask. These are single-seed development results, not proof that account inputs
must always help or that every branch architecture is unsuitable.

## Shorting requirement remains unmet

The treatment's final results on the three negative-holding stocks:

| Stock | Holding | Policy total | Net short P&L on $100k initial equity |
| --- | ---: | ---: | ---: |
| AMZN | -7.73% | -52.16% | -$25,748.94 |
| DIS | -47.38% | -40.65% | -$2,557.13 |
| NEE | -14.63% | -43.40% | -$8,506.06 |

None has positive total return and positive short contribution. Both controls
also have negative total returns on all three; their positive short contributions
are outweighed by long losses. The declining-stock cohort remains reporting-only,
never a model input, checkpoint metric or reason to override the failed gate.
See [short-side audit](short-side.json) and [all 156 symbol/phase rows](per-symbol.csv).

## Verification and provenance

Source: `4b05a6aba75a16d55c111c8165b83518a8c71ad7`, package hash
`a57b2e1da32631a91823bd26feb147267f4e9057947579f2d6971bfb09e8cbf0`.
The source export shares the original cumulative ledger/allocation lock. Actual
Colab kernel was Python 3.13.16. No dependency or CLI installation was upgraded.

Offline regression passed 237 tests. Three 512-step CPU smokes preceded three
50k CPU cells. Six selected/final smoke input contracts pass, including exactly
invariant market-only exposure means under account perturbations. Six frozen
bundles pass 468 exact predictions, immutable normalization/data and invalid-
input checks; loading also validates each policy class and exact mask buffer.
Native smokes and 50k research reproduce the prior full-state runs' selected/final
13-tensor policies and every checkpoint score exactly. All source observations,
actions and rewards remain compatible with legacy native bundles.

All six owned sessions stopped with verified cleanup. Conservative study usage:
**0.060740 units**, cumulative **35.748563/150**, leaving **114.251437** project
units and **94.251437** before teardown at 130. Measured meter-window debits total
0.04, and rounded balances changed 350.73 to 350.69; estimates/readings are
reported separately, never added. Each allocation included setup, idle time and
retrieval in its two-hour deadline. No budget stop, retired-test evaluation or
broker orders occurred.

- [Protocol](../../configs/experiments/exposure_architecture.json),
  [methodology](../../docs/policy-architecture.md), [preflight](preflight.json).
- [Screen and both gates](screen.json), [compute](compute.json),
  [frozen checks](frozen-checks.json), [smokes](smoke.json).
- [Trained smoke input contracts](trained-input-contract-smoke.json),
  [native smoke parity](native-smoke-parity.json),
  [native research parity](native-screen-parity.json).

Interactive checkpoint charts are generated at
`artifacts/exposure-architecture/screen.html`; extracted models, selected/final
reports, complete curves and frozen bundles are retained locally and published
through allowlisted GitHub assets. Original provider bars are not redistributed.

## Next bounded diagnostic

Retain account context and test uniform averaging of already-trained stateless
PPO/SAC seed predictions before allocating more training. Fix member lists and
weights in advance, verify matching action/feature/feed/normalization contracts,
and replay one shared account path rather than average independent returns.
Use development-only bars, compare both selected and final phases, and report
individual references plus every ensemble candidate. No declining-stock label
may choose a member or trade. A single ensemble built from three seeds is not
three independent ensemble runs; any promising result still needs separate
robustness and prospective confirmation. This next diagnostic is not yet run.

## Published artifacts and cleanup

[Download frozen models and interactive reports](https://github.com/jason490/RL-trading/releases/tag/exposure-architecture-2026-10-19). All 18 server
asset SHA-256/size pairs match ([verification](release-verification.json)).
Cleanup removed 27 duplicate archives/markers, 314,922,712 logical bytes
(300.33 MiB), with all 326 retained hashes unchanged. Final archives matched
every extracted file; payloads matched retained source/data/plans before removal.
Original models, complete curves/reports, source export, immutable data and
cumulative ledger remain ([cleanup](cleanup.json)).
