# PPO action initialization — 2026-10-19

The four-arm 50k seed-11 CPU screen is complete, frozen bundles are verified,
and every screen runtime has stopped. Confirmation of the promoted `wide` arm
against a fresh standard control on seeds 23/37 is running separately; no
three-seed or benchmark-success claim is made yet.

All arms use the same original 22-stock/four-ETF dataset, 2021–2023 development
validation, basic features, net-equity reward, costs, signed exposure and learned
four-action contract. No retired test was evaluated.

| Initialization | Selected / final return | Final stocks | Final ETFs | Mean max drawdown |
| --- | ---: | ---: | ---: | ---: |
| Standard | 4.59% / 4.59% | 5.16% | 1.44% | -29.56% |
| Low noise | 7.18% / 7.18% | 8.60% | -0.66% | -19.08% |
| Wide protection | **7.96% / 7.96%** | 8.12% | 7.09% | -29.81% |
| Wide + low noise | 7.43% / 7.43% | 8.27% | 2.79% | -18.34% |
| Buy-and-hold | 30.60% | 32.63% | 19.47% | — |

`wide` passes the declared screen: selected and final gain +3.37 percentage
points, stock gain +2.96 points, ETF gain +5.65 points, drawdown worsens only
0.24 points versus standard. The combined arm passes too but ranks second by
the predeclared final-return criterion. Lower noise alone fails because ETF
returns deteriorate, despite the higher overall average and smaller drawdown.
These are one-seed development results, still substantially below holding.

The short-side objective is also **not met**: `wide` has zero of three declining
stocks with both positive total policy return and positive short net P&L in
this screen. Other arms each have one of three. These future-negative stock
labels are diagnostics only and did not select checkpoints or promote the arm.

## Verification and retained evidence

- Four distinct 512-transition CPU smokes passed on actual Python 3.13.16;
  all four sessions stopped. `smoke.json` records measured/estimated usage.
- Four research cells completed exactly 50k steps with 10k curves and separate
  final post-update points. `screen.json` records the matched source, identities,
  group metrics, short diagnostics, report hashes and exact numeric gate.
- Eight frozen bundles across smoke/screen passed 624 exact CPU prediction
  comparisons, immutable normalization/data and incompatible-schema checks.
- The new standard control exactly reproduces the earlier 50k control: all
  13 policy tensors and six checkpoint returns match (`control-reproduction.json`).
- `screen-compute.json` records screen usage and all owned shutdown evidence.
  No unrelated sessions were stopped.

Training source is `dab2e3360b10f0e06f6b7e2818770e92cf266e80`, captured before
allocation. Subsequent complete-case dataset engineering on main did not alter
the uploaded source. Confirmation uses an exact committed-source export whose
package hash matches the screen and whose `.runtime` ledger and allocation lock,
data and artifacts resolve to the original project paths; no new budget is
created (`confirmation-source.json`).

The interactive screen chart is retained locally at
`artifacts/action-initialization/screen.html`; curves contain incremental
percentage-point changes and use the final post-update record at duplicate
step counts. Publication and duplicate cleanup follow completion of this study.
See [protocol](../../configs/experiments/action_initialization.json) and
[methodology/operations](../../docs/action-initialization.md).
