# Reward objective study — completed; screen failed

| Reward | Selected return | Final return | Final stocks | Final ETFs | Mean final maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| net_log_equity | 22.59% | 22.59% | 25.13% | 8.64% | -32.50% |
| net_initial_equity_change | 23.38% | 20.43% | 22.81% | 7.36% | -32.65% |

Matched buy-and-hold: 30.60% overall, 32.63% stocks and 19.47% ETFs. The candidate improves selected return by only 0.79 pp, then loses 2.16 pp versus the control at final weights. Final stocks lose 2.32 pp and ETFs 1.28 pp versus control. The predefined gate fails; no seeds23/37, promotion, or longer reward run is authorized by this result.

## Short-side requirement

All three negative-holding stocks have positive final short P&L but negative total returns. They therefore fail the joint profitability objective. No future decline label selected weights or shaped rewards.

| Candidate final symbol | Total net return | Net short P&L |
| --- | ---: | ---: |
| AMZN | -13.55% | $62.08 |
| DIS | -44.43% | $459.49 |
| NEE | -14.93% | $1,279.05 |

## Verification and operations

Two512-step smokes and two50k research cells completed on actual Google Colab CPU, Python3.13.16. All four bundles pass312 exact CPU prefix predictions, deterministic repeat, immutable normalization, and schema/feed/nonfinite rejection. The default control exactly reproduces all prior checkpoint scores and all13 selected/final tensors ([parity](native-control-parity.json)). No research training occurred locally.

The [protocol](../../configs/experiments/reward_objectives.json) was committed before allocation. Both arms used one frozen source revision27f0d343b263a4a35ff5a1837ce1cb2efcbfb9f7, the development-only26-symbol dataset, unchanged costs/actions/discount/starts and full account state. No retired-test data or broker orders were used. One-seed development screening is not independent confirmation.

[Screen and checkpoint curves](screen.json), [smoke acceptance](smoke-acceptance.json), [research frozen checks](frozen-checks.json), [compute](compute.json), [methodology](../../docs/reward-objectives.md).

All four owned sessions stopped and shutdown was verified. Study conservative usage0.046815 units (measured account debit0.03); project35.795378/150, remaining114.204622, reported rate0.00/hour and zero active assignments. Shared ledger and recovery records remain intact. Publication verification and duplicate cleanup follow before removing transfer copies.

## Interpretation and next evidence

Changing from log to initial-capital equity differences did not solve final instability or long losses on declining stocks. Keep net-log as the default. This rules out this reward-only hypothesis under the tested settings; it does not justify adding arbitrary short bonuses or extending unsuccessful training. Any next experiment requires a new diagnosis and preregistered matched protocol; prospective evidence remains necessary.
