# Forest signal feasibility — completed, gate failed

Six fixed 128-tree forests were fitted in Google Colab CPU on Python 3.13.16.
Two discarded prefix trees plus 768 final trees total 770 fitted trees; these
are not RL transitions. The source revision was `cfe6eb2fa857e39d2ce34e715d5c5ba37bae838f`.

| Predictor | Probe mean squared error | Worse than own ridge |
| --- | ---: | ---: |
| Constant | 0.00594294 | — |
| Masked ridge | 0.00578529 | — |
| Full ridge | 0.00578210 | — |
| Masked forest, 3-seed mean | 0.00672440 | 16.23% |
| Full forest, 3-seed mean | 0.00679195 | 17.47% |

Both arms fail, with zero passing seeds. Full features have higher error in all
three paired seeds. Both arms fail the 2020 yearly constraint; daily rank IC
(0.01579 masked, 0.02457 full) does not override the fixed error gate. No
PPO transfer, trading return, profitable-short claim or model promotion follows.

The immutable payload contains 45,266 completed fit rows and 18,616 internal
probe rows across 26 symbols. Fit labels end in 2017; probe decisions span
2018–2020. That probe lies inside older RL training history and is not fresh
market evidence. Project validation and retired test data were not used.

All 111,696 forecasts were restored from six non-pickle numerical bundles and
checked against the returned sklearn predictions, with exact local repeats,
frozen normalization and schema/source/nonfinite rejection. See
[restoration](restoration.json), [metrics and gates](result.json),
[fit summaries](fit-summary.json), [metadata](metadata.json),
[plot](comparison.svg), and [compute](compute.json).

The run used 0.010000 conservative incremental units: project total
35.871240/150, leaving 114.128760. Account balance is separately 350.60;
reported rate is zero with zero active assignments. The owned runtime was
stopped and cleanup verified. Forecasts and fitted models remain available. Ten published assets passed server
SHA-256/size checks. Duplicate-only cleanup removed 11 files (15,788,589 bytes);
all 246 retained file hashes are unchanged.

[Downloadable frozen models and evidence](https://github.com/jason490/RL-trading/releases/tag/forest-signal-2026-10-20).
[Protocol](../../configs/experiments/forest_signal.json),
[guide](../../docs/forest-signal.md), [cleanup](cleanup.json).

The declining-stock objective remains positive total return **and** positive
net short P&L after costs. This forecast experiment does not demonstrate either.
Stop this fixed forest configuration; diagnose regime/target generalization
before committing another training protocol. Do not expand this failed setup
with more trees or longer PPO training by default.

## Why the fixed forests failed

A no-fit [error decomposition](forecast-errors.json) uses all saved forecasts
without changing predictions, gates or checkpoints. Both exact identities pass:
`MSE = var(target) + var(prediction) - 2 cov(target,prediction) + bias²`
and `MSE = row-weighted daily mean-error² + within-day error variance`.

In2020, mean pooled forecast/target covariance is−0.0006715 masked and
−0.0007574 full, versus positive0.0002562 for full ridge. Forest error therefore
includes wrong-way variation, not merely a constant calibration offset. Daily
common-error terms contribute0.011698/0.011816 of total2020 MSE
0.015874/0.016256. These are descriptive error components, not causal claims.

In2018, roughly46.35% of gross targets are negative, but forests predict negative
for only0.50%/1.46% of rows (mean across seeds). In2020 their negative forecast
precision is only22.94%/25.36% despite39.20% negative targets. These are overlapping
20-session gross-return signs, not trading signals or net short profitability.
Labels are reporting-only; no label-based sampling or checkpoint selection occurs.

The pooled bias is small (0.000284/0.000398), so correcting one global mean is
not an adequate explanation. No second forest centering experiment is justified
by this evidence alone. Stop this family rather than add trees or PPO updates.
Six decomposition tests pass, including unequal panel weights, undefined sign
metrics and invalid vectors. Original source prediction SHA remains unchanged.

Next investigate training-regime coverage before selecting another learner:
audit a fixed chronological pre-2010 ETF history extension against existing
2010–2020 history. This is a proposed data audit, not permission to silently merge
providers, use validation decline labels, or train on retired data. First freeze
the date range, symbols, provider/actions/coverage checks and comparison design.
New fitting requires its own bounded matched protocol and original development
gates; all-market and prospective requirements remain intact.
