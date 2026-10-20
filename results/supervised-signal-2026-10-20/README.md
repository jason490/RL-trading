# Supervised signal feasibility — completed; neither MLP passed

This is predictive error on the internal 2018–2020 training probe, **not trading return**. Fit labels end in 2017. No project-validation or retired-test labels were used. No PPO policy changed.

| Predictor | Probe MSE | Mean daily rank correlation |
| --- | ---: | ---: |
| Constant | 0.00594294 | Undefined (constant forecasts) |
| Ridge masked | 0.00578529 | 0.0089 |
| MLP masked, three-seed mean | 0.00627077 | 0.0422 |
| Ridge full | 0.00578210 | 0.0107 |
| MLP full, three-seed mean | 0.00668966 | 0.0393 |

Masked MLP mean MSE is 5.52% worse than the constant and 8.39% worse than its ridge control; full is 12.56% and 15.70% worse. Neither has a passing seed. Both worsen versus the baselines in 2020. Full-context MSE is worse than masked in all three paired seeds. Positive rank correlation alone cannot pass the fixed combined gate. No PPO warm-start allocation follows.

Ridge reduces MSE about 2.7% versus the constant, but its mean daily rank correlation is only 0.009–0.011, below the declared 0.02 signal threshold. This descriptive control result does not rescue the failed MLP gate or establish profitable trading.

## Actual execution and verification

One Google Colab CPU allocation completed the two discarded prefix updates and 12,000 fixed full-fit updates, plus constant/ridge fits. Actual Python 3.13.16; no GPU or local fitting. Source f6c20e427441d96a5c120901210cb241196bc591, clean captured revision. Each candidate uses identical initial tensors per paired seed, fixed fit-only preprocessing, and final weights only. Fit loss is logged every 200 updates without probe checkpoint selection.

All six saved predictors restore on all 18,616 immutable probe rows: 111,696 MLP forecasts. Local repeats are bitwise exact; maximum cloud/CPU difference is 5.96e-8, within the predeclared 1e-6 bound. Constant/ridge predictions, labels, sessions and symbols match; normalization stays frozen and wrong schema/source/nonfinite inputs are rejected. Returned metrics and gate were independently recomputed.

[Result and gate](result.json), [restoration proof](restoration.json), [training loss](training-loss.json), [source metadata](metadata.json), [data preflight](preflight.json), [compute](compute.json), [protocol](../../configs/experiments/supervised_signal.json), [definitions](../../docs/supervised-signal.md).

One owned session stopped and termination verified. Project 35.861240/150 units; this study conservatively adds 0.004601 units, remaining 114.138760. Account 350.61, rate 0.00, active assignments 0. No orders or trading returns are produced by these predictors. Publication verification precedes duplicate cleanup.

## Interpretation and next diagnosis

The current supervised MLP is rejected; more updates or transferring these weights to PPO is not justified by this gate. Inspect forecast bias and variance by year to distinguish calibration/extrapolation error from the weaker ranking signal before proposing another model. Any calibration or combination must be fitted on earlier training-era data under a new protocol, with fixed probe gates; it cannot use this probe as a fitting set or relabel the failed study. Overlapping targets, correlated symbols and a retrospective universe prevent an independent profitability claim.
