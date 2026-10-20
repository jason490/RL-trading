# Fixed-forecast centering — completed; descriptive gate failed

No new model fitting or Colab allocation. The transformation used only complete same-session forecast panels, with source weights, normalization, predictions and target data unchanged. The original supervised study remains failed.

| Arm | Raw MLP mean MSE | Centered mean MSE | Reduction from raw | Centered versus ridge | Daily rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| masked | 0.00627077 | 0.00580726 | 7.39% | 0.38% worse | 0.0422 |
| full | 0.00668966 | 0.00582247 | 12.96% | 0.70% worse | 0.0393 |

Both centered arms beat the constant by just over2%, but remain0.38%/0.70% worse than the corresponding ridge. The gate requires at least2% improvement versus BOTH baselines, at least two passing seeds, sufficient rank correlation and yearly noninferiority. Neither arm has a passing seed;2018 and2020 also fail the ridge yearly constraint. Every paired seed improves from its raw MLP forecast, but this does not satisfy the competitive gate.

## Causality and integrity

The preregistered rule subtracts each raw MLP panel mean and adds the frozen same-arm ridge panel mean. All716 sessions have exactly26 configured symbols. The latest same-session feature availability is before every next entry-session UTC midnight. The transformation has no label argument, no learned coefficient and no missing-panel fallback. Daily ranking correlations remain unchanged for all six cases.

The source failed report, restoration proof, immutable data, cached forecasts and all model files remain checksum-identical. The complete data/availability proof was committed in2c407a5 before scoring. These are touched2018–2020 internal-probe results, not independent confirmation or trading returns. No project-validation or retired-test labels, simulation changes, orders or policy allocation were used.

[Protocol](../../configs/experiments/signal_centering.json), [source and availability preflight](preflight.json), [complete summary](summary.json), [paired seed CSV](paired.csv), [interactive comparison](comparison.html), [methodology](../../docs/signal-centering.md).

## Decision and operations

Do not transfer the failed predictors or centered outputs to PPO under this diagnostic. Retain the original failure and the new descriptive negative result separately. Additional adjustments chosen from this probe would be further development, not fresh evidence.

A distinct tabular tree ensemble is the next learner to investigate under a new fixed protocol, rather than continuing this MLP centering family. Check Python/dependency compatibility and do lower-credit CPU acceptance before allocation. The same target folds, normalization boundaries and predictive/trading distinction remain required.

No new Colab units were spent; cumulative project usage remains35.861240/150. All previously owned sessions are stopped. No temporary downloads or model copies were created. The unique transformed prediction file remains in ignored local artifacts; source datasets, models, ledger and failure evidence are retained.
