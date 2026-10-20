# Fixed-forecast common-component diagnosis

The supervised MLPs failed their predictive gate. Their positive within-day ranks
coexisted with negative pooled prediction/target covariance. This separate
preregistered diagnostic tests the common time-varying forecast component without
fitting or changing any model. The original failure remains recorded.

## Fixed recipe

For each session, feature arm and seed:

`centered forecast = raw MLP forecast - mean(raw MLP forecasts) + mean(frozen ridge forecasts)`

Both means use the same complete 26-symbol panel. The ridge belongs to the same
feature arm and was fitted only on labels available by 2017. Coefficients are
fixed at 1/-1/+1. No weight, intercept, window or threshold is estimated from
the probe. This changes forecast levels while preserving all within-day pairwise
differences and rankings. These remain gross return forecasts, not portfolio
returns, trading actions or guaranteed short profits.

## Availability and universe

Every configured tradable symbol must occur exactly once per session. The common
decision cutoff is the maximum feature-availability timestamp in that panel.
Every input must be available on the decision session, and the cutoff must be
strictly before every row's next entry-session UTC midnight. This conservative
bound requires no assumption about intraday open times. Incomplete, duplicated,
nonfinite, non-UTC or late panels fail closed; no subset selection, carry-forward,
or future declining-stock label is allowed.

The transformation function receives forecasts, symbols, sessions and availability
metadata. It has no label interface. Outcome labels are read only afterward for
descriptive predictive scoring. The fixed universe is retrospective; this does
not solve survivorship bias or prove future multi-symbol feed availability.

## Protocol and reproducibility

[The protocol](../configs/experiments/signal_centering.json) fixes the original
six model forecasts, complete panel, same-arm ridge anchors and arithmetic recipe.
Before transformed scoring, preflight revalidates the original failed report,
restoration receipt, immutable target payload, cached predictions, model checksums
and all availability panels. Commit protocol, implementation and preflight first.

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_signal_centering.py \
  --preflight-only --output artifacts/signal-centering-preflight.json

# After committing source, protocol and the tracked preflight receipt:
PYTHONPATH=src .venv/bin/python scripts/replay_signal_centering.py \
  --output artifacts/signal-centering
```

The full command requires clean committed source and the identical preflight,
rejects an existing output directory, preserves every protected input hash and
writes transformed forecasts separately. It reuses the original numeric
MSE/year/IC/seed gate without modifying it, but labels the new outcome
**descriptive only**. Pairwise raw-to-transformed errors and rank preservation are
reported for every seed and arm. No final-test data, project-validation labels,
new model fit, normalization update, Colab allocation or PPO training occurs.

The 2018–2020 internal probe has already influenced this diagnostic and is
training-era development evidence, not independent confirmation. Even a positive
result cannot relabel the original study as passed, promote an RL policy, prove
stock/ETF outperformance, or trigger PPO allocation automatically. A later
causal feature/trading experiment would require its own preregistered controls
and prospective evidence.

## Completed result

Centering lowers masked/full MLP error by7.39%/12.96%, preserves ranks and beats
the constant by2.28%/2.03%. It remains0.38%/0.70% worse than ridge, with zero
passing seeds and failed yearly ridge constraints. The unchanged descriptive
gate fails. Source hashes stay unchanged; no training or Colab allocation.
Original failure and promotion restrictions remain.
[Evidence and charts](../results/signal-centering-2026-10-20/README.md).
