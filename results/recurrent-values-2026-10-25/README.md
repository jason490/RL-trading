# Saved recurrent value-target consistency

The fixed diagnostic completed52 stochastic training-window replays and26,624
transitions in40.44 seconds. It used the accepted legacy RecurrentPPO control's
selected80k and final100k weights, fixed original26-symbol panel, last513 training
observations ending2020-12-02 and512 transitions per window. Protocol and collector
were committed as2011e33 before execution. This is local inference, not training.

## Outcome

Primary summaries use the first256 decisions of each window, with at least256
later observed transitions. Errors are in discounted log-equity reward units,
not portfolio return percentages. Values predict stochastic training-policy
rewards under unrestricted exposure; they are not deterministic unit-exposure
development scores.

| Target, all26 symbols | Selected RMSE | Final RMSE | Selected descriptive EV | Final descriptive EV |
| --- | ---: | ---: | ---: | ---: |
| Observed discounted rewards, zero tail |0.073052|0.074707|0.029178|−0.083630|
| Critic-bootstrapped discounted return |0.071639|0.073497|0.034502|−0.077577|
| GAE lambda return |0.021894|0.019783|0.490642|0.548952|

Final value predictions average0.006786 against observed zero-tail targets−0.003868,
a descriptive positive difference0.010654. Stocks have final zero-tail EV−0.094660;
ETFs0.247949. Per-group values are in [summary.json](summary.json). Largest
bootstrap effects on primary targets are0.011478 selected and0.013033 final.

On these particular paths the final checkpoint agrees better with shorter,
self-bootstrapped GAE targets while agreeing worse with the longer observed
reward sums. This differs from interpreting sparse rollout explained variance
as evidence of broad expected-return accuracy. It does not prove a critic bug,
overfitting, a causal explanation for development deterioration or a preferred
reward/GAE coefficient.

## Interpretation limits

There is one seeded stochastic path per symbol/phase. Overlapping observations
are correlated; policies induce different portfolios and rewards even with common
Gaussian draws. Zero-tail sums are censored; bootstrapped and GAE targets partly
reuse the critic. They are not independent, unbiased calibration labels. These
training windows do not prove every row was sampled during original fitting.
No checkpoint, symbol, threshold or experiment was selected from these results.
All failed families, including VIX, remain closed.

## Verification and retention

[Independent verification](verification.json) checks all52 windows/26,624 rows,
fixed panel/seeds/end dates, direct NumPy discount-matrix arithmetic, SB3 float32
GAE targets and all group metrics. Maximum discounted-target error4.45e−16,
GAE error9.39e−8 and group-metric error0.0. Every forward value was cross-checked
against a separate critic calculation from the same previous hidden state;
caller recurrent states remained unchanged. Actual learned weights/Adam records,
source files/data and original usage ledger are unchanged. Sixteen focused
non-optimizing target/collector/restoration checks pass; new code passes Ruff.

Full derived reward/value/target traces, hashes and the terminal progress journal
remain under `artifacts/recurrent-values-2026-10-25/`. No temporary output or
live worker remains; worker exited0. Source models stay in the
[VIX research release](https://github.com/jason490/RL-trading/releases/tag/vix-context-2026-10-25).
No new model duplication, provider requests, development replay, retired-test
access, optimizer steps, Colab allocation or broker orders. Project usage remains
39.272458/150; last official idle account348.80, floor150/teardown170 unchanged.

Next assess a separately fixed training-only benchmark/noise decomposition before
any new training hypothesis. No additional reward term, coefficient search,
longer run or reopening of failed families is authorized by this diagnostic alone.
