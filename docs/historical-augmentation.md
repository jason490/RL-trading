# Matched historical ETF augmentation

The [source/overlap audits](historical-regimes.md) accepted a chronological
2001–2009 four-ETF panel. The [fixed experiment](../configs/experiments/historical_augmentation.json)
now compares original episodes, recent-ETF oversampling and earlier-ETF
augmentation. It tests extra market history against extra ETF sampling, without
selecting future losing symbols. Three Colab smokes have passed; the50k screen is running.

## Dataset arms

Original seven indicators, normalizer and every original episode are identical
across arms. Validation remains the same26-symbol2021–2023 development panel.
The original arm contains26 training episodes. Recent augmentation appends four
duplicate training ETF episodes; earlier augmentation appends four independently
prepared2001–2009 ETF episodes. Each earlier series has its own causal feature
warmup and final20-session purge; old bars never change existing feature warmup
or split boundaries. All payloads are immutable and checksummed.

The augmented arms have8/30 expected ETF episode probability, including4/30
additional episode probability, versus4/26 original. `full_window_uniform`
samples an episode uniformly, then a start allowing every configured transition
(512 training,128 smoke). This avoids different near-boundary truncation rates.
Record realized starts/transitions by symbol and earlier/recent history; nominal
probabilities do not guarantee identical realized sample counts in a short run.
Insolvency may still legitimately terminate an episode. No price/outcome class
enters sampling. Existing legacy and causal sampler behavior stays unchanged.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_historical_arms.py
```

[Preparation evidence](../results/historical-augmentation-2026-10-20/preflight.json)
records70,200/81,000/78,976 training rows for original/recent/earlier arms.
Earlier history adds8,776 independently warmed/purged rows. Readback, original
observations, validation and normalization all match exactly. New earlier
observations may lie outside the original normalizer's range; preprocessing is
frozen deliberately, rather than adding a normalization confound.

## Fixed fitting and gates

Three512-transition CPU Colab smokes precede seed11 screens of50,000 transitions
per arm. Native PPO, wide initialization, full execution state, net-log-equity
reward,1,000-step rollouts,50 batch size and10k checkpoint cadence are fixed.
Both selected and final candidates must improve at least2 percentage points
against each control, avoid worsening final stock/ETF groups, and keep drawdown
within2 points. Both phases also need stock and ETF holding excess and median
stock excess at least−2 points. Failure stops the screen; no longer run/grid.

Only a passing candidate proceeds to seeds23/37 with both controls. Confirmation
requires the fixed multi-seed gates and at least two target-passing seeds.
Profitable shorts remain separate reconciled reporting. These gates guide
research on repeatedly used development data; frozen prospective evaluation is
still required. Retired2024–September2026 data stays excluded. Yahoo revisions,
whole-share adjusted-unit limitations and survivor-stock bias remain unresolved.

The study is capped at2 incremental units, shares the original150-unit ledger
and130-unit working stop, and uses bounded2-hour owned allocations. CPU profiling
supports this small-network workload. Actual Colab Python must be rechecked.
Metered job/artifact integration and source freeze remain required before launch.

Version15 now binds dataset identities, fixed protocol, CPU jobs and full-window
sampling on the launcher and worker. Returned archives check PPO/init/state/
reward settings, checkpoint curves, complete frozen metadata and reconciled
symbol/history sampling counts. Invalid checksummed contracts fail. Sampling
statistics are copied before parity resets to prevent nested-count mutation.
Actual metered Colab acceptance is next; no fitting result is implied.

## Colab acceptance

Three512-transition CPU smokes pass on actual Python3.13.16; all three owned
sessions stopped and234 restored predictions are exact. Archive checks and
normalization/schema rejection pass. Conservative project35.888374 after smoke.
The50k three-arm seed11 screen is launched from frozen revision4be5219 and shares
the original2-unit study allowance. [Evidence](../results/historical-augmentation-2026-10-20/README.md).

## Fixed screen reporting

After complete artifact retrieval and owned shutdown:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_history.py \
  --state .runtime/history-screen-state.json \
  --output artifacts/historical-regimes/screen-report.json
```

The reporter revalidates each archive against its fixed plan, verifies immutable
payloads and shared source/configuration, and requires identical full holding
paths and symbol panels across both phases/all arms. Short-side cohorts are
reporting-only. Both dual-control gains and absolute selected/final group/median
thresholds must pass; development promotion never sets objective achievement.
Tests reject one-control-only gains, phase-specific ETF/median failures, excess
drawdown and nonfinite/incomplete metrics. No new training is run by reporting.
