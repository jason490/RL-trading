# Chronological historical-regime coverage

The failed forest forecasts show wrong-way pooled covariance in2020 and weak
negative-target coverage. Before another learner, audit whether a chronological
history extension can supply broader training regimes. This does not assume
that earlier bars will improve returns or that past declines predict shorts.

[Protocol](../configs/experiments/historical_regime_audit.json) fixes all four
baseline ETFs,2001–2009 inclusive.2001 is the first complete calendar year after
the youngest baseline ETF's listing. No outcomes choose the symbols or dates.
Every XNYS session is required, including the exchange's special closures.

The existing Yahoo adapter and immutable accumulation retain source OHLCV,
dividends and splits; acquisition has bounded transient retries, no fallback or
price correction. The audit reconciles every normalized field against its raw
source, rejects missing/extra/duplicate rows, negative dividends, invalid split
factors and unmodeled capital-gain distributions. Yahoo split-adjusted prices
must have normalized split factor1; source splits are retained, never applied
again. Source consistency is not independent corporate-action verification.
Retrospective Yahoo vintages and ETF-only survivorship limits remain explicit.

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_historical_regimes.py --acquire \
  --output artifacts/historical-regimes/audit.json
# Reuse the immutable snapshot without downloading:
PYTHONPATH=src .venv/bin/python scripts/audit_historical_regimes.py \
  --snapshot data/raw/historical-regimes/SNAPSHOT_ID \
  --output artifacts/historical-regimes/readback.json
```

This audit creates no dataset merge, training run or Colab allocation. Regime
counts use only past63/252-session closes, by symbol and calendar year. They are
price-return descriptors, excluding dividends, not independent outcomes or
future losing-stock labels. The first63/252 rows have no corresponding label.

If accepted, next audit price-unit continuity with the existing snapshot and
freeze a matched training protocol: original history, equal-mass recent-ETF
sampling, and old-ETF augmentation. Match transitions, ETF sampling mass, schema,
costs, evaluation dates and selection rules. This separates sampling changes
from extra market history. Keep both stock and ETF selected/final gates,
drawdown, seeds11/23/37, short-side reports and prospective confirmation. No
retired-test access or silent split-boundary changes are authorized by this audit.

Current stage: implementation and source acceptance pending. The full trading
objective remains unachieved; inspect verification/results before allocating.
