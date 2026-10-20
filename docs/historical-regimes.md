# Chronological historical-regime coverage

The failed forest forecasts show wrong-way pooled covariance in2020 and weak
negative-target coverage. Before another learner, audit whether a chronological
history extension can supply broader training regimes. This does not assume
that earlier bars will improve returns or that past declines predict shorts.

[Protocol](../configs/experiments/historical_regime_audit.json) fixes all four
baseline ETFs,2001–2009 inclusive.2001 is the first complete calendar year after
the youngest baseline ETF's listing ([IWM inception: May22,2000](https://www.ishares.com/us/products/239710/iwm-ishares-russell-2000-etf)). No outcomes choose the symbols or dates.
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

## Accepted audit evidence

[Results](../results/historical-regimes-2026-10-20/README.md) confirm9,052 bars,
complete four-symbol coverage and exact raw/normalized agreement. Cached
readback produces the identical report without downloading. Past-only joint
63/252-session declines cover23.55% earlier versus8.61% recent eligible rows.
The existing2010–2020 comparison is checked under the identical audit, using
only training dates from the retained snapshot. Reproduce it with:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_historical_regimes.py \
  --output artifacts/historical-regimes/new-comparison.json
```

No dataset merge or model fit occurred. Audit a fixed2010 overlap next to check
cross-snapshot price-unit continuity before preparation. The full trading
objective remains unachieved; inspect verification/results before allocating.

## Preregistered cross-download overlap

[Seam protocol](../configs/experiments/historical_overlap.json) fixes a separate
2009-12-01 through2010-03-31 download for all four ETFs. December overlaps the
earlier snapshot; January–March overlap the existing recent snapshot. This
covers both sides of the seam and distribution periods. Source/action audits
run first; row sets and raw split events must agree. Fixed price tolerances
are1e-6 absolute plus1e-7 relative, cash-dividend tolerance1e-8, volume/split
factors/times exact. Retrieval timestamps/revision IDs may differ. Any mismatch
fails; no rescaling, widening tolerances after seeing results, fallback or merge.

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_historical_overlap.py --acquire \
  --output artifacts/historical-regimes/overlap.json
```

A successful local overlap is evidence of consistent reported units at the seam,
not proof of full-history action accuracy or historical publication vintages.
Eleven overlap tests and eight history tests pass before acquisition. Actual
source overlap acceptance now passes:332 bars, zero numeric differences,
matching actions/times and exact cached readback.69 related tests pass;
[results](../results/historical-regimes-2026-10-20/README.md#cross-download-seam-overlap--accepted).
No merge or training allocation occurred.

Next preparation must preserve existing episodes and fit-only normalization.
Construct old training episodes separately with independent warmup/end purging.
Match expected ETF sampling mass between recent-ETF and old-ETF augmentation,
use full512-transition windows in all three arms, and record realized transition
mass by symbol/period. Freeze data identities and numeric training gates before
allocation. Seam agreement alone does not justify model promotion.
