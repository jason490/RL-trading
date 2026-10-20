# Chronological ETF history audit — accepted for controlled preparation

The preregistered2001–2009 Yahoo panel has2,263 sessions per symbol,9,052 bars
acrossSPY/QQQ/IWM/DIA. Every expected XNYS session is present; every normalized
OHLCV/dividend field matches the retained raw source. Immutable snapshot:
`f44c7b0831d8322c029acf0229b754221d53cc20f8fc641a24afa4d3c1fc093b`.

IWM's2005-06-09 source2-for-1 split is retained while executable split factor
remains1, avoiding double adjustment. Dividend-event counts are37/20/37/109
forSPY/QQQ/IWM/DIA. Independent action verification and historical data vintages
remain unverified; source consistency does not establish unbiased market truth.

A matched past-only rule (close below both63-session and252-session previous
closes) covers23.55% of eligible earlier ETF rows versus8.61% in2010–2020.
Both panels use identical source/action/coverage checks. Initial252 rows per
symbol are ineligible. Calendar lengths differ, so compare fractions, not raw
counts. The earlier interval was fixed chronologically before acquisition;
no future returns selected dates, stocks, samples or checkpoints. Counts are
price-regime descriptors, excluding dividends, not tradable return labels.

[Source/action audit](audit.json), [period comparison](comparison.json),
[exact no-download readback](readback.json),
[protocol](../../configs/experiments/historical_regime_audit.json),
[guide and commands](../../docs/historical-regimes.md).

No training, dataset merge, provider fallback or Colab allocation occurred.
Original training/validation data and the retired test are untouched. Project
usage remains35.871240/150; there is no task-owned runtime to clean up. New
immutable source chunks and audit reports are retained locally; no duplicates
need removal. This is ETF coverage, not historical stock membership/delisting
coverage or evidence of improved policy returns/profitable shorts.

Next freeze a short2010 overlap audit against the existing snapshot to check
price-unit continuity before combining separately retrieved histories. If it
passes, implement a matched three-arm original/recent-ETF/old-ETF comparison with
equal ETF sampling mass, transitions, observations and costs. Keep the original
stock/ETF selected-and-final gates, seeds11/23/37, short-side reporting and
prospective requirements. No automatic learner allocation follows this audit.
