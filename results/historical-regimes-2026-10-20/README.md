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

The fixed seam overlap now passes (details below). Next implement a matched
three-arm original/recent-ETF/old-ETF comparison with
equal ETF sampling mass, transitions, observations and costs. Keep the original
stock/ETF selected-and-final gates, seeds11/23/37, short-side reporting and
prospective requirements. No automatic learner allocation follows this audit.

## Cross-download seam overlap — accepted

The separate frozenDec2009–Mar2010 download contains332 bars (83 sessions per
ETF). All88 earlier-overlap and244 recent-overlap rows have exactly zero
OHLCV/dividend differences, identical normalized split factors, raw split
events and open/event/availability times. Cached readback is identical.
[Overlap checks](overlap.json), [readback](overlap-readback.json),
[preregistered protocol](../../configs/experiments/historical_overlap.json).

Source snapshot `dab959f56792b6347ad068c8d275eb4c5d51190292b9fcd6102971d2dc33836b`
and every original snapshot remain unchanged. No tolerance widening, price
rescaling, dataset merge or fitting.69 history/overlap/calendar/core/regime tests
pass. Local seam consistency does not independently verify full historical
actions/vintages or solve surviving-stock bias. Project remains35.871240/150.

For the next matched preparation, keep existing validation episodes and the
original training-only normalizer identical. Build old history as separate
training episodes with independent feature warmup and20-session end purging,
so adding early bars does not alter existing training/validation observations.
Compare original episodes, additional recent-ETF episodes and additional old-ETF
episodes. The latter two must have equal expected ETF sampling mass. Require
full512-transition training windows in every arm and report actual sampled
transitions by symbol/history period; variable near-boundary episodes otherwise
confound nominal sampling mass. Freeze the complete numeric training protocol
and prepared identities before any Colab smoke or fitting.
