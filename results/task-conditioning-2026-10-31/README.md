# Task-conditioning preflight — 2026-10-31

The cached source audit confirms107 stock annotations and four broad ETF
annotations on the pinned111-symbol training payload. Dataset/normalizer identity
matches the fixed protocol. Dataset bytes, protocol bytes and the cumulative
usage ledger are unchanged by the audit. No fitting, development/test evaluation,
provider request or Colab allocation occurred.

[Evidence](type-evidence.json) deliberately supplies no `known_from` dates:
the universe configuration is a retrospective snapshot and does not establish
historical type availability. It is not a valid historical classification
manifest. Training admission remains pending historical primary-source records
covering the exact training universe, with reviewed availability at or before each
first training session. Existing survivor/revised-history limitations persist.

The fixed [protocol](../../configs/experiments/task_conditioning.json) compares
masked shared, tagged shared and tagged routed SAC. All use full22 observations,
64x64 trunks and equal100k-transition screens. Smokes retain research64x64
geometry while reducing replay batch/warmup. Checkpoints are selected on the
fixed training window; no development checkpoint reselection. Final training
admission must pass before any development scoring. Multi-seed development and
prospective evaluation remain necessary; this preflight is not a performance
result or runtime-profile acceptance.

The factory has no-optimization tests for full SB3 SAC save/load parity, replay
tag retention and role mismatch rejection. No worker or ordinary factory was
changed to start this experiment. Replay-update behavior, frozen-bundle metadata,
production manifest and centrally metered Colab worker integration remain pending.

## Historical source investigation

[Cached SEC candidate coverage](sec-candidate-coverage.json) joins each stock's
first training session to the retained original-candidate filing ledger by its
explicit `discovery_symbols`. Candidates are ordered by archive acceptance time;
only acceptance dates strictly earlier than the first training day count, avoiding
same-day availability assumptions.97/107 stocks have an earlier candidate filing.
The ten without one in this ledger are ABBV, DIS, EQIX, MLM, O, PLD, ROST, TMUS,
TTWO and WELL. This is a scope for further source review, not a declaration that
these issuers lacked public filings or were not stocks. Current CIK discovery,
corporate predecessors and historical ticker continuity remain unverified for
all107; no record is admitted merely because a10-K or10-Q exists. Source/audit/
usage-ledger hashes are retained and unchanged.

A primary-source lead for QQQ is the
[2008 PowerShares QQQ Trust registration](https://www.sec.gov/Archives/edgar/data/1067839/000120677408000229/powershares_485bpos.htm).
Its historical trust identity must be linked to the provider's canonical QQQ
series; the URL is a review lead, not a checksummed collected/admitted record.
Later prospectuses describing earlier history cannot automatically acquire the
earlier publication date. No provider call, raw archive download, historical
classification admission or Colab allocation was made during this investigation.

## Primary lead reviewed and integration checks

The January31,2008 QQQ registration above identifies the trust as an ETF and lists
the contemporary tickerQQQQ. This supplies historical fund-type evidence for that
trust, while the provider's canonical QQQ series still needs an explicit identity
link and durable source/acceptance receipt. No manifest record was admitted from
a search snippet or a publication-date assumption.

The environment/fitted-replay helpers now use the existing sampler, frozen
normalizer, evaluator, simulator and holding baseline. Synthetic checks verify
all roles sample identically under shared seeds and use the correct per-episode
tags, with no local optimization. Worker/controller/archive and actual Colab
acceptance remain pending. The source and account ledger were not changed.

## Actual cached primary-instance inspection

[Cached document evidence](cached-stock-evidence.json) now inspects85 retained
originals:33 exact expected-ticker/form/common-share hints,52 without an exact
eligible ticker hint and22 stocks without a cached earlier instance.175 protected
inputs remain unchanged; no network requests, model inputs, optimization or Colab.
GOOGL/LIN/MDLZ show GOOG/PX/KFT and need explicit alias/security review. No validity
interval or historical classification is automatically admitted. See the
[instrument evidence guide](../../docs/instrument-type-evidence.md) for the
extractor, conservative exclusions, reproduction and next source scope.
