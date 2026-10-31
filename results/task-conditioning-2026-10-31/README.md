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

## Actual bounded original-cover collection and review

The first three unresolved cached cases, ABT/ADBE/AEP, completed their six fixed
index/primary acquisition operations. [Receipts](original-cover-collection.json)
bind7,298,946 retained raw bytes to official endpoints; no retry or Colab.
[Verified review](original-cover-review.json) confirms common-equity statements
for all three and an explicit ADBE trading-symbol statement. ABT/AEP ticker links
remain partial. Original index filing/acceptance metadata support conservative
availability on2010-02-23/2010-01-26/2010-03-02 respectively, all before training.
This is the declared two-XNYS-session assumption, not exact dissemination proof.
All protected original source, ledger and code hashes remain unchanged. No
historical interval/classification admission or model input was enabled.

## Incorporated exhibit and fund source results

[AEP exhibit collection](aep-exhibit-collection.json) retrieved the explicitly
indexed EX-13 HTML in one request (17,995,757 bytes); its
[review](aep-exhibit-review.json) supports a common-stock NYSE market reference,
without upgrading the abbreviation to a separately verified ticker/history link.
ABT has no EX-13 annual-report attachment in the retained index.

[Four-fund collection](fund-source-collection.json) completed eight fixed
index/primary operations and retained3,598,117 bytes. [Review](fund-source-review.json)
verifies dated fund-type observations for SPY/QQQ/DIA/IWM, with conservative
availability2010-01-29/2010-01-28/2010-03-02/2009-11-02. QQQ's source ticker is
QQQQ. IWM's source is a bank424B2, so source registrant19617 is not an IWM issuer
mapping. All original/master/code/ledger hashes remain unchanged. No historical
interval admission, fitting, new Colab credits or retired-test access.

## Master-source sequence stopped with evidence retained

The deterministic master selector finds105/107 earlier financial-filing candidates:
36 already have relevant document observations,69 need selected primary HTML,
and ABBV/DIS need the separately located event/predecessor candidates.
[Source plan](stock-master-selection.json) remains unchanged.

Batches0–3 completed32 primaries; batch4 retained MLM/MRK/MSFT then stopped on
NEE's cross-directory joint-filer link before its primary request. Batches5–8
never started. Original global stop and per-batch claims remain; no retry or
Colab allocation. Attempts retained90,366,834 bytes; complete32-document
[hints](stock-master-complete-batch-hints.json) contain28 cover/13 ticker hints,
and the stopped batch has separately labeled [partial hints](stock-master-batch-4-retained-hints.json).

[NEE source inspection](nee-joint-filer-inspection.json) confirms official master
records for both FPL Group753308 and Florida Power & Light37634 under the same
accession; the original index points its primary to37634. Strict endpoint binding
rejected this legitimate joint-filer case. It remains a recorded failed attempt,
pending an explicit verified joint-filer resolution and reviewed resume, not a
license for arbitrary cross-CIK links. All declared protected files/ledger remain
unchanged; project usage39.92585919/150. Model results and metadata admission are
unchanged.
