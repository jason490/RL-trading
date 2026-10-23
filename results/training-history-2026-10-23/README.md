# Training-history availability and window-mass audit

Date-only audit of the immutable expanded111 Yahoo training panel. No market
outcomes, model inference/fitting, preprocessing changes, provider calls or Colab
allocation. Existing2021–2023 development values are unread; retired2024+ absent.
Dataset file hash and unchanged-source proof are in [coverage.json](coverage.json).

## Findings

| Candidate source | Stocks / ETFs | Minimum bars per symbol | Expected2020 execution share |
| --- | ---: | ---: | ---: |
| Existing full history |107 /4|1,946 stocks;2,700 ETFs|2.45% stocks;2.43% ETFs|
| Fixed2016 onward subset |107 /4|1,239 each|7.32% each|

Every symbol supports a complete512-transition window in both sources. Full
source spans2010–2020, with different listing/readiness starts. The subset uses
2016-01-01 onward and the existing purged2020-12-02 endpoint; no new bars or
labels. This is a coverage candidate, not an approved training dataset.

A uniformly drawn start does not imply uniformly visited dates. Middle dates
belong to many possible windows; dates near the history ends belong to few.
Expected shares are exact enumeration-equivalent probabilities for a complete
window, assigning each transition to its next-bar execution date. The first bar
is observation-only; the last bar can execute a transition. This follows the
actual sampler's exclusive upper bound `len(bars)-512`. Group means assume
uniform episode frequency within each shuffled stock/ETF deck, not uniform
weight across all111 symbols. Current group probability remains22/26 stocks,
4/26 ETFs. Partial final windows and finite realized samples can differ.

Existing training logs contain symbol/group starts and transitions, not actual
year counts. These expected masses cannot reconstruct what an individual run
visited. No claim that poor2020 visitation caused deterioration or that a more
recent window will improve returns. A recent subset also discards older market
conditions, reduces unique observations and increases repeat exposure.

## Next experiment prerequisites

Before allocating, add measured decision/execution year counts to training logs
with sum/identity checks. Freeze a single full-history versus2016-onward training
comparison, unchanged64-unit architecture/actions/costs/100k budget, common
untrained starts and original26-symbol development dates. Verify immutable
subset construction, original training-only normalization held constant to
isolate composition, feature warmup inherited causally, train-only cutoff and
exact unchanged development episodes. Set numeric selected/final relative,
stock/ETF/median/risk gates and shared allowance before training. Failure closes
that fixed comparison; passing alone only permits multi-seed and prospective
confirmation. Do not choose dates by observed profits or vary years/model/reward
simultaneously. Capacity family stays closed.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_training_history.py \
 --dataset data/datasets/cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b/dataset.json \
 --output .runtime/training-history-coverage.json
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_history_coverage.py
```

11 checks pass: exact brute-force window expectation, boundary/off-by-one cases,
invalid/nontraining inputs, outcome-free sources, unchanged input and unread
development episode. Project38.60166524/150 unchanged; last official account
349.34, zero rate/assignments. No sessions allocated; all prior owned sessions stopped.

## Actual subset and numeric protocol prepared

[Preflight](preflight.json) passes: recent dataset
`5fb2be66917f4498f0589f063ee60640e1d636f9cd4204e347aa033b3229f732`,111 training
symbols,137,529 training rows,19,058 exact common development rows. All retained
source bars/features/ATR exact and original normalization unchanged. Source
hash unchanged, no retired-test/provider/training/Colab access.

[Fixed protocol](../../configs/experiments/training_history.json) and
[category guide](../../docs/training-history.md) pin same64 architecture/common
untrained starts and unchanged100k/10k settings,2-unit allowance from38.60166524,
selected+final2pp relative improvement and stock/ETF/median−2pp absolute gates.
Measured decision/execution year counters implemented and required for acceptance.
23 subset/coverage checks pass. Dedicated Colab integration/construction/smokes
remain before allocation. This is prepared methodology, not performance evidence.

## Integrated construction acceptance

Version28 plans/worker/runner/retrieval/frozen-package/report acceptance integrated
under distinct `recency` runtime names, preserving older historical interfaces.
111 focused tests pass, including actual tensor/RNG construction, strict artifact
and year-count rejection, runtime-profile binding, report routing and historical/
capacity/guard/account-floor regressions. Source compiles. Actual two immutable
smoke tasks validate and both environments/checkers/untrained recurrent forwards
pass. [Construction](construction.json); [commands and contracts](../../docs/training-history.md).
No local learning, new Colab allocation or performance evidence. Commit/freeze,
usage/default-Python check and two512 CPU smokes next;100k plans remain blocked
until actual accepted smoke bytes and runtime approval are available.

## Actual full/recent Colab smokes accepted

Two512-transition CPU smokes from clean2dfa1fbd45d6a3c120fb11311c9d285781556b57
(defaultPython3.13.16) completed and retrieved; both owned sessions verified
stopped.156 exact frozen recurrent action/hidden/cell/repeat/reset predictions
pass, with immutable normalization and schema/feed/nonfinite rejection. Actual
selected/final model/optimizer records and checkpoint identities restore; both
full64 architectures have61,513 parameters,80 Adam updates/40 nominal epochs.
Identical untrained tensors/CPU RNG and actual symbol draw counts pass. Required
decision/execution year logs reconcile to512 transitions per arm; year masses
intentionally differ. No smoke returns used to select settings.

Measured train/evaluate durations138.8675s full,147.3082s recent. Conservative
runtime projection3504.0053s (58.4min), including1200s setup/retrieval/shutdown
reserve, passes7200s. This is a mixed-duration heuristic, not a speed guarantee.
Exact accepted smoke bytes bind the prepared two100k seed11 plans. Same frozen
worker source and original protocol/data/reward/model/steps/gates retained.

Project38.62221390/150; study conservative increment0.02054866 (observed account
0.02), including setup/idle/transfers/shutdown. Official account349.32, zero
rate/assignments. Original baseline38.60166524433091/shared2-unit allowance,
project150/130 and account150/170 safeguards preserved. No orders/test access.

[Acceptance](smoke-summary.json), [frozen parity](smoke-restoration.json),
[compute and owned cleanup](smoke-compute.json), [runtime approval](runtime-profile.json).
Next same-source full/recent100k screen; no performance result established.


## Completed history-composition screen — 2026-10-23

Status: fixed seed11 comparison failed; family closed. No confirmation,
extension, start-year/window grid, promotion or performance goal achieved.
Both fresh100k runs completed in Google Colab CPU/defaultPython3.13.16 from
frozen clean2dfa1fbd45d6a3c120fb11311c9d285781556b57. Earlier sections retain
preparation/smoke history; this is the latest outcome.

| Arm | Selected return | Selected step | Final return |
| --- | ---: | ---: | ---: |
| Full2010–2020 |19.6245%|80,000|18.4844%|
| Recent2016–2020 |20.9523%|50,000|5.5976%|
| Matched holding |30.6019%|—|30.6019%|

Mean net total returns across separate26-symbol episodes on repeatedly reused
2021–2023 development history; neither annualized nor a combined portfolio or
independent test. Candidate selected improvement1.3278pp misses the2pp gate;
final change−12.8868pp fails. Candidate final stock/ETF4.4602/11.8533% versus
holding32.6265/19.4660%. Its final stock change−17.3073pp versus control fails
the nonworsening group gate; ETF improves11.4258pp. Mean drawdown worsening
0.0127pp is within risk tolerance but cannot compensate for failed return gates.
Both phases fail stock/ETF/median−2pp absolute targets. Selected stock/ETF excess
−10.3746/−5.6617pp, median−8.5500pp; final−28.1664/−7.6127pp, median−15.6457pp.
Reporting-only declining-stock joint positive total AND net short success1/3
selected→0/3 final; no robust profitable-short capability established.

## Actual training years and verification

Measured2020 execution share rises as intended: full stocks1.978%/ETFs3.005%,
recent stocks6.637%/ETFs8.263%. These realized finite samples differ from exact
complete-window expectations. Per-symbol/global/decision/execution/pair margins
reconcile; actual symbol draw counts match. Recent history changed composition
without providing stable development improvement. It does not prove all recency
methods fail, or establish a sole cause of deterioration. This fixed family is
closed; a future study needs a distinct preregistered hypothesis, not a year grid.

Full control reproduces the prior capacity control's exact selected/final tensor
hashes and symbol sampling, supporting instrumentation invariance. It was trained
fresh, not resumed or selected from old weights. Both arms restore identical
pinned untrained tensors/RNG. Strict source/data/accounting/holding checks and
actual selected/final models/Adam records restore. Final20,000 actual Adam steps,
1000 nominal epochs,61,513 parameters.156 screen frozen recurrent action/state/
repeat/reset checks pass;312 including smokes. Normalization remains immutable;
schema/feed/nonfinite rejection passes. Retired2024+ unopened; no local fitting,
new provider call or orders in reporting.

## Compute, artifacts and next work

All four smoke/screen sessions verified stopped. Official account349.27, zero
rate/assignments. Original project ledger38.68329334/150, remaining111.31670666;
study conservative increment0.08162810 versus observed account debit0.07,
including setup/idle/transfers/shutdown. Original baseline38.60166524433091/shared
2-unit allowance and project150/130/account150/170 safeguards retained.

Strict summary, compact10k learning curves with incremental changes, measured
year CSV, restored models, checkpoints, complete reports/source/datasets/ledger
retained. Interactive comparison is `artifacts/training-history/summary.html`.
Verified GitHub publication and duplicate-only cleanup follow this outcome commit.
Next diagnose accepted saved paths before choosing a distinct experiment; no
automatic training restart or retired-test evaluation.

[Strict report](summary.json), [curve](learning-curves.csv), [actual years](measured-years.csv), [frozen checks](restoration.json), [control reproduction](control-reproduction.json), [compute and cleanup](compute.json).


## Verified publication and duplicate cleanup

[GitHub release](https://github.com/jason490/RL-trading/releases/tag/training-history-2026-10-23)
contains17 server-size/SHA-256-verified assets, including four accepted complete
frozen bundles, selected/latest weights, interactive comparison and individual
selected equity reports, compact curves, measured-year CSV and control-reproduction
proof. Download comparison.html or extract validation-reports.tar.gz for charts.
Raw provider bars, credentials and private runtime files excluded.

Only two verified duplicate publication archives removed (35,931,249 bytes);
148 retained hashes unchanged. Original transports, models/checkpoints/reports,
source archive, both immutable datasets, ledger and plan.html preserved.
[Receipts](release-verification.json), [cleanup](cleanup.json). All four sessions
remain verified stopped; project38.68329334 unchanged. Fixed family closed.

Strict summary reproduces byte for byte from retained transports, with both actual
selected/final models and Adam states reloaded. [Reproduction](reproduction.json).
