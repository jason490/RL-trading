# Training-history composition research

Different training years change the market conditions, observation diversity and
repeat exposure encountered by a model. A newer window may adapt better or lose
useful history. Only a matched experiment can establish its effect on development
performance; repeated development results still need prospective confirmation.

The [date-only coverage audit](../results/training-history-2026-10-23/README.md)
checks full cached history versus a fixed2016-onward subset. Both support all111
symbols and complete512-session transition windows. Expected2020 execution mass
is about2.4% under full history and7.3% under the subset. Uniform window starts
underweight edge years; these are expectations, not actual visitation counts.

`reporting/history_coverage.py` owns exact window/date expectation. The thin
`scripts/audit_training_history.py` reads only training session dates and records
input hashes. It neither generates datasets nor fits models. Use the result
README for reproducible commands, assumptions and pre-allocation requirements.

Next instrument actual decision/execution year counts, then freeze one matched
history-composition protocol. Keep evaluation dates, model, reward, execution,
normalization and budget fixed; preserve train-only cutoffs and retired-test
isolation. Do not interpret this coverage audit as evidence to promote a model
or reopen the failed capacity experiment.

## Measured year visitation

Sampled training environments now include `year_counts` version1 in
`sampling_statistics()`. Existing runner checkpoint callbacks write it into
`training-sampling.json`; final metadata includes the same measured snapshot.
It records start years, decision years, next-bar execution years, decision→
execution year pairs, and per-symbol starts/transitions/year counts. A Dec31
observation followed by a Jan execution belongs to different decision/execution
years. Only actually stepped transitions count, including partial final windows.

Counters reset with the existing sampling-statistics reset, not each episode.
Resetting midway through an episode can legitimately produce transitions without
an episode start in the new measurement interval. Snapshots are copied and cannot
mutate counters. Counter reads consume no RNG or prices and never affect actions,
sampling, normalization or rewards. Legacy unsampled environments still return
no sampling diagnostics. All sampler modes use the instrumentation.

`environment/sampling/years.py` validates global, per-symbol and year-pair
margins, nonnegative integer counts and training-only year keys. Expanded-panel
validation cross-checks against independently recorded symbol/group counts.
Old artifacts lacking the optional field remain valid historical evidence; they
cannot establish actual year visitation. New history-composition plans must
explicitly require this field rather than silently accepting missing logs.

## Immutable recent suffix and fixed comparison prepared

The [actual subset preflight](../results/training-history-2026-10-23/preflight.json)
passes:111 symbols,137,529 recent training rows,19,058 exact unchanged development
rows. Recent dataset `5fb2be66917f4498f0589f063ee60640e1d636f9cd4204e347aa033b3229f732`
retains source bars/features/ATR suffixes starting2016-01-01. Original full-history
normalization stays fixed across arms; this isolates sampled-row composition,
not recency of preprocessing. Causal indicator context may predate2016. No feature
recomputation, provider request, test reconstruction or source mutation.

[Numeric protocol](../configs/experiments/training_history.json) freezes full/recent
same64-unit RecurrentPPO (61,513 parameters),21 observations, original seven market
features,100k seed11,10k evaluation, same costs/actions/reward/holding rules and
common pinned untrained tensors/RNG. Both selected and final must improve2pp,
final stock/ETF cannot worsen, drawdown worsening at most2pp. Both phases must
reach stock/ETF and median-stock excess at least−2pp versus holding. Failure
closes this fixed family with no start-year/window/step/model/reward grid.
Only passing allows paired11/23/37 confirmation then prospective evaluation.

Shared2-unit incremental allowance starts at project38.60166524433091; project
150/130 and account150/170 safeguards persist. Two512 full64 CPU smokes, actual
checkers,156 frozen checks, optimizer/initialization/sampling-year proof and a
smoke-bound runtime projection must pass before screen allocation. Original
central fleet/ledger/deadline/cleanup/recovery code is reused. No allocation yet:
dedicated plan/worker/runner/retrieval/report contracts and acceptance remain.

```bash
PYTHONPATH=src .venv/bin/python scripts/preflight_training_history.py \
 --output .runtime/training-history-preflight.json
PYTHONPATH=src .venv/bin/python -m pytest -q \
 tests/test_training_history.py tests/test_history_coverage.py
```

23 subset/coverage checks pass: exact suffix, copied validation/normalization,
source immutability, tamper rejection, short/misaligned/duplicate/future/test
sources and exact expected year mass. Actual immutable dataset save/read and
strict source/subset revalidation pass. No local training.

Frozen protocol canonical SHA-256:
`fefff889ad189136e55b2689fd0532f50e8a636f7cdf96b94e4008449e653a49`.
Changes to this numeric protocol after commitment require an explicit versioned
new study; do not silently alter gates in response to results.

## Version28 Colab integration and construction acceptance

`recency_plan`/`recency_study` distinguish this comparison from the existing
version15 historical augmentation experiment. Old `history_plan` and its scripts
retain their interfaces. Common fleet, bootstrap, metering, account-floor,
checkpoint integrity and retrieval logic remain shared.

`training/training_history.py` binds exact protocol/dataset identities, per-arm
payloads, common64-unit initial tensors/RNG and restored optimizer records.
`runtime/colab/recency_artifacts.py` requires actual measured year evidence and
independent symbol/year margins, exact input snapshot, checkpoint, execution,
normalization and frozen metadata. Missing counters reject new-study retrieval.
Source-export package includes `recency-initialization.npz` without raw bars.

The actual two immutable smoke payloads validate before allocation. Both source
environments pass Gymnasium/SB3 checkers; untrained recurrent predictions and
unchanged model/optimizer records pass. See [construction evidence](../results/training-history-2026-10-23/construction.json).
111 focused contracts/regressions pass, including wrong dataset/model/optimizer/
snapshot/missing-year/future-year rejection, runtime-profile stale/failed/overdue
blocking, report arm/dataset routing, confirmation gating, prior historical/
capacity/guard contracts and account floor. Compilation passes; no local fitting.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python \
 scripts/build_recency_jobs.py --smoke --output .runtime/recency-smoke-jobs.json
```

Commit/freeze source and verify official usage/default Python before running
those tasks with the existing central fleet. Retain study baseline38.60166524433091
and shared2-unit allowance across all phases/attempts. Full64 architecture in
both512 smokes; no smoke-return selection. After exact frozen checks, model/
optimizer restoration, measured-year acceptance and verified owned cleanup:

```bash
PYTHONPATH=src .venv/bin/python scripts/accept_recency_smokes.py \
 --state .runtime/recency-smoke-state.json \
 --restoration .runtime/recency-smoke-restoration.json \
 --source-proof .runtime/recency-source-v1/export-proof.json \
 --output .runtime/recency-smoke-acceptance.json
PYTHONPATH=src .venv/bin/python scripts/build_recency_jobs.py \
 --output .runtime/recency-screen-jobs.json
```

The accepted file's exact raw bytes bind the generated runtime profile. Controller
validation checks it against actual acceptance before allocation. Preserve that
file unchanged; screen report requires the same profile via `--runtime-profile`.
Long/confirmation tasks reject absent approval or failing screen evidence.
Same per-symbol draw counts are required; year distributions/offsets intentionally
differ. Initial-manifest hash in this protocol uses canonical JSON digest;
reused capacity initialization loader independently verifies its original raw
manifest hash and snapshot bytes. No new untrained files or trained weights reused.

Actual two512 Colab smokes remain before100k; no allocation or new usage yet.
Project38.60166524/150, last official account349.34, zero assignments/rate.

## Actual Colab smoke acceptance and runtime approval

Two512 full/recent CPU smokes from2dfa1fb/defaultPython3.13.16 complete/retrieved
and verified stopped.156 frozen sequence/repeat/reset predictions, normalization/
input rejection and actual model/optimizer/checkpoint restoration pass. Same
61,513-parameter architecture and pinned tensors/RNG;80 actual Adam updates and
40 nominal epochs. Actual symbol draw counts match; measured year logs reconcile
to512 each, with intentionally different year distributions. No smoke-return
selection. Durations138.8675/147.3082s yield conservative3504.0053s/58.4min
projection including1200s reserve, below7200s. Exact acceptance bytes bind100k
plans and controller pre-allocation checks. Same-source seed11 full/recent100k
screen next; no performance claim. Project38.62221390, study increment0.02054866,
official account349.32 zero rate/assignments. Original shared2-unit baseline and
project/account limits unchanged. [Evidence](../results/training-history-2026-10-23/README.md).


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

[Evidence](../results/training-history-2026-10-23/README.md).
