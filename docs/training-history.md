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
