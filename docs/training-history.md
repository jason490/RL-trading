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
