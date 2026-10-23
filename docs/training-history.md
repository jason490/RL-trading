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
