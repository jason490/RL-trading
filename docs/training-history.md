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
