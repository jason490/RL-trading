# Accepted off-policy fitted-period diagnosis

All four fixed saved-policy replays completed without fitting, optimizer steps,
normalization changes, provider requests or Colab allocation. All four descriptive
fitted comparability flags fail. The failed performance family remains closed;
these results do not select a checkpoint, establish a cause or promote a model.

## Results

Mean net total returns across the same26 separate-symbol episodes, not annualized
or a combined portfolio. Fitted executions cover2018-01-02–2020-12-02, with the
2017-12-29 decision anchor. Development numbers refer to reused2021–2023 and
therefore have different regimes/durations. Compare each period to its own holding
baseline; raw cross-period returns do not identify a causal generalization effect.

| Saved state | Fitted stocks | Fitted ETFs | Fitted mean | Development mean |
| --- | ---: | ---: | ---: | ---: |
| GRU-SAC selected10k |44.51%|32.44%|42.66%|31.28%|
| GRU-SAC final100k |41.88%|61.28%|44.86%|12.92%|
| SAC selected10k |30.71%|16.91%|28.59%|32.31%|
| SAC final100k |62.78%|29.35%|57.64%|30.01%|
| Matched holding |58.15%|47.15%|56.45%|30.60%|

GRU-SAC final improves fitted ETF excess to+14.13pp, while fitted stock excess
remains−16.27pp and median stock excess−9.92pp. SAC final improves fitted stock
excess to+4.63pp, but ETFs remain−17.80pp and median stock excess−2.43pp. Each
still misses the fixed−2pp stock/ETF/median descriptive requirements. The selected
states fail them as well. No state achieves positive total policy return plus
positive short P&L on either of the two fitted-period negative-holding stocks;
that cohort is reporting-only, never an input or selection rule.

Selected-to-final fitted means rise for both packages while their development
means fall. That is consistent with a training/development tradeoff, not pure
deterioration on every period or proof of overfitting as the sole cause. The
group/median weaknesses also prevent claiming broad fitted-period success.
Eligibility does not establish that every replay bar was visited during learning.
Nothing here justifies extra width, steps, indicators or a rescue of the closed
family. An unchanged20-session holding limit and learned protections remain active.

## Acceptance and provenance

[summary.json](summary.json) records the fixed protocol, source receipt binding,
four unchanged-state fingerprints, group/symbol excess, cost/participation groups,
reporting-only decline diagnostics, matched holding identity and80 protected file
hashes. Source is committed `a415552`; protocol SHA256 is
`cfa2cac907c5292762b8a2b12dbdd97254d01b71ad988188049d216a76ff4fe2`.

Each state completed26 episodes/19,136 execution rows:104 episodes and76,544 rows
total. All source/model files, policy/target tensors, complete optimizer moments/
groups, raw entropy parameters, counters, copied GRU actor and original training
normalization remain unchanged. Matching holding paths are identical across all
four replays. Runtime guards reject model fitting, backward/autograd, Adam steps
and normalization fitting. Default evaluation still rejects training labels;
the dedicated diagnostic preserves explicit `train` scope.

The replay uses local CPU inference only; execution backend/version details are
retained in the summary. No new interpreter or dependencies were installed.
RecurrentPPO's identical final policy was not replayed; its existing fitted-period
failure remains historical evidence, not an independent new seed or confirmation.
See the [identity audit](../recurrent-offpolicy-2026-10-30/ppo-historical-identity.json).

Full paths and four interactive chart pages are retained under
`artifacts/recurrent-offpolicy-fit-2026-10-31/`, named
`gru_sac-selected`, `gru_sac-final`, `native_sac-selected`, `native_sac-final`
with `.json`/`.html` suffixes. Charts show equity/holding, drawdown, prices,
learned protection zones, fills and exposure for every symbol. Models remain in
their original accepted cell directories; no weight copies were created. Generated
paths stay outside Git; compact evidence and documentation are tracked.

## Research decision

Do not allocate more training for this failed family or treat fitted flags as
promotion gates. Before another learned hypothesis, characterize a small fixed
set of training-only feasible reference policies under the same four-action,
cost/protection/holding constraints. This can help separate benchmark friction
from learned participation choices without tuning against development results.
It needs its own committed bounded diagnostic; it does not permit changing the
simulation rules or choosing directions from future declining-stock labels.

Retired2024+ data stays closed. Project ledger remains **39.92585919/150**;
no Colab session was allocated and no new learning occurred. The broader research
goal remains active; no robust/prospective winner has been established.
