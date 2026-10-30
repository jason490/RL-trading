# Fitted-period policy diagnosis

Latest reuse audit (2026-10-30): the new100k RecurrentPPO reference exactly
matches the retained full-history final policy tensors and complete Adam state
from the2026-10-23 study, on the same immutable dataset/normalization. The
training-window helper and shared evaluator body remain unchanged. Its earlier
fitted-period failure is relevant historical evidence; no duplicate replay or
independent confirmation is claimed. The new post-update selected40k differs
from the old pre-update selected80k, so the old selected fitted result cannot
be substituted for it. No such equivalence is claimed for nativeSAC. See the
[bounded identity audit](../results/recurrent-offpolicy-2026-10-30/ppo-historical-identity.json).
This retains the closed family's status and never changes the active GRU run.

The fixed TD3 state screen failed alongside earlier PPO/SAC state experiments.
Before another model change, distinguish a failure to approach holding on fitted
history from a training/development contrast. Neither observation alone proves
underfitting or overfitting: market regimes, durations, initialization, source
revisions and optimization objectives differ.

[Protocol](../configs/experiments/training_fit_diagnostic.json) fixes final weights
from the existing 50k seed-11 full-state native PPO, SAC and TD3 runs. PPO used
wide initialization; the others standard. This is descriptive within-policy
analysis, not a controlled algorithm ranking. No best-checkpoint selection,
model fitting, optimizer call, normalization update, provider request or Colab
allocation is allowed.

## Window and accounting

Use only existing training bars/features in immutable dataset 672a2162….
December 29, 2017 is the decision anchor; executions run January 2, 2018 through
December 2, 2020. The latter is the actual existing training cutoff after split
purging; never reconstruct missing December bars. All 26 symbols must have the
same endpoints. Preserve aligned bars/features/ATR and explicit `train` labels.
The ordinary evaluator continues to reject `train`; a separately named training
diagnostic invokes the same simulator loop without mislabeling fitted data as
validation. Preprocessing remains the original training-fitted normalization,
not a new fit or an independent holdout.

Use unchanged costs, corporate actions, next-session fills, actions and holding
limits, with matched holding on each actual episode. Compare per-group excess
and median stock excess, rather than raw returns across different periods.
Training comparability flags require stock/ETF excess and median stock excess
at least -2pp. Flags diagnose only: they cannot override failed development
gates or authorize promotion. A training-pass/development-fail contrast is
consistent with a generalization gap, not proof of its cause. Failure on both
periods is weak benchmarked fitted-period performance, not a mathematical test
of model capacity or optimizer correctness.

## Reproducibility

Source states must be complete with verified owned cleanup. Latest weights,
metadata and saved final development paths are checksummed; existing development
accounting is validated against immutable bars. Policy tensors, update counts,
normalization and source files must remain unchanged. Full derived training
paths/charts remain under ignored artifacts; compact group/symbol results are
tracked. Retired test data stays unopened, and future declining-stock cohorts
remain reporting-only.

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
  .venv/bin/python scripts/diagnose_training_fit.py \
  --output artifacts/training-fit/summary.json \
  --artifacts artifacts/training-fit/paths
```

Use new paths; the command refuses to overwrite evidence. Current status:
protocol and explicit training-window implementation prepared; 20 slicing,
split-isolation, report-contract and direction-report checks pass without
fitting. Actual saved-policy replay is not complete yet. Project usage remains
36.149107/150; all owned Colab sessions are stopped.

## Actual inference-only outcome

The committed replay completed:26 episodes/19,136 execution rows per model.
SAC fitted stocks67.06% versus holding58.15%, ETFs53.86% versus47.15%, median
stock excess-0.46pp: training comparability passes. Its same final policy
development mean7.93% versus30.60% and group excess-23.94/-15.70pp fail. This
contrast is consistent with a generalization gap, not proof of a single cause.
PPO/TD3 fail the training flags; do not describe all learners as only overfit.

All13 source hashes, policy tensors including SAC alpha, optimizer update counts
and normalization are unchanged. No fitting, checkpoint selection, retired data
or Colab. Three published assets verify; duplicate release copies removed while
full derived paths remain. Project36.149107/150; all owned runtimes stopped.
[Actual evidence](../results/training-fit-2026-10-20/README.md).

Next audit existing expanded survivor data for a fixed SAC training-diversity
hypothesis, keeping original normalization and validation unchanged with a fresh
matched original-universe control. Failed prior PPO expansion remains negative
evidence. No allocation before coverage/source/sampling checks and a separately
committed bounded protocol. Preserve all absolute/multi-seed/prospective gates.

## Fitted decline/rebound decisions

A saved-path20-session/10% audit finds older50k SAC short requests40.63% before
continued declines versus31.44% before rebounds, conditional on accepted
protective decisions. Mean forward return after past declines is positive;
blind decline oversampling need not teach persistent shorting. Future buckets
are reporting-only and overlapping, not independent trades or profit evidence.
PPO/TD3 differ in source/initialization; no causal algorithm ranking.
[Results, caveats and hashes](../results/training-regimes-2026-10-20/README.md).
No fitting or new inference/Colab was needed. Next audit actual100k diversity
final policies on the same fixed fitted window before another hypothesis.

## New100k diversity-policy fitted replay

Both final policies pass descriptive fitted comparability: original91.72%,
expanded62.43%, holding56.45% on the identical2018–2020 common panel. Development
10.38%/24.19% versus30.60% remains failed. Weights/SAC alpha/update counts,
normalization and source hashes are unchanged. Decline/rebound reporting finds
expanded more long-biased even on fitted persistent declines. No optimizer/
generalization cause is proven and no training outcome selects a checkpoint.
[Exact results and retained path locations](../results/sac-diversity-fit-2026-10-20/README.md).
Next consider a separately fixed regularization comparison with fresh matched
expanded-data control; this is a proposal, not authorization to skip preflight.

## Fixed recurrent selected/final fitted-period comparison

[Protocol](../configs/experiments/recurrent_fit_diagnostic.json) fixes all four
cash-study arms and both already-selected/final weights. Replay the original26
symbols only on the established2017-12-29 decision anchor and2018-01-02 through
2020-12-02 executions. Additional85 training symbols are excluded from the
comparison, without changing the immutable expanded source or normalization.
All source training episodes still reject future/misaligned rows before filtering.
The default legacy training-window behavior remains strict; standard evaluation
still rejects train. An explicit recurrent diagnostic carries memory, resets per
symbol, uses masked observations and each accepted arm's execution mode.

The implementation requires committed protocol/source before actual replay. It
checks original metadata/weights against the retained publication inventory and
verifies unchanged policy tensors, full Adam state fingerprint, counters,
normalization and files after inference. Full derived paths stay under ignored
artifacts, compact group/per-symbol summaries in results. No fitting, optimizer
step, provider request, new checkpoint selection or Colab allocation.

Use group/median excess against matched holding; periods differ in regimes and
training windows may not have sampled every eligible bar. The−2pp descriptive
flags cannot establish capacity/overfit causes or override failed development
gates. The closed cash family stays closed. Commit verified implementation and
protocol before invoking:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" \
 .venv/bin/python scripts/diagnose_recurrent_fit.py \
 --output results/recurrent-fit-2026-10-22/summary.json \
 --artifacts artifacts/recurrent-fit-2026-10-22
```

Actual replay pending at this implementation milestone. Project38.309006
retained; all owned sessions stopped.

## Actual recurrent fitted-period replay

Eight saved selected/final models complete26 episodes/19,136 rows each, with
unchanged tensors, full Adam fingerprints, counters, normalization and source
hashes; all matched holding paths equal. No fitting/provider/Colab/test access.
All eight fitted stock/ETF/median flags fail. Fitted holding58.15% stocks,47.15%
ETFs. Posthoc final stocks27.22%, ETFs68.43%; cash final19.92%/27.07%. Cash
fitted mean improves3.22→21.02% while development29.24→6.63%; binary fitted
mean7.79→15.28% while development34.38→−1.25%. This is consistent with a
generalization tradeoff, not pure deterioration on every period; fitted benchmark
weakness remains and no cause/promotion proven.32 focused checks passed before
committed protocol6f2e10c and actual replay. All sessions stopped, project38.309006
retained. [Evidence](../results/recurrent-fit-2026-10-22/README.md).

Recurrent fitted-period charts published: [release](https://github.com/jason490/RL-trading/releases/tag/recurrent-fit-2026-10-22), six server sizes/digests verified,6,722,998 duplicate bytes removed and30 retained hashes unchanged. Original complete paths retained, no model duplication or new Colab charge. [Evidence](../results/recurrent-fit-2026-10-22/README.md).

## History-model paired training-window diagnosis

[Fixed protocol](../configs/experiments/history_fit_diagnostic.json) compares saved
full/recent selected/final seed11 models on the same eligible2018–2020 window.
`common_training_views` checks exact accounting bars/features/ATR, schema, names
and preprocessing across both immutable sources. Same original26 panel, unit-sign
execution,21 masked observations and simulator; no normalization or model fitting.
The complete optimizer fingerprint is shared in `reporting/optimizer.py`; the
existing recurrent-fit script import remains compatible.

39 focused fit/optimizer/history-contract checks pass. Actual replay remains
pending until committed code/protocol. Different market periods and missing
bar-level visitation proof limit interpretation; descriptive−2pp training flags
cannot select checkpoints or override failed development gates. Source hashes,
weights, optimizer moments and normalization must remain unchanged.
[Method and command](../results/history-fit-2026-10-23/README.md).


## Actual saved-policy replay completed

All four original26-symbol replays complete:19,136 execution rows per model,
76,544 total. Both immutable datasets produce identical common eligible
2018-01-02–2020-12-02 execution bars/features/ATR and original normalization,
anchored2017-12-29. All four descriptive training stock/ETF/median−2pp flags fail.

| Saved model | Training-window stocks | Training-window ETFs | Development mean |
| --- | ---: | ---: | ---: |
| Full selected |40.53%|50.22%|19.62%|
| Full final |33.32%|36.13%|18.48%|
| Recent selected |32.36%|28.54%|20.95%|
| Recent final |46.66%|31.18%|5.60%|
| Matched training-window holding |58.15%|47.15%|—|

Separate-symbol mean net total returns, not annualized or a combined portfolio.
Training and development have different dates/regimes/durations; raw cross-period
returns cannot identify a causal generalization effect. Recent training-window
mean improves31.77→44.28% while development20.95→5.60% declines, consistent with
a generalization tradeoff but not proof of its sole cause. Full training mean
42.02→33.76% and development19.62→18.48% both decline. Final recent median stock
excess−4.88pp still misses−2pp; all fitted stock means underperform holding.

This does not establish an accepted winner, justify more width/steps, reopen
history/capacity families or select another checkpoint. Actual visitation year
counts cannot prove every replay row was sampled. All phases are preexisting
saved weights and flags are reporting-only. No fitting, optimizer step,
normalization change, provider request, Colab allocation or retired-test access.
Complete weights/Adam moments and source hashes remain unchanged. Original
project ledger38.68329334/150 and last official account349.27 zero usage remain;
all prior owned sessions stopped.


[Evidence](../results/history-fit-2026-10-23/README.md).


History-model fitted-period charts published:7 server assets verified,104 equity
plots on four pages, no model duplication or raw bars. Duplicate chart archive
removed (3,373,270 bytes),19 retained hashes unchanged; full paths retained. No
additional Colab units; project38.68329334 unchanged.
[Release](https://github.com/jason490/RL-trading/releases/tag/history-fit-2026-10-23).


[Saved path behavior comparison](../results/history-fit-actions-2026-10-23/README.md)
uses explicit `period='training-window'` in reporting helpers. Default scope
continues to require2021–2023 validation; explicit training allows only the fixed
2018–2020 diagnostic range, with original reconciliation unchanged. No test or
arbitrary period scope. Source/model/report hash checks precede comparison.
