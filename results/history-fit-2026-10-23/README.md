# Saved history-model common training-window replay

Status: actual replay completed; all four descriptive benchmark flags fail.
Latest outcome below; earlier text preserves preregistration history.
No fitting, optimizer updates, provider requests, Colab allocation or retired-test
access. The failed history-composition family remains closed.

[Numeric protocol](../../configs/experiments/history_fit_diagnostic.json), canonical
SHA-256 `e319c26f6cddecfaf3b2b3e847941c77c2bdf0e0ce8d7f8e466353f6b14a3143`.

Replay both preexisting selected/final seed11 full/recent policies from the
[accepted study](../training-history-2026-10-23/README.md). Same original22 stocks
and four ETFs, anchor2017-12-29, executions2018-01-02 through2020-12-02. Both
immutable source datasets must yield exactly identical bars/features/ATR/window
and frozen original preprocessing. Preserve train labels and masked21-observation
profile; declared unit-sign exposure, learned protections and default cost/20-day
holding simulator remain unchanged. All source/model/report hashes verified
against retained accepted inventory before replay; no new checkpoint selection.

Question: does the recent model's selected-to-final deterioration also appear on
eligible training-period rows? Compare group/median excess versus matched holding,
using−2pp descriptive flags only. Different market periods/regimes/durations
prohibit causal raw-return comparisons. Eligibility/year counts do not prove
that every replay bar was sampled. Neither flags nor a training-period gain can
override failed development gates, reopen the family or justify promotion.

39 focused fit/optimizer/history-contract checks pass. Paired-view mismatch,
normalization/schema/features/ATR/bar tampering, wrong windows/splits and
uncommitted implementation reject. Full optimizer tensor/moment fingerprint
shared in reporting/optimizer.py; previous recurrent-fit interface/regressions
remain supported. Source compiles. No learning or new usage. Project38.68329334,
last official account349.27 zero rate/assignments; all prior owned sessions stopped.

Commit protocol/implementation before running:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.runtime/matplotlib" .venv/bin/python \
 scripts/diagnose_history_fit.py --output results/history-fit-2026-10-23/summary.json \
 --artifacts artifacts/history-fit-2026-10-23
```

Original immutable full/recent datasets and models remain unchanged. Detailed
derived paths/charts stay outside Git, compact group summaries remain tracked.


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

[Strict summary and unchanged-source proof](summary.json), [group CSV](groups.csv),
[reporting-only short-side diagnostics](short-side.json). Full derived reports:
`artifacts/history-fit-2026-10-23/<arm>/<phase>/training.json` and `.html`.
Next publish allowlisted derived charts/compact results without duplicating source
models, then inspect these saved fitted/development differences before a distinct
preregistered hypothesis. No new training has been authorized by a diagnostic flag.


## Verified derived-chart publication

[Release](https://github.com/jason490/RL-trading/releases/tag/history-fit-2026-10-23)
contains7 server-size/SHA-256-verified assets. comparison.html shows group excess;
extract equity-reports.tar.gz and open index.html for104 derived policy/holding
plots. No model duplication or raw bars. Compact group/short-side evidence and
fixed protocol included. Only the duplicate publication archive removed
(3,373,270 bytes),19 retained hashes unchanged; all full derived paths retained.
[Server receipts](release-verification.json), [cleanup](cleanup.json). No additional
Colab usage; project38.68329334 unchanged and prior sessions stopped.
