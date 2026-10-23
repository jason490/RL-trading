# Saved history-model common training-window replay

Status: fixed protocol and verified implementation prepared; actual replay pending.
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
