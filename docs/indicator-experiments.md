# Matched 250k indicator follow-up

## Question and scope

The prior learning study trained basic inputs to 250k but combined inputs only
to 100k. Its results cannot determine whether added features need more training.
This separately authorized follow-up keeps the exhausted study and its gate
unchanged. The committed [protocol](../configs/experiments/indicator_followup.json)
declares nine fresh CPU SAC runs, seeds 11/23/37 for each input family:

1. Combined daily/weekly supply/demand zones and sector context, 250k steps.
2. Supply/demand zones without sector context, 250k steps.
3. Sector context without supply/demand zones, 250k steps.

The completed original-reward, basic-input 250k family is the control. It need
not be retrained: clean Git source proofs must establish matching learning code,
dependencies, simulation assumptions and budgets before comparisons are accepted.
The unchanged four-component actions, costs, reward, 26 tradables, dates and
training-only normalizers keep this an input-family comparison. Kernel Python
is checked before research dependency installation; use Colab's own interpreter.

## What counts as useful

Report all seeds, selected and final returns, costs, turnover, maximum drawdown
and behavior. For each candidate versus the basic-input control, require at
least two points of mean selected-return improvement, positive paired gains in
at least two of three seeds, no mean final-return decline, and no more than
two points of mean drawdown worsening for either selected or final policies.
This feature gate is declared before runs; it is separate from the old within-run
50k learning gate, which remains descriptive context. No validation result
establishes out-of-sample trading profitability.

With all four families, report supply/demand's effect with and without sector
context, sector context's effect with and without zones, and their interaction.
Three seeds provide descriptive variability, not strong significance evidence.
Never attribute a combined-input difference solely to supply/demand.

## Logging, budget and next work

Append compact validation JSONL every 10k transitions. Preserve CSV/standalone
HTML graphs, post-update 50k/100k/250k weights, selected/latest weights, frozen
bundles and final replay. Keep failed cells and provenance. Publish derived
reports and verified model assets; raw bars and private runtime logs stay local.
Cleanup follows successful retrieval and release-checksum verification.

Start from the existing 34.032604-unit project ledger. Use one CPU runtime at a
time, minute metering, the unchanged 150-unit ceiling/130-unit working stop and
a two-hour allocation deadline including setup/download/shutdown. The longer
execution window reserves ten minutes for retrieval and termination. No automatic
restart follows budget stop. Do not reinstall the existing official CLI.

The 2024–September 2026 audit is retired and never evaluates or selects these
runs. No broker orders are authorized. After input attribution, audit historical
universe coverage and plan a separate churn experiment. Representative declining
and delisted stocks require contemporaneous membership, correct terminal returns
and verified provider coverage; do not pick stocks using future recovery labels.

## Rebuilding the comparison report

```bash
PYTHONPATH=src .venv/bin/python scripts/report_indicators.py \
  BASELINE_ARTIFACT_ROOT_11 BASELINE_ARTIFACT_ROOT_23 BASELINE_ARTIFACT_ROOT_37 \
  NEW_INDICATOR_ARTIFACT_ROOTS \
  --output artifacts/public/indicator-followup-2026-10-16/comparison.html
```

The reporter accepts only complete CPU SAC 250k original-reward cells and rejects
duplicate seeds, changed simulator/algorithm/dependency assumptions, unmatched
symbol/benchmark coverage and missing post-update endpoints. Clean Git learning
source proofs are mandatory. A family needs all three seeds; interaction
attribution waits for all four families. JSON records the exact gate components,
paired gains and input-family contrasts. CSV summarizes complete families; the
standalone graph compares selected/final returns, training curves and drawdown.
The baseline control is retained in reports but does not count as a new remote run.

## Training-regime coverage before adding stocks

`scripts/audit_training_regimes.py DATASET --output results/STUDY/training-regimes.json`
audits only training episodes through 2020. At each eligible session it computes
trailing 252- and 63-session split-adjusted price changes; the first 252 sessions
in each episode are warmup. A decline bucket means the trailing-year price change
is below −20%; a two-window decline additionally requires a negative trailing
quarter. No future recovery, validation or test outcome labels these observations.
Dividends are excluded: this describes price-trend coverage, not strategy returns.
Overlapping windows are descriptive and are not independent market events.

The existing training set already contains two-window declines in 16 of its 26
tradables. Adding losers is therefore not equivalent to introducing the first
negative examples. Nevertheless, the universe is a retrospectively selected
surviving-company sample; this audit does not verify historical membership or
delisting returns. Those data are required before claiming a representative
historical universe. The audit changes neither data nor episode sampling and
leaves the matched indicator experiment intact.

Incomplete families expose individual completed-seed curves as dotted lines,
without a family aggregate or feature gate. Their records remain in JSON so
single-seed gains/regressions are visible while other seeds run. Complete-family
bar summaries and factorial contrasts retain their three-seed requirements.
