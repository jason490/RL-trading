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
