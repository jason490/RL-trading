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

## Separating a longer budget from a different run

The family report records `mean_selected_gain_after_100k_pp` and
`mean_final_gain_after_100k_pp` using the 100k milestones inside those same fresh
250k runs. Do not attribute an early best checkpoint to later training merely
because the run's requested budget is 250k. Previously published 100k cohorts can
differ through numerical/backend variability even with fixed seeds and versions;
those historical scores are context, not a substitute for within-run anchors.

## Completed combined-input family

All three combined-input 250k runs are retrieved, with source/dependency checks
matching the basic-input control. Selected mean validation return is **46.82%**
versus **23.96%** for basic inputs; selected mean drawdown improves 5.52 points.
Two of three selected seeds improve. Final checkpoints average **−1.73%**, versus
**14.46%**, and final drawdown worsens 3.80 points. The full paired feature gate
therefore fails despite stronger selected policies. These selected validation
results are optimistic and supply no new held-out performance claim.

Within these same fresh runs, selected returns improve 18.87 points after 100k,
while final return falls 8.64 points from its 100k anchors. Historical 100k cohorts
are separate runs and must not be treated as exact counterfactual trajectories.
Supply/demand-only and sector-only comparisons are still required for attribution.

## Interrupted attempt and recovery

Four fresh cells are complete, including supply/demand seed 11 (selected and final
validation return 13.42%). Supply/demand seed 23 lost its CLI execution connection
after the retrieved 200k checkpoint; bounded final-download retries found no
archive. Its CPU session was stopped and verified. No metering or budget stop
occurred; cumulative estimated project consumption was 34.537024 units. Preserve
that checkpoint and failure record, exclude its partial trajectory from complete
cell aggregates, and retry that seed once from the unchanged learning source.
Completed cells are skipped. See the tracked recovery record in the results.

## Completed supply/demand-only family

All three 250k seeds are retrieved and their runtimes stopped. Selected mean
validation return is **18.39%**, versus basic inputs' **23.96%**; final mean is
**11.83%**, versus **14.46%**. Only seed 23 improves selected return. Selected
mean drawdown improves 0.34 points and final drawdown improves 1.75 points, but
returns fail the paired feature gate. Seed 37 selects 17.19% at 160k and finishes
at 3.40%; all trajectories remain published rather than dropping weaker seeds.

Within these same runs, selected return gains 10.08 points after 100k and final
return gains 8.53 points from its 100k anchors. Longer training helped this family
relative to its own early checkpoints, while it still underperformed the matched
basic-input control. At that point the sector-only family was still needed before attributing
the combined family's selected gains or calculating feature interactions.

## Complete nine-cell outcome

Sector-only seeds select **46.35%, 29.25%, and 16.74%** and finish at
**6.14%, −29.09%, and −10.95%**. Their selected mean is **30.78%** and final mean
**−11.30%**. All three feature families fail the preregistered paired gate.
The complete comparison and all unsuccessful seeds are retained in the
[results](../results/indicator-followup-2026-10-16/README.md). The supply/demand
selected-return effect is −5.57 points without sectors and +16.04 with sectors;
its interaction is +21.61 points (seed standard deviation 38.81). These noisy
validation-selected contrasts do not establish independent predictive benefit.

All nine new and three control frozen bundles passed CPU prediction parity,
normalization immutability and schema rejection. All owned sessions are stopped.
Cumulative conservative usage is 35.098226 of 150 units (114.901774 remaining).
The user subsequently authorized a separate algorithm/memory stability comparison
and configurable concurrent runtimes under centralized shared-budget monitoring.
No new test audit or broker orders are part of either experiment.
