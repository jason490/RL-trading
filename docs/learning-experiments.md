# Learning diagnostics and sequential experiments

The initial 36-run study and its test audit are preserved. New work uses only
training data and 2021–2023 validation. The inspected 2024–September 2026 audit
cannot serve as a new untouched test; later claims require a fresh locked audit.
No broker orders are authorized.

## Diagnose before modifying rewards

`evaluation/behavior.py` summarizes existing simulation rows/events into cost
components, fill reasons, full closures, long/short/flat exposure, risk clipping,
rejections and protective exits followed by an entry within one session. Partial
exits remain fills, not complete trades. `scripts/diagnose_behavior.py` reads
selected-validation files and records selected-checkpoint SAC entropy coefficients.
It never opens test reports. The derived results are under
`results/learning-study-2026-10-15/`.

Compare against cash, buy-and-hold and a constant full-long policy using the same
simulator, widest permitted stop/zone and 20-session holding limit. This separates
some management constraints from learned positioning; it is not an oracle or a
recommended trading strategy. Fees and impact are already included in equity.
An extra turnover penalty changes the objective and needs its own experiment.

Initial diagnostics show substantial fees/adverse-fill costs and frequent
protective exit/re-entry. Selected SAC entropy coefficients are already small;
these observations do not establish exploration as the dominant cause. Keep the
net-equity reward unchanged for the first controlled CPU comparisons.

## Ordered research gates

Keep the current 26 tradables, chronological splits, feature schemas, simulator
costs and seeds 11/23/37 fixed. First compare baseline SAC and combined-input SAC
on CPU, with checkpoint curves at 10,000-transition intervals and explicit
post-update milestones at 50,000 and 100,000 transitions. Train fresh comparisons;
do not mix old-reward replay into a changed-reward experiment.

The preregistered protocol is [learning_study.json](../configs/experiments/learning_study.json).
Improvement means at least a two-percentage-point increase in mean validation
return at the later post-update milestone, positive paired gains in at least two
of three seeds, and average maximum drawdown worsening by at most two percentage
points. Selected-checkpoint mean return must also gain two points over its matched
50k value and exceed the historical 19.79% family mean by two points. Also report selected-checkpoint returns and buy-and-hold excess; passing
this learning gate does not establish a profitable trading strategy.

If the first stage does not pass, test reward scaling independently, then longer
training up to 250,000 transitions, and a separately declared turnover penalty if
needed. Do not add stocks while diagnosing the current universe. Stop on a
budget/deadline/metering failure, a satisfied declared learning gate, or exhaustion
of the bounded experiment protocol. Improvement is not guaranteed; retain weak
runs and report a plateau rather than running indefinitely or changing the goal.

The cumulative 150-unit ceiling and 130-unit working stop remain unchanged.
Prefer one seed per CPU allocation for longer runs so setup, retrieval and
shutdown fit the two-hour allocation deadline. Check the actual Colab kernel
Python before installing research dependencies, and reuse the official CLI.

## Compact percentage-gain records

Each validation checkpoint appends one durable JSONL row to
`learning-curve.jsonl`. It contains aggregate and per-symbol metrics, elapsed
time, selected training-loss/entropy diagnostics, and no daily price/trade trace.
`learning-curve.csv` and `learning-curve.html` expose the latest evaluation at each
step. A post-update evaluation supersedes an on-step evaluation at the same
step in the graph; both remain in the append-only journal for provenance.

- `return_pct`: total net validation return, multiplied by 100.
- `change_pp`: percentage-point difference from the preceding distinct checkpoint.
- `equity_change_pct_from_previous`: relative change between the two terminal-equity multiples.
- `gain_since_50k_pp`: percentage-point difference from the 50k post-update milestone.
- `best_return_pct`: best observed validation return; distinct from the current value.
- `excess_return_pp`: current return less the matching buy-and-hold return.

Compact mode overwrites the rolling checkpoint/replay instead of keeping weights
at every 10k interval. Selected weights, latest weights and explicit 50k/100k/250k
post-update milestone weights remain. Detailed daily reports are retained only
for the selected and final models. Reward scaling affects the learning signal;
evaluation and reported returns always use actual net equity after modeled costs.

```bash
PYTHONPATH=src .venv/bin/python -m rl_stocks.runtime.colab.runner \
  data/datasets/DATASET_HASH --learning-study --learning-seed 11 \
  --learning-steps 100000 --reward-scale 1 --accelerator CPU \
  --cli-python /path/to/CLI_PYTHON
```

The learning plan admits only seeds 11/23/37, budgets 50k/100k/250k and separately
declared scales 1/10/100. Remote acceptance verifies the exact reward config,
all expected curve intervals, post-update milestones and matching CSV values,
in addition to model/frozen checksums and the original budget/cleanup rules.

## Multi-seed comparison and stopping decision

`scripts/report_learning.py ARTIFACT_ROOT... --output results/STUDY/curves.html`
produces an HTML graph, aggregate CSV and compact JSON. Individual seed curves
are available in the graph legend. Family aggregates require all three seeds;
incomplete cells cannot produce a passing stage. Same-step post-update records
replace on-step points only in the plotted view, preserving the full journal.
The gate also requires positive selected-checkpoint gains in two of three seeds.
A single exceptional seed, monotonic best-so-far curve, or drawdown deterioration
cannot satisfy the declared gate by itself. The JSON records every gate component
and remains explicitly validation-only.

## Completed unchanged-reward CPU comparison (2026-10-15)

All six 100k cells completed with compact curves, artifact acceptance and verified
runtime shutdown. Baseline SAC's mean final validation return rose 12.79 points
from its matched 50k checkpoints to 16.50%; selected checkpoints averaged 19.70%.
It failed the preregistered gate: drawdown worsened 2.84 points and selected return
did not exceed the historical floor. Combined inputs selected 11.78% on average;
none of its seeds improved its best checkpoint after 50k. More transitions alone
did not pass the declared improvement gate. The next stage tests reward scale 10
on the higher-scoring baseline family, using fresh seeds and real-equity reporting.
The complete checkpoint CSVs and gate evidence are in
`results/learning-study-2026-10-15/cpu-100k/`. Local acceptance: 115 tests passed
in 76.265 seconds, including transport-failure handling. No new test audit ran.

### What the selected baseline policies actually do

The three unchanged-reward CPU seeds' selected checkpoints were long in 92.8%
of validation sessions, short in 1.8% and flat in 5.4%, with mean absolute exposure
85.8%. Each separate stock episode averaged 744 fills but only 57 new entries:
most fills adjusted existing positions. Modeled costs averaged $4,611 on each
$100,000 starting-equity episode, already deducted from reported net returns.
These are equal-symbol, equal-seed averages, not statistics for one investable
shared portfolio. `cpu-100k/behavior-summary.json` preserves the underlying
seed summaries and selected bundle identities.

The action's first component specifies desired total exposure, not an incremental
purchase. Subsequent components select a stop at 0.5–5 ATR, a first profit target
at 0.5–8 ATR, and a target-zone width of 0.25–4 ATR. Near-target execution closes
roughly half the current position; the far target closes the remainder. Whole
shares, next-session fills, gap checks, modeled costs and the holding limit apply.

### Per-stock comparison against holding the same stock

For the selected unchanged-reward baseline CPU checkpoints, the three-seed mean
beat buy-and-hold in 6 of 26 separate validation episodes. Mean net return was
19.70% versus 30.60%, a 10.90-point shortfall. PLD outperformed by 21.58 points;
XOM underperformed by 82.62 points. NEE and DIS beat holding by losing less, not
by earning positive returns. Full per-symbol returns, seed variability, costs,
drawdowns and source identities are preserved in `cpu-100k/per-stock-comparison.csv`
and its JSON companion. These validation-selected comparisons are optimistic
relative to a fresh audit. The policy has a 20-session holding cap and protective
rules; the buy-and-hold benchmark holds continuously. Both use matched dates and
modeled fee/impact rates, and neither forces liquidation at the data boundary.
