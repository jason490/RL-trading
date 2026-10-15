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
