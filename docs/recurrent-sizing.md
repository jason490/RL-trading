# Fixed recurrent exposure-sizing replay

[The saved-action audit](../results/recurrent-actions-2026-10-21/README.md) finds
roughly94% direction agreement but smaller requested long size before protections.
This motivates a single sizing diagnostic, not a change to the failed recurrent
state protocol or proof that full exposure is beneficial.

[The immutable protocol](../configs/experiments/recurrent_unit_sizing.json)
fixes the saved masked recurrent100k seed11 selected/final models and four cells:
unrestricted and unit-magnitude exposure for each phase, all26 original development
episodes. A nonzero first action component becomes its sign (−1/+1); zero remains
zero. The three learned stop/near/width outputs are unchanged. Every symbol uses
the same rule. No future-return label activates the transform. Costs, borrow,
whole shares,100k equity,1× cap, next-session execution, protective fills and
20-session maximum holding remain. Risk approval can reduce actual exposure.

A fresh simulator and fresh recurrent memory start each episode/mode. Modified
fills change subsequent causal account observations and hidden/cell state. No
weights, preprocessing, updates, checkpoint choice or training length change.
Local CPU inference is permitted; no Colab allocation or fitting is needed.

Before either candidate replay, BOTH unrestricted paths must match the original
accepted reports: exact symbols/sessions/quantities/baselines, daily equity≤$0.10,
costs≤$0.01, actions≤2e-6, returns/drawdown≤1e-6 absolute difference. Exact report
digests and maximum differences are separately recorded. Any failure stops before
candidate evaluation, saves the failure evidence, and does not relax tolerances.

Candidate selected AND final gains must each be≥2pp versus the unchanged policy;
final stock/ETF gains nonnegative and mean drawdown worsening≤2pp. Selected AND
final stock/ETF excess and median stock excess must each be≥−2pp versus holding.
The diagnostic cannot achieve the overall goal or promote a policy by itself:
three-seed confirmation and prospective evidence remain required. Failure means
no magnitude grid, threshold sweep, phase reselection or extra training here.

Require clean committed source/protocol, verified original owned-session shutdown,
accepted report hashes, frozen selected-weight integrity and unchanged model/
normalizer/source hashes. Save four compressed full paths plus a compact summary.

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_recurrent_sizing.py \
 --output artifacts/recurrent-unit-sizing
```

Implementation is prepared and tested; actual replay has not run at preparation.
12 filter/recurrent-state/reference checks pass without fitting. The recurrent
synthetic replay confirms carried state within episodes, reset between symbols,
changed account observations and preserved protective components. No actual
financial result or profitability claim follows from those fixtures. Project
usage remains38.014408/150, with no active owned sessions.
