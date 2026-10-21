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

## Completed fixed sizing replay — full gates failed

Both unrestricted phases pass preregistered tolerances with exact quantities;
reports are not byte-identical (maximum equity differences$0.013815/$0.005020).
Four cells complete with unchanged weights/updates/normalization. Unit selected/
final means28.53%/32.80% versus control24.99%/24.04%, holding30.60%. Final stocks
35.40% (+2.77pp excess), ETFs18.55% (−0.92pp), median stock excess+0.41pp meet
final comparability targets. Selected ETF excess−3.36pp and median stock−7.56pp
fail; final mean drawdown worsens3.21pp versus allowed2. Unit joint short success
0/3 in both phases. Preserve the full failure: no threshold sweep, phase reselection,
extra training or promotion under this replay protocol. No new Colab units.

Eight published asset sizes/digests match; verified duplicates removed, retained
paths reproduce all four group/short metrics exactly.12 filter/state/reference
checks passed. Overall goal remains unmet. A separately preregistered training/
execution-alignment hypothesis may examine why deploying unit size reverses
selected/final ranking; it is not confirmation of this failed replay or a proven
cause. [Outcome and audit](../results/recurrent-unit-sizing-2026-10-21/README.md).
