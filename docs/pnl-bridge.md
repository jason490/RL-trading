# Overnight and intraday saved-fill bridge

This fixed descriptive diagnostic covers the two accepted update-guard arms,
selected and final checkpoints, and all original26 development symbols. No fit,
new prediction, provider access, Colab, retired-test access or checkpoint choice.
Commit this method before opening immutable bars for the actual diagnostic.

For prior quantity q, split s, post-split whole quantity Q=trunc(q*s), previous
close C0, current open O and close C, the exact equity-change identity is:

`q*(s*O-C0) + Q*(C-O) + sum(delta*(C-reference_fill)) + Q*dividend - fees - impact - borrow`

The first term includes fractional split cash-in-lieu. Actual signed fills are
used in simulator order, including gap targets whose timestamps can differ from
execution order. Impact must equal signed delta times actual-minus-reference
fill price. Event quantity chains, closing inventory, equity chains and every
session's sum must reconcile within1e-5 absolute/1e-9 relative dollars; terminal
sum also reconciles. All inputs must be finite and only complete matched
2021–2023 validation paths are accepted. No inferred OHLC path or target/stop
ordering replaces actual fills.

Reconstruct the declared holding baseline with its original first-open whole
share entry, corporate actions and costs; require exactly matching saved equity
and sessions. The holding baseline does not pay short borrow. Report equal-symbol
stock/ETF component differences in percentage points of original capital.
The inherited intraday and fill-to-close terms offset inventory changes; neither
alone is the causal value of a fill. Overnight difference is actual portfolio
exposure arithmetic, not proof that holding-rule removal would achieve it.
Changes to policy/protection would alter subsequent actions and equity.

Run after committed implementation:

```bash
PYTHONPATH=src .venv/bin/python scripts/diagnose_guard_bridge.py \
 --output results/guard-bridge-2026-10-22/summary.json
```

The script verifies exact accepted reports/panel and immutable dataset accounting,
opens only development episodes, retains source hashes and compact per-symbol
component totals, and checks source immutability afterward. No raw bars are
included in the public summary. Original source/data/ledger remain retained.

40 focused checks pass before actual execution, including synthetic long/short,
fractional splits, dividends, partial exits, reversals, borrow, ordered fills,
corrupt quantities/costs/equity, and hand-calculated holding reconstruction.
Actual results pending. No policy promotion follows from this diagnostic.


## Actual bar-backed cash-flow bridge

Committed method96dddd9 precedes four accepted phases/104 episodes. Every
policy/holding session reconciles (worst error1.28e-10 dollars); source hashes
unchanged, no fitting/new prediction/provider/Colab/test access. Guard final
ETF excess−13.3541pp comprises overnight−7.6351, inherited intraday−1.2040,
fill-to-close−1.5044, dividends−0.4841 and fees/impact/borrow−2.5265pp.
Overnight differences are substantial, not proof of a holding-rule cause. Next
fixed partition by inherited side and preceding actual exit reason before any
intervention. Project38.39201565 unchanged, all sessions stopped.
[Evidence](../results/guard-bridge-2026-10-22/README.md).


## Fixed overnight partition

Partition all four accepted phases by inherited flat/long/short side and the
preceding session's neither/protection/holding/both exit flags; initial session
has its own flag. Retain all15 buckets and actual partial-target overlap. No
current-session events label the preceding overnight move. Report policy and
holding overnight contributions and their difference, scaled by original equity,
with equal-symbol stock/ETF means. Counts and component sums must reconcile to
the original exact bridge. Flags are descriptive associations, never proof that
removing an exit would reproduce holding.41 focused tests pass before actual
execution. Run the same script to a new output directory; preserve prior output.
