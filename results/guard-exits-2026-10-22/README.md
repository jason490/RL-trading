# Saved exit and inherited-position partition

Four accepted phases/104 episodes, exact accepted hashes and original26 panel.
No fit, model restoration, data/provider access, Colab or retired-test evaluation.
[Summary and per-symbol partitions](summary.json), [group table](groups.csv).

## Method

Prior-session closing quantity defines inherited flat/long/short position, with
initial position flat. Same-session events define neither, protection, holding
limit or both flags. All12 buckets remain, including zero counts. Protection
includes partial target fills. Flags may overlap and do not imply every fill
closed the position. Event order is simulator order, which can differ from
timestamp order for gap target fills. Validate event quantity/delta chains,
closing positions, equity chain, holding terminal return and aggregate excess.

Policy and holding daily dollar equity changes are each divided by original
$100,000 and expressed in percentage points. Their differences telescope to
terminal net excess, despite diverging policy/holding equity bases. Equal-symbol
means preserve group identities. These are observed-session partitions, not
causal attribution, geometric daily returns, cost-free replay or isolated gap
P&L. Reports lack opening prices for sessions without fills; exact overnight
versus intraday decomposition requires immutable bars. Split-induced quantity
changes without matching fills are rejected; current revised Yahoo dataset has
normalized split1.

## Findings

Guard final ETF shortfall13.35pp partitions:

| Inherited position / events | Mean sessions | Excess contribution |
| --- | ---: | ---: |
| Flat / neither |36.25|−5.13pp|
| Long / neither |652.50|−2.40pp|
| Long / protection |0.75|−0.03pp|
| Long / holding limit |34.50|−1.06pp|
| Short / neither |8.00|−4.74pp|

Guard final stock shortfall4.85pp includes−4.60pp on flat/no-exit sessions,
−0.93pp long/holding sessions, while inherited-short buckets contribute+0.79pp.
This does not contradict positive net short contribution: daily excess includes
matched holding changes and other fills on that day.

Control final ETF long/protection and long/both contributions−11.74/−1.03pp
are much larger, but guard still underperforms with few protective-fill days.
Guard selected→final ETF flat/neither contribution worsens−2.55→−5.13pp and
short/neither−0.07→−4.74pp. Protection is not a sufficient observed explanation
for the guarded final miss.

## Next diagnostic

Preregister an exact no-fit cash-flow identity using immutable matched bars:
prior quantity × split-adjusted overnight move, opening inventory/trades ×
intraday price moves, cash dividends, fees/impact/borrow, and holding equivalents.
Preserve actual event order and conservative target-gap pricing; do not infer
fill phase from timestamp alone. Require full per-session reconciliation before
interpreting missed overnight exposure after mandatory holding exits. No
holding-rule relaxation, feature change, family extension or selection follows
from these descriptive buckets alone.

26 focused tests pass, including inherited versus closing position, overlapping
exit flags, corrupt quantity/holding/equity rejection and legacy audit selection.
Sources unchanged. Project38.39201565 and official account349.51 unchanged; all
owned sessions remain stopped.
