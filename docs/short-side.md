# Short-side research

The policy can request negative signed exposure using the existing four-component
action schema. Simulation enables shorts by default, limits gross exposure to
100%, and uses whole shares, next-session fills, protective stops, partial target
exits and a 20-session holding limit. A lower stock price can produce positive
short P&L. The model must learn when to enter, cover or reverse; a falling stock
does not guarantee a profitable policy.

## What counts as success

Report the same-symbol cost-aware buy-and-hold return, the policy's total net
return, and its separate long/short marked P&L. A policy that loses 10% when
holding loses 40% has reduced losses; it has not gone positive. A policy with
positive short P&L but larger losses from long trades has also failed the user's
positive-total-return objective.

For stocks with negative full-episode buy-and-hold returns, report both:

- Whether total policy return is positive.
- Whether the short contribution after costs is positive.

This declining-stock group is an **ex-post reporting cohort only**. Its labels
must not become observations, training sampling weights, reward bonuses,
checkpoint selection criteria or a source of future direction. Any later
training regime sampling must use a separately declared causal rule based only
on information available at the decision. Include rising stocks and ETFs in
the broader objective so short-only behavior does not pass by construction.

## Accounting method

`evaluation/direction.py` starts flat and reconstructs side-specific cash flows
from the complete prefix of episode fills. It processes splits, fractional
cash in lieu and signed dividends before fills; short positions pay dividends.
It allocates entry and exit fees and adverse fill costs to the affected side,
splitting a reversal in proportion to shares closed and opened. It subtracts
daily borrow charges from the short side. The remaining position is marked at
each session close, so short-sale proceeds are never mistaken for profit.

Long P&L plus short P&L must equal net marked equity minus initial equity at
**every session**, within floating-point tolerance. Quantity and session
mismatches fail the audit. The terminal mark includes unrealized gains/losses,
consistent with existing evaluation; it does not add a hypothetical liquidation.
These contributions are cash P&L divided by initial equity, not independent
compounded long-only/short-only portfolio returns. Active-session counts include
overnight exposure and positions opened and closed intraday.

New evaluation JSON and HTML reports contain `directional`. Old saved reports
can be audited without retraining or changing their checkpoints:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_short_side.py \
  /path/to/matched/run-11 /path/to/matched/run-23 /path/to/matched/run-37 \
  --universe configs/universes/sector_research.json \
  --output artifacts/short-audit
```

Pass explicit run folders containing `metadata.json`, `selected-validation.json`
and `validation.json`. Dataset, simulation, transition budget and algorithm seed
sets must match for each algorithm/initialization variant. Variants remain
separate even when algorithm and seed are identical; older runs without an
initialization field are labeled `standard`. Local immutable datasets must
still be available. The command rejects test reports, unpaired seed sets and
duplicate algorithm/variant/seed cells. `summary.json` preserves
source report checksums; `per-symbol.csv` includes all stock/ETF rows and kinds.
Selected and final weights remain separate.

## Verified evidence and limits

[The 2026-10-19 audit](../results/short-side-2026-10-19/README.md) covers nine
previously trained 100k PPO/SAC/recurrent PPO runs, three seeds each. It adds no
training or checkpoint selection. Unit tests verify profitable shorts on a
falling stock after fees, impact, borrow and dividends; losses on rising prices;
disabled shorts; reversals; scale-outs; split cash in lieu; gap stops; flat
paths; early termination prefixes; and reconciliation failures.

Broker borrow availability, recalls, actual variable rates and instrument
eligibility are not established by this simulation. Existing frozen policies
remain research-only. No broker orders are authorized or submitted.

## Known execution state

[Versioned execution observations](execution-observations.md) expose held stops,
targets, partial-exit status and prior protective actions in an optional full
profile. This addresses demonstrated state aliasing for both long and short
positions. It requires matched Colab validation before claiming improved short
profitability; legacy frozen policies retain their original input contract.
