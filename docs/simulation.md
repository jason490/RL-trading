# Portfolio accounting and execution simulation

The pure-Python simulator is independently testable before Gymnasium/SB3 installation. It models a single instrument per episode, default $100,000 cash, whole shares, signed long/short holdings, at most 100% approved target notional and a maximum 20-session holding period.

## Accounting source of truth

Marked equity is cash + signed quantity × current price. A buy consumes cash; a sale credits cash, including short-sale proceeds. Basis is weighted when adding in the same direction, retained for reductions, reset when flat, and reset to the new fill when reversing. Every trade records the signed change and resulting holdings. Repeating an unchanged desired target does not submit another order-sized addition.

For 100 shares bought at $100 with zero costs, cash becomes $90,000. Marking at $110 gives $101,000 equity. Closing then shorting 100 at $110 and marking at $100 gives another $1,000 gain. Tests exercise both directions, basis changes and close-before-reverse ordering. Costs are separate cash debits and reduce net equity exactly once.

## Costs and assumptions

| Setting | Default | Treatment |
| --- | --- | --- |
| Commission/fees | 1 bp | abs(delta shares) × adverse fill × fee rate |
| Half-spread + slippage | 2 bp | Fill above reference for buys, below for sells |
| Short borrow | 3% annual / 252 sessions | abs(short quantity) × close × daily rate |
| Cash interest | 0 | No cash-yield credit |
| Exposure cap | 100% | Cost-aware next-open target gate |
| Quantity | Whole shares | Floor desired absolute quantity |
| Holding limit | 20 sessions | Close at completed session after age reaches limit |
| Short eligibility | Enabled in simulation | Declared research assumption, not broker authorization |

Fees, adverse impact and borrow are included in cumulative costs. Turnover uses traded reference notional. Margin financing beyond the declared borrow rate, broker-specific borrow availability, stochastic liquidity and taxes are not modeled. Actual price changes/costs can alter exposure after the gate. Hard constraints remain outside reward shaping.

## Per-bar order of operations

1. Decode yesterday's completed available bar, ATR, known equity and target action.
2. Verify that its availability precedes today's execution open.
3. Apply today's effective split/share/basis/protection conversion and dividend cash flow.
4. Process existing overnight stop/target gaps before voluntary adjustments.
5. If no protective gap fill superseded the signal, validate entry geometry and cap desired quantity at the next open; close a reversal first.
6. Create/update protection only when opening or explicitly changing protective action components.
7. Replay intrabar stops/zone limits with conservative stop-first ordering.
8. Increment holding age, enforce the holding limit, charge short borrow and mark the close.
9. Compute net log equity reward and accounting diagnostics.

No future open is a feature in yesterday's observation. It is an execution outcome that can reduce/reject a previously sized request. Existing protection survives a rejected voluntary update. Changes to raw action and actual approved/filled quantity are recorded separately.

## Protection and gaps

Long geometry is stop < entry < target near < target far; short geometry reverses it. A new next-open entry outside that geometry is skipped with `gap_invalid_geometry`. An existing stop that gaps through its level fills at the worse of open/trigger, plus adverse costs; it never guarantees the trigger price.

When the same daily OHLC bar touches stop and profit target with unknown chronology, stop wins. No finer-bar chronology is invented. Take-profits fill at the declared limit even if the open is favorable; favorable gap improvement is deliberately omitted. This conservative replay convention should receive sensitivity analysis before market performance claims.

Near closes half of the current whole-share position, rounding down but exiting at least one share. Far closes the remaining shares. Both boundaries affect a position large enough to scale out. For a one-share position, near closes everything; far has no remaining shares to act on. Protection remains on the remainder after the near exit. A gap near fill suppresses voluntary refill in the same bar; a later daily signal may deliberately restore desired exposure.

Unchanged protective components remain anchored to the original protection. A changed protective vector is an explicit update decoded from the current completed decision reference and checked at the next open. Protective exits always override holding preferences.

## Corporate actions and missing data

Raw effective split ratios multiply shares and divide basis/protective prices; fractional residuals settle at the open. Already split-adjusted source bars must use split=1. Dividends credit long holdings and debit shorts before opening changes, using an ex-date cash-flow approximation rather than actual payment-date cash availability. Data quality must verify that convention before default acceptance.

Verified missing sessions break feature warmup and episodes. Nothing silently fills arbitrary OHLC gaps. An overnight market gap between observed consecutive sessions is simulated from the next open; it differs from absent data.

## Boundaries and reward

Reward is log(new marked equity / previous marked equity). It includes all cost/cash-flow effects once; realized P&L is not added again. Nonpositive equity causes true termination with finite reward -20. Data end and artificial episode limits are truncations and use marked terminal equity, with no artificial liquidation cost. This supports timeout bootstrapping but means final returns can include open-position P&L.

The simulator is an explicit research approximation. Finer execution replay, realistic financing/borrow archives and a broker order lifecycle remain separate extensions. Tests establish deterministic accounting, not realism sufficient for live performance claims.
