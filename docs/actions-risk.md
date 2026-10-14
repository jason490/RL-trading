# Policy action and risk contract

Version `target-zone-v1` uses a normalized four-component continuous Box shared by PPO/SAC/TD3. Every component is finite; malformed/nonfinite actions fail before accounting changes. The decoder clips to [-1,1] and records the raw and clipped vectors.

| Index | Meaning | Decoding |
| --- | --- | --- |
| 0 | Signed desired **total** exposure | Negative short, positive long, zero flat; abs(action) is equity allocation |
| 1 | Stop distance | (0.5 + (a+1)/2 × 4.5) × causal ATR |
| 2 | Target near distance | (0.5 + (a+1)/2 × 7.5) × causal ATR |
| 3 | Zone width beyond near | (0.25 + (a+1)/2 × 3.75) × causal ATR |

Stop/near/width are learned independently. There is no fixed reward-to-risk formula. Whole-share desired quantity is sign(exposure) × floor(abs(exposure) × known equity / completed decision close).

For a long: stop < executable entry < near < far. For a short: far < near < executable entry < stop. All protective levels must be positive. The canonical decision includes symbol, decision/expiry timestamps, reference, raw/clipped action, signed exposure/quantity, absolute notional, direction and side, stop/near/far, ordered lower/upper zone and intent. ATR/reference/equity must be finite and positive.

## Reconciliation and approvals

The quantity is a target holding, not another order to add each inference call. Compare it with actual quantity to derive hold/add/reduce/close/reverse. A reversal first closes existing holdings before opening the opposite side. Small changes naturally disappear under whole-share rounding; there is no additional hidden deadband.

At the next open, a separate gate caps absolute target quantity to the current marked equity / adverse-cost-adjusted price, subject to at most 100% exposure. It may reduce a yesterday-sized target after a gap. The simulator records `approved_target_quantity`, `risk_clipped`, rejection reason and eventual actual quantity separately. `approved_target_quantity=None` means no voluntary target was executed (for example after a protective gap fill or rejected geometry).

Effective splits convert yesterday's reference, quantity and protective geometry into today's share/price units before the gate. Existing protection is similarly adjusted. These are corporate-action transformations, not extra policy exposure.

Long simulation respects available cash via conservative cost-aware sizing. Short collateral/borrowing is simplified: a configurable short-enabled switch and notional cap are enforced, and daily borrow cost is charged. Real broker shortability/borrow/margin restrictions remain outside this research simulator and must be audited before paper execution.

Existing protection remains anchored when the three protective action components are unchanged. An explicit change updates levels from the completed decision reference; a gap can reject that proposal. Actual protective fills override voluntary signal changes and holding preferences. Read [simulation.md](simulation.md) for ordering and scale-out details.

Research decisions include an expiry; frozen `decide()` additionally checks source availability, current time and maximum age. No decision function sends an order. A future broker adapter must revalidate buying power, account/asset restrictions, position state, tick sizes, order lifecycle and protective coverage independently.

Finite actions that yield nonpositive protective prices are rejected by the simulator with `nonpositive_protective_price`; the raw action is retained and existing protective orders still execute. Frozen research decision decoding rejects that geometry rather than exporting an executable proposal. A tiny exposure that rounds to zero shares is flat, so closing a position is not incorrectly blocked by unused stop/target geometry.
