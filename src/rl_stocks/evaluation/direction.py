"""Cash-flow attribution by position direction, including still-open positions.

This is an evaluation-only diagnostic. It must never supply observations, rewards
or future-return labels to a policy.
"""
import math


def directional_attribution(episode, rows, events, initial=100000.):
    """Reconcile long/short marked P&L to every recorded equity observation.

    Start flat, process corporate actions before fills, and allocate reversal
    fees in proportion to closing/opening shares. Sale proceeds alone are never
    counted as short profit: the remaining signed position is marked each close.
    """
    bars = episode["bars"][1:]
    if len(rows) > len(bars) or not rows:
        raise ValueError("nonempty episode prefix required for direction attribution")
    bars = bars[:len(rows)]  # Bankruptcy can terminate before the dataset cutoff.
    sessions = {b["session"] for b in bars}
    by_session = {s: [] for s in sessions}
    for event in events:
        session = event["time"][:10]
        if session not in by_session:
            raise ValueError("fill outside evaluation episode")
        by_session[session].append(event)
    cash = {"long": 0., "short": 0.}
    costs = {"long": 0., "short": 0.}
    active = {"long": 0, "short": 0}
    quantity = 0
    max_error = 0.
    side = lambda q: "long" if q > 0 else "short"
    for bar, row in zip(bars, rows):
        if row["session"] != bar["session"]:
            raise ValueError("direction attribution session mismatch")
        touched = set()
        if quantity:
            direction = side(quantity)
            touched.add(direction)
            exact = quantity * bar.get("split", 1.)
            quantity = math.trunc(exact)
            cash[direction] += ((exact - quantity) * bar["open"]
                                + quantity * bar.get("dividend", 0.))
        for event in by_session[bar["session"]]:
            delta = event["delta"]
            if not delta or event["quantity"] != quantity + delta:
                raise ValueError("fill quantity does not reconcile")
            closed = min(abs(quantity), abs(delta)) if quantity * delta < 0 else 0
            opened = abs(delta) - closed
            portions = []
            if closed:
                portions.append((side(quantity), -math.copysign(closed, quantity)))
            if opened:
                portions.append((side(delta), math.copysign(opened, delta)))
            for direction, signed_shares in portions:
                fraction = abs(signed_shares) / abs(delta)
                cash[direction] -= signed_shares * event["price"] + fraction * event["fee"]
                costs[direction] += fraction * (event["fee"] + event["impact"])
                touched.add(direction)
            quantity = event["quantity"]
        borrow = row.get("borrow_cost", 0.)
        cash["short"] -= borrow
        costs["short"] += borrow
        for direction in touched:
            active[direction] += 1
        if quantity != row["quantity"]:
            raise ValueError("end-of-session quantity does not reconcile")
        marked = dict(cash)
        if quantity:
            marked[side(quantity)] += quantity * bar["close"]
        error = abs(sum(marked.values()) - (row["equity"] - initial))
        max_error = max(max_error, error)
        if not math.isclose(sum(marked.values()), row["equity"] - initial,
                            rel_tol=1e-9, abs_tol=1e-5):
            raise ValueError("direction P&L does not reconcile to equity")
    return {
        "method": "signed-cash-flows-plus-final-mark-v1",
        "long_net_pnl": marked["long"], "short_net_pnl": marked["short"],
        "long_return_contribution": marked["long"] / initial,
        "short_return_contribution": marked["short"] / initial,
        "long_costs": costs["long"], "short_costs": costs["short"],
        "long_active_sessions": active["long"], "short_active_sessions": active["short"],
        "sessions": len(rows), "max_equity_reconciliation_error": max_error,
        "includes_open_position_mark": True,
    }


def declining_stock_summary(results, stock_symbols):
    """Ex-post reporting cohort only; never a model-selection or sampling label."""
    records = []
    for key, result in results.items():
        hold = result["baselines"]["buy_hold"]["metrics"]["net_return"]
        if result["symbol"] not in stock_symbols or hold >= 0:
            continue
        attribution = result["directional"]
        policy = result["metrics"]["net_return"]
        records.append({"episode": key, "symbol": result["symbol"],
                        "policy_return": policy, "buy_hold_return": hold,
                        **attribution,
                        "positive_policy_and_short_pnl": policy > 0 and attribution["short_net_pnl"] > 0})
    return {"cohort": "ex-post-negative-buy-hold-stocks-reporting-only",
            "count": len(records),
            "positive_policy_and_short_pnl_count": sum(r["positive_policy_and_short_pnl"] for r in records),
            "records": records}
