"""Paired saved-path observations; additive P&L differences are not causes."""

from collections import defaultdict
import math

from rl_stocks.core.contracts import digest

PROTECTIVE = {"stop", "target_near", "target_far"}


def side(quantity):
    return "long" if quantity > 0 else "short" if quantity < 0 else "flat"


def compare_paths(before, after):
    """Caller also validates full accounting; this checks paired rows/fill chains."""
    if (
        before["symbol"] != after["symbol"]
        or before["split"] != "validation"
        or after["split"] != "validation"
    ):
        raise ValueError("same development symbol required")
    left, right = before["rows"], after["rows"]
    dates = [r["session"] for r in left]
    if (
        not left
        or dates != sorted(set(dates))
        or dates != [r["session"] for r in right]
        or any(not "2021-01-01" <= d <= "2023-12-31" for d in dates)
        or digest(before["baselines"]["buy_hold"]) != digest(after["baselines"]["buy_hold"])
    ):
        raise ValueError("identical chronological development sessions and holding paths required")
    events = []
    for report in (before, after):
        by_day = {d: [] for d in dates}
        for event in report["events"]:
            day = event["time"][:10]
            if day not in by_day:
                raise ValueError("saved fill outside paired sessions")
            by_day[day].append(event)
        events.append(by_day)
    first = dict(
        action=None,
        approved_quantity=None,
        closing_quantity=None,
        closing_side=None,
        fill_reasons=None,
        protective_reasons=None,
    )
    counts = {k: 0 for k in first}
    buckets = defaultdict(lambda: dict(sessions=0, net_difference_pp=0.0))
    prior = [100000.0, 100000.0]
    quantities = [0, 0]
    biggest = dict(session=None, net_difference_pp=0.0)
    total = 0.0
    for lrow, rrow in zip(left, right, strict=True):
        day = lrow["session"]
        if lrow["price"] != rrow["price"] or not math.isfinite(lrow["price"]) or lrow["price"] <= 0:
            raise ValueError("same finite saved market price required")
        inherited = "/".join(side(q) for q in quantities)
        reason_lists = []
        pnl = []
        for i, row in enumerate((lrow, rrow)):
            if (
                not math.isfinite(row["equity"])
                or not math.isfinite(row["previous_equity"])
                or not math.isclose(row["previous_equity"], prior[i], abs_tol=1e-6, rel_tol=0)
                or type(row["quantity"]) is not int
                or type(row["approved_target_quantity"]) is not int
                or len(row["raw_action"]) != 4
                or any(not math.isfinite(v) or abs(v) > 1 for v in row["raw_action"])
            ):
                raise ValueError("finite chained equity, whole shares and bounded actions required")
            for event in events[i][day]:
                if event["quantity"] - event["delta"] != quantities[i]:
                    raise ValueError("saved fill quantity chain mismatch")
                quantities[i] = event["quantity"]
            if quantities[i] != row["quantity"]:
                raise ValueError("saved closing quantity mismatch")
            reason_lists.append([e["reason"] for e in events[i][day]])
            pnl.append(row["equity"] - prior[i])
            prior[i] = row["equity"]
        protection = [[r for r in reasons if r in PROTECTIVE] for reasons in reason_lists]
        differences = dict(
            action=lrow["raw_action"] != rrow["raw_action"],
            approved_quantity=lrow["approved_target_quantity"] != rrow["approved_target_quantity"],
            closing_quantity=lrow["quantity"] != rrow["quantity"],
            closing_side=side(lrow["quantity"]) != side(rrow["quantity"]),
            fill_reasons=reason_lists[0] != reason_lists[1],
            protective_reasons=protection[0] != protection[1],
        )
        for key, different in differences.items():
            if different:
                counts[key] += 1
                if first[key] is None:
                    first[key] = dict(
                        session=day,
                        before_closing_side=side(lrow["quantity"]),
                        after_closing_side=side(rrow["quantity"]),
                        before_reasons=reason_lists[0],
                        after_reasons=reason_lists[1],
                    )
        delta = (pnl[1] - pnl[0]) / 1000.0
        total += delta
        key = inherited + (
            "/protection_different" if differences["protective_reasons"] else "/protection_same"
        )
        buckets[key]["sessions"] += 1
        buckets[key]["net_difference_pp"] += delta
        if abs(delta) > abs(biggest["net_difference_pp"]):
            biggest = dict(session=day, net_difference_pp=delta)
    for i, report in enumerate((before, after)):
        if not math.isclose(
            prior[i] / 100000.0 - 1, report["metrics"]["net_return"], abs_tol=1e-10, rel_tol=0
        ):
            raise ValueError("saved return endpoint mismatch")
    expected = 100 * (after["metrics"]["net_return"] - before["metrics"]["net_return"])
    if not math.isclose(total, expected, abs_tol=1e-8, rel_tol=0):
        raise ValueError("paired daily bridge did not reconcile")
    return dict(
        symbol=before["symbol"],
        sessions=len(left),
        first_differences=first,
        difference_session_counts=counts,
        buckets=dict(buckets),
        largest_absolute_daily_difference=biggest,
        final_minus_selected_pp=total,
    )
