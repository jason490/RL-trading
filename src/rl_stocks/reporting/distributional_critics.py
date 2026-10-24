"""Fixed critic-study gates and historical-control checks; never select anew."""

import copy
import math

from rl_stocks.reporting.history import screen_gate as matched_screen_gate
from rl_stocks.training.distributional_critics import protocol


def historical_reproduction(current, historical, current_reports, historical_reports):
    """Require every registered checkpoint and selected/final symbol within bounds."""
    spec = protocol()["historical_control_reproduction"]
    expected = spec["all_checkpoint_scores"]
    actual = current["validation_checkpoints"]
    if len(actual) != len(expected):
        raise ValueError("exact historical checkpoint schedule required")
    discrepancies = []
    for row, reference in zip(actual, expected, strict=True):
        if (row["steps"], row["phase"]) != (reference["steps"], reference["phase"]):
            raise ValueError("historical checkpoint phase/order mismatch")
        difference = 100 * abs(row["score"] - reference["score"])
        if not math.isfinite(difference):
            raise ValueError("finite historical checkpoint discrepancy required")
        discrepancies.append(difference)
    if (
        historical["git"]["revision"] != spec["source_revision"]
        or historical["selected_score"] != spec["selected_score"]
        or historical["validation_checkpoints"][-1]["score"] != spec["final_score"]
    ):
        raise ValueError("registered historical native SAC source required")
    historical_points = historical["validation_checkpoints"]
    if len(historical_points) != len(expected) or any(
        any(row[k] != reference[k] for k in ("steps", "phase", "score"))
        for row, reference in zip(historical_points, expected, strict=True)
    ):
        raise ValueError("registered historical checkpoint values required")
    per_symbol = {}
    for phase in ("selected", "final"):

        def panel(report):
            result = {r["symbol"]: r["metrics"]["net_return"] for r in report.values()}
            if len(result) != len(report) or len(result) != 26:
                raise ValueError("one original development episode per symbol required")
            return result

        left, right = panel(current_reports[phase]), panel(historical_reports[phase])
        if set(left) != set(right):
            raise ValueError("same historical-control symbol panel required")
        values = {s: 100 * abs(left[s] - right[s]) for s in left}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("finite historical per-symbol discrepancy required")
        per_symbol[phase] = values
    maximum = max(discrepancies)
    symbol_maximum = max(v for phase in per_symbol.values() for v in phase.values())
    passed = (
        maximum <= spec["maximum_checkpoint_mean_return_discrepancy_pp"]
        and symbol_maximum <= spec["maximum_selected_and_final_per_symbol_return_discrepancy_pp"]
    )
    return dict(
        passed=passed,
        checkpoint_discrepancy_pp=discrepancies,
        maximum_checkpoint_discrepancy_pp=maximum,
        per_symbol_discrepancy_pp=per_symbol,
        maximum_per_symbol_discrepancy_pp=symbol_maximum,
        tolerances_changed=False,
        automatic_retry_authorized=False,
    )


def screen_gate(cells, reproduction):
    """Both fresh controls, absolute gates and historical reproduction are required."""
    spec = copy.deepcopy(protocol())
    gate = spec["screen_gate"]
    gate["minimum_selected_and_final_stock_and_etf_excess_pp"] = gate[
        "minimum_selected_and_final_stock_and_etf_holding_excess_pp"
    ]
    gate["minimum_selected_and_final_gain_pp_vs_each_control"] = gate[
        "minimum_selected_and_final_mean_gain_pp_vs_each_control"
    ]
    gate["minimum_selected_and_final_median_stock_excess_pp"] = gate[
        "minimum_selected_and_final_median_stock_holding_excess_pp"
    ]
    result = matched_screen_gate(cells, spec)
    result["performance_gates_passed"] = result["passed"]
    result["historical_control_reproduced"] = reproduction["passed"] is True
    result["passed"] = result["passed"] and result["historical_control_reproduced"]
    result["promoted_candidate"] = spec["candidate"] if result["passed"] else None
    result["interpretation_valid"] = result["historical_control_reproduced"]
    return result
