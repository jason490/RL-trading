"""Fixed critic-study gates and historical-control checks; never select anew."""

import copy

from rl_stocks.reporting.history import screen_gate as matched_screen_gate
from rl_stocks.training.distributional_critics import protocol


def historical_reproduction(current, historical, current_reports, historical_reports):
    from rl_stocks.reporting.historical_control import historical_reproduction as reproduce

    return reproduce(
        current,
        historical,
        current_reports,
        historical_reports,
        protocol()["historical_control_reproduction"],
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
