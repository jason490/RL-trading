"""Fixed VIX performance/reproduction gates; never reselect checkpoints."""

from rl_stocks.reporting.historical_control import historical_reproduction as reproduce
from rl_stocks.reporting.history import screen_gate as matched_gate
from rl_stocks.training.vix_spec import preregistration


def historical_reproduction(current, historical, current_reports, historical_reports):
    return reproduce(
        current,
        historical,
        current_reports,
        historical_reports,
        preregistration()["historical_control_reproduction"],
    )


def screen_gate(cells, reproduction):
    spec = preregistration()
    result = matched_gate(cells, spec)
    result["performance_gates_passed"] = result["passed"]
    result["historical_control_reproduced"] = reproduction["passed"] is True
    result["passed"] = result["passed"] and result["historical_control_reproduced"]
    result["promoted_candidate"] = spec["candidate"] if result["passed"] else None
    result["interpretation_valid"] = result["historical_control_reproduced"]
    result["failure_closes_fixed_family"] = not result["passed"]
    return result
