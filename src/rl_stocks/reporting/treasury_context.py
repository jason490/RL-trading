"""Fixed Treasury performance/reproduction gates; never reselect checkpoints."""

from rl_stocks.reporting.historical_control import historical_reproduction as reproduce
from rl_stocks.reporting.history import screen_gate as matched_gate
from rl_stocks.training.treasury_spec import protocol as preregistration


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


def validate_sampling(cells):
    """Paired filtered paths match; original2010 legacy locations may differ."""
    from rl_stocks.core.contracts import canonical

    if set(cells) != set(preregistration()["arms"]):
        raise ValueError("all three Treasury sampling records required")
    if canonical(cells["full"]["sampling"]) != canonical(cells["masked"]["sampling"]):
        raise ValueError("full/masked actual symbol/group/year sampling must match")
    if canonical(cells["full"]["sampling_paths"]) != canonical(cells["masked"]["sampling_paths"]):
        raise ValueError("full/masked actual window/date paths must match")
    common = [{k: v for k, v in cells[a]["sampling"].items() if k != "year_counts"}
              for a in preregistration()["arms"]]
    if any(canonical(row) != canonical(common[0]) for row in common[1:]):
        raise ValueError("fixed legacy/filtered symbol deck and group coverage required")
    return dict(full_masked_paths_exact=True, full_masked_counts_exact=True, legacy_symbol_group_coverage_exact=True,
                legacy_year_locations_intentionally_different=True)
