"""Predeclared VIX gates reject deterioration, missing controls and mismatch."""

import copy

import pytest

from rl_stocks.reporting.vix_context import historical_reproduction, screen_gate
from rl_stocks.training.vix_spec import preregistration as protocol


def cells():
    def cell(score):
        return {
            phase: dict(
                score=score,
                mean_max_drawdown=-0.1,
                groups={
                    g: dict(
                        net_return=score,
                        excess_return=score - 0.3,
                        median_excess_return=score - 0.3,
                    )
                    for g in ("stock", "etf")
                },
            )
            for phase in ("selected", "final")
        }

    return dict(legacy=cell(0.28), masked=cell(0.28), full=cell(0.31))


def test_screen_requires_both_controls_absolute_and_reproduction():
    data = cells()
    assert screen_gate(data, dict(passed=True))["passed"]
    assert not screen_gate(data, dict(passed=False))["passed"]
    data["masked"]["final"]["score"] = 0.305
    assert not screen_gate(data, dict(passed=True))["passed"]
    data = cells()
    data["full"]["selected"]["groups"]["etf"]["excess_return"] = -0.021
    assert not screen_gate(data, dict(passed=True))["passed"]
    data = cells()
    data["full"]["final"]["mean_max_drawdown"] = -0.121
    assert not screen_gate(data, dict(passed=True))["passed"]


def test_screen_missing_control_and_nonfinite_reject():
    data = cells()
    del data["legacy"]
    with pytest.raises(ValueError):
        screen_gate(data, dict(passed=True))
    data = cells()
    data["full"]["final"]["score"] = float("nan")
    with pytest.raises(ValueError):
        screen_gate(data, dict(passed=True))


def historical():
    spec = protocol()["historical_control_reproduction"]
    m = dict(
        git=dict(revision=spec["source_revision"]),
        selected_score=spec["selected_score"],
        validation_checkpoints=copy.deepcopy(spec["all_checkpoint_scores"]),
    )
    report = {str(i): dict(symbol=str(i), metrics=dict(net_return=0.1)) for i in range(26)}
    return m, dict(selected=report, final=copy.deepcopy(report))


def test_reproduction_fixed_tolerances_and_checkpoint_phases():
    original, reports = historical()
    current = copy.deepcopy(original)
    changed = copy.deepcopy(reports)
    assert historical_reproduction(current, original, changed, reports)["passed"]
    current["validation_checkpoints"][0]["score"] += 0.000101
    assert not historical_reproduction(current, original, changed, reports)["passed"]
    current = copy.deepcopy(original)
    changed["selected"]["0"]["metrics"]["net_return"] += 0.000501
    assert not historical_reproduction(current, original, changed, reports)["passed"]
    current["validation_checkpoints"][-1]["phase"] = "on_step"
    with pytest.raises(ValueError):
        historical_reproduction(current, original, changed, reports)


def test_risk_and_stock_median_gates_cannot_be_bypassed_by_high_mean():
    data = cells()
    data["full"]["final"]["groups"]["stock"]["median_excess_return"] = -0.0201
    assert not screen_gate(data, dict(passed=True))["passed"]
    data = cells()
    data["masked"]["final"]["groups"]["etf"]["net_return"] = 0.32
    assert not screen_gate(data, dict(passed=True))["passed"]
    data = cells()
    verdict = screen_gate(data, dict(passed=False))
    assert verdict["performance_gates_passed"] and not verdict["interpretation_valid"]
    assert verdict["failure_closes_fixed_family"] and verdict["promoted_candidate"] is None
    assert not verdict["objective_achieved"] and verdict["prospective_confirmation_required"]


def test_historical_original_cannot_be_substituted_or_relabelled():
    original, reports = historical()
    current = copy.deepcopy(original)
    original["git"]["revision"] = "unregistered"
    with pytest.raises(ValueError):
        historical_reproduction(current, original, reports, reports)
    original, reports = historical()
    changed = copy.deepcopy(reports)
    changed["final"]["0"]["symbol"] = "different"
    with pytest.raises(ValueError):
        historical_reproduction(original, original, changed, reports)


def test_screen_report_refuses_running_fleet_before_reading_any_prices(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from test_vix_jobs import accepted_profile

    accepted_profile(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "report_vix", Path(__file__).parents[1] / "scripts/report_vix_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state, source = tmp_path / "state.json", tmp_path / "source.json"
    state.write_text(json.dumps(dict(status="running", cleanup_verified=False)))
    source.write_text("{}")
    with pytest.raises(ValueError, match="owned-cleaned"):
        module.report(state, source, tmp_path)
