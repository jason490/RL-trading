"""Predeclared critic gates reject deterioration, missing controls and mismatch."""

import copy

import pytest

from rl_stocks.reporting.distributional_critics import historical_reproduction, screen_gate
from rl_stocks.training.distributional_critics import protocol


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

    return dict(sac_native=cell(0.28), tqc_untruncated=cell(0.28), tqc_truncated=cell(0.31))


def test_screen_requires_both_controls_absolute_and_reproduction():
    data = cells()
    assert screen_gate(data, dict(passed=True))["passed"]
    assert not screen_gate(data, dict(passed=False))["passed"]
    data["tqc_untruncated"]["final"]["score"] = 0.305
    assert not screen_gate(data, dict(passed=True))["passed"]
    data = cells()
    data["tqc_truncated"]["selected"]["groups"]["etf"]["excess_return"] = -0.021
    assert not screen_gate(data, dict(passed=True))["passed"]
    data = cells()
    data["tqc_truncated"]["final"]["mean_max_drawdown"] = -0.121
    assert not screen_gate(data, dict(passed=True))["passed"]


def test_screen_missing_control_and_nonfinite_reject():
    data = cells()
    del data["sac_native"]
    with pytest.raises(ValueError):
        screen_gate(data, dict(passed=True))
    data = cells()
    data["tqc_truncated"]["final"]["score"] = float("nan")
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
