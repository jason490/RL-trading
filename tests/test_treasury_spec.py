"""Single fixed information hypothesis, unchanged optimization and strict gates."""

from copy import deepcopy

import pytest

from rl_stocks.training.treasury_spec import binding, parameters, protocol
from rl_stocks.training.vix_spec import parameters as vix_parameters
from rl_stocks.reporting.history import screen_gate


def cells():
    def cell(score):
        return {
            p: dict(
                score=score,
                mean_max_drawdown=-0.1,
                groups={
                    g: dict(
                        net_return=score,
                        excess_return=score - 0.3,
                        median_excess_return=score - 0.3,
                    )
                    for g in ["stock", "etf"]
                },
            )
            for p in ["selected", "final"]
        }

    return dict(legacy=cell(0.28), masked=cell(0.28), full=cell(0.31))


def test_parameters_identical_original_control_no_model_construction():
    for arm in ["legacy", "masked", "full"]:
        for smoke in [False, True]:
            assert parameters(arm, smoke) == vix_parameters(smoke)
    spec = protocol()
    data = binding()
    assert spec["parameters"]["gae_lambda"] == 0.95 and spec["parameters"]["gamma"] == 0.99
    assert (
        spec["steps_per_cell"] == 100000
        and spec["execution"]["maximum_incremental_study_units"] == 2
    )
    assert (
        data["dataset_contracts"]["full"]["eligibility_sha256"]
        == data["dataset_contracts"]["masked"]["eligibility_sha256"]
    )
    assert data["dataset_contracts"]["full"]["training_rows"] == 282666
    assert spec["training_readiness"]["validation_rows"] == 19058
    assert spec["shared_initialization"]["parameter_counts"] == dict(legacy=61513, augmented=62537)
    assert (
        spec["execution"]["project_ceiling"] == 150
        and spec["execution"]["minimum_account_balance"] == 150
    )
    assert (
        spec["execution"]["project_teardown"] == 130
        and spec["execution"]["account_teardown_balance"] == 170
    )


def test_both_controls_absolute_median_risk_and_both_phases_required():
    spec = protocol()
    data = cells()
    assert screen_gate(data, spec)["passed"]
    for arm in ["legacy", "masked"]:
        changed = deepcopy(data)
        changed[arm]["final"]["score"] = 0.305
        assert not screen_gate(changed, spec)["passed"]
    for phase in ["selected", "final"]:
        changed = deepcopy(data)
        changed["full"][phase]["groups"]["etf"]["excess_return"] = -0.021
        assert not screen_gate(changed, spec)["passed"]
    changed = deepcopy(data)
    changed["full"]["final"]["mean_max_drawdown"] = -0.121
    assert not screen_gate(changed, spec)["passed"]
    changed = deepcopy(data)
    changed["full"]["selected"]["groups"]["stock"]["median_excess_return"] = -0.021
    assert not screen_gate(changed, spec)["passed"]
    assert not screen_gate(data, spec)["objective_achieved"]


@pytest.mark.parametrize("arm,smoke", [("other", False), ("full", 1)])
def test_no_unregistered_arms_or_boolean_aliases(arm, smoke):
    with pytest.raises(ValueError):
        parameters(arm, smoke)
