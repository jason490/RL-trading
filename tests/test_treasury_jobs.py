from copy import deepcopy

import pytest

from rl_stocks.runtime.colab.jobs import treasury_plan, validate_plan
from rl_stocks.training.treasury_spec import protocol
from test_treasury_runtime import accepted_profile


@pytest.mark.parametrize("arm", list(protocol()["arms"]))
def test_actual_bound_smoke_and_gated_full_plans(tmp_path, arm):
    smoke = treasury_plan(arm, smoke=True)
    validate_plan(smoke)
    assert smoke["version"] == 36 and smoke["jobs"][0]["steps"] == 512
    assert smoke["jobs"][0]["algorithm"] == "recurrent_ppo"
    with pytest.raises(ValueError):
        treasury_plan(arm)
    full = treasury_plan(arm, profile=accepted_profile(tmp_path))
    validate_plan(full)
    assert full["jobs"][0]["steps"] == 100000


@pytest.mark.parametrize(
    "field", ["algorithm", "steps", "overrides", "initialization", "boolean_alias"]
)
def test_plan_semantic_tampering_rejected(field):
    plan = deepcopy(treasury_plan("full", smoke=True))
    if field == "algorithm":
        plan["jobs"][0][field] = "sac"
    if field == "steps":
        plan["jobs"][0][field] = 2048
    if field == "overrides":
        plan["jobs"][0][field]["n_epochs"] = 1
    if field == "boolean_alias":
        plan["jobs"][0]["overrides"]["n_steps"] = True
    if field == "initialization":
        plan["initialization_contract"]["rng_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_plan(plan)



@pytest.mark.parametrize("seed", [23, 37, True])
def test_smoke_seed_cannot_expand_search(seed):
    with pytest.raises(ValueError):
        treasury_plan("full", seed=seed, smoke=True)


@pytest.mark.parametrize("field", ["binding_sha256", "dataset_contract", "protocol_sha256"])
def test_fixed_data_and_protocol_cannot_change(field):
    plan = treasury_plan("masked", smoke=True)
    if field == "dataset_contract":
        plan[field]["identity"] = "0" * 64
    else:
        plan[field] = "0" * 64
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_augmented_plans_share_only_initializer_and_eligible_bounds():
    full = treasury_plan("full", smoke=True)
    masked = treasury_plan("masked", smoke=True)
    legacy = treasury_plan("legacy", smoke=True)
    assert full["initialization_contract"] == masked["initialization_contract"]
    assert full["dataset_contract"]["market_features"] == 9
    assert legacy["dataset_contract"]["market_features"] == 7
    assert full["dataset_contract"]["identity"] != masked["dataset_contract"]["identity"]
